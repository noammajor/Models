import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

# ── path setup ────────────────────────────────────────────────────────────────
_HYBRID_DIR = Path(__file__).parent.resolve()
_ROOT_DIR   = _HYBRID_DIR.parent
_LEJEPA_DIR = _ROOT_DIR / "LE-JEPA"
_NTP_DIR    = _ROOT_DIR / "NTP"
_SHARED_DIR = _ROOT_DIR / "shared"

# LE-JEPA must be first so plain-name imports (Encoder, augmentations, losses)
# resolve to LE-JEPA rather than any same-named module elsewhere.
for _p in [str(_LEJEPA_DIR), str(_NTP_DIR), str(_SHARED_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Encoder import Encoder
from models.patchTST import NTPHead   # from NTP/models/patchTST.py

# ── Load HybridAugmentations by explicit path ─────────────────────────────────
_aug_spec = importlib.util.spec_from_file_location(
    "hybrid_augmentations", _HYBRID_DIR / "HybridAugmentations.py")
_aug_mod  = importlib.util.module_from_spec(_aug_spec)
_aug_spec.loader.exec_module(_aug_mod)
HybridAugmentationPipeline = _aug_mod.HybridAugmentationPipeline

# ── Load HybridTraining by explicit path to avoid collision with LE-JEPA/Training.py ─
_training_spec = importlib.util.spec_from_file_location(
    "hybrid_training", _HYBRID_DIR / "HybridTraining.py")
_training_mod  = importlib.util.module_from_spec(_training_spec)
_training_spec.loader.exec_module(_training_mod)

compute_hybrid_loss = _training_mod.compute_hybrid_loss
evaluate            = _training_mod.evaluate
save_model          = _training_mod.save_model
train_and_evaluate  = _training_mod.train_and_evaluate

# ── Load downstream tasks from LE-JEPA (same encoder interface) ───────────────
_forecasting_spec = importlib.util.spec_from_file_location(
    "lejepa_forecasting", _LEJEPA_DIR / "Forecasting.py")
_forecasting_mod  = importlib.util.module_from_spec(_forecasting_spec)
_forecasting_spec.loader.exec_module(_forecasting_mod)
forecasting = _forecasting_mod.forecasting

_cls_spec = importlib.util.spec_from_file_location(
    "lejepa_classification", _LEJEPA_DIR / "Classification.py")
_cls_mod  = importlib.util.module_from_spec(_cls_spec)
_cls_spec.loader.exec_module(_cls_mod)
classification = _cls_mod.classification

_anom_spec = importlib.util.spec_from_file_location(
    "lejepa_anomaly", _LEJEPA_DIR / "Anomaly.py")
_anom_mod  = importlib.util.module_from_spec(_anom_spec)
_anom_spec.loader.exec_module(_anom_mod)
anomaly_detection = _anom_mod.anomaly_detection


class HybridModel(nn.Module):
    """
    Hybrid LEJEPA + NTP pretraining.

    Single shared encoder (LEJEPA-style, bidirectional) trained on two tasks:
        total_loss = φ · lejepa_loss + (1−φ) · ntp_loss

    LEJEPA task: two DWT-augmented views → MSE(z1, z2) + λ·SIGReg
    NTP task:    context patches → NTPHead → MSE vs horizon patches

    φ=1 → pure LEJEPA,  φ=0 → pure NTP.
    """

    def __init__(
        self,
        config: dict,
        input_dim: int,       # patch_size (P_L)
        num_patches: int,     # ratio_patches (total: context + horizon)
        train_loader,
        val_loader,
        test_loader,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.phi    = float(config['phi'])

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        # ── Shared encoder ────────────────────────────────────────────────────
        self.encoder = Encoder(
            num_patches    = num_patches,
            dim_in         = input_dim,
            embed_dim      = config['encoder_embed_dim'],
            nhead          = config['nhead'],
            num_layers     = config['num_encoder_layers'],
            mlp_ratio      = config['mlp_ratio'],
            drop_rate      = config['drop_rate'],
            attn_drop_rate = config['attn_drop_rate'],
            pe             = 'sincos',
            learn_pe       = False,
            res_attention  = True,
        )

        # ── Augmentation pipelines (LE-JEPA + TSDiNO union) ───────────────────
        self.augment_v1 = HybridAugmentationPipeline(
            config, dwt_mode=config.get('view1_dwt_mode', 'soft_threshold'))
        self.augment_v2 = HybridAugmentationPipeline(
            config, dwt_mode=config.get('view2_dwt_mode', 'high_perturb'))

        # ── NTP prediction head ────────────────────────────────────────────────
        d_ff = config.get('ntp_head_d_ff', 4 * config['encoder_embed_dim'])
        self.ntp_head = NTPHead(
            d_model         = config['encoder_embed_dim'],
            d_ff            = d_ff,
            patch_len       = input_dim,
            dropout         = config.get('head_dropout', 0.2),
            act             = config.get('act', 'gelu'),
            horizon_patches = config['horizon_t'],
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        lr = config.get('lr_adamw', 5e-4)
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.ntp_head.parameters()),
            lr           = lr,
            weight_decay = config['weight_decay'],
        )

        # ── Scheduler: linear warmup + cosine annealing ────────────────────────
        steps_per_epoch    = len(train_loader) if train_loader is not None else 1
        self.warmup_epochs = max(1, int(config.get('warmup_ratio', 0.05) * config['num_epochs']))

        if train_loader is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr           = lr,
                epochs           = config['num_epochs'],
                steps_per_epoch  = steps_per_epoch,
                pct_start        = config.get('warmup_ratio', 0.05),
                anneal_strategy  = 'cos',
                div_factor       = 10.0,
                final_div_factor = 100.0,
            )
        else:
            self.scheduler = None

        self.path_save    = config['path_save']
        self.best_encoder = None

        # Forecasting attributes — set by run_hybrid before calling forecasting
        self.forcast_train = None
        self.forcast_val   = None
        self.forcast_test  = None
        self.epoch_t       = config.get('epoch_t', 70)


# ── Bind methods ──────────────────────────────────────────────────────────────
HybridModel.compute_hybrid_loss = compute_hybrid_loss
HybridModel.evaluate            = evaluate
HybridModel.save_model          = save_model
HybridModel.train_and_evaluate  = train_and_evaluate
HybridModel.forecasting         = forecasting
HybridModel.classification      = classification
HybridModel.anomaly_detection   = anomaly_detection
