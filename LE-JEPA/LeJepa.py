import copy
import torch
import torch.nn as nn

from Encoder import Encoder
from augmentations import AugmentationPipeline
from Training import (
    compute_lejepa_loss,
    evaluate,
    save_model,
    train_and_evaluate,
)
from Forecasting import forcasting_zeroshot
from Classification import classification_zeroshot


class LeJEPA(nn.Module):
    """
    LE-JEPA: Learning with Empirical Joint-Embedding Predictive Architecture.

    Architecture:
      - Single shared encoder f_θ (no EMA, no predictor, no stop-gradient)
      - Two augmented views of the same window → encode both → MSE + SIGReg

    Loss:
      (1 - λ) * pred_loss  +  λ * sigreg
      where pred_loss = MSE(z1, z2) over all (B*C, n_patches, D) elements.
    """

    def __init__(
        self,
        config: dict,
        input_dim: int,       # patch_size (P_L)
        num_patches: int,     # ratio_patches
        train_loader,
        val_loader,
        test_loader,
    ):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = Encoder(
            num_patches=num_patches,
            dim_in=input_dim,
            embed_dim=config["encoder_embed_dim"],
            nhead=config["nhead"],
            num_layers=config["num_encoder_layers"],
            mlp_ratio=config["mlp_ratio"],
            drop_rate=config["drop_rate"],
            attn_drop_rate=config["attn_drop_rate"],
            pe='sincos',
            learn_pe=False,
            res_attention=True,
        )

        # ── Augmentation pipelines — two views, different DWT modes (like DINO) ─
        # view 1: smooth global view (soft_threshold)
        # view 2: perturbed local view (high_perturb)
        self.augment_v1 = AugmentationPipeline(config, dwt_mode=config.get("view1_dwt_mode", "soft_threshold"))
        self.augment_v2 = AugmentationPipeline(config, dwt_mode=config.get("view2_dwt_mode", "high_perturb"))

        # ── Optimizer: AdamW or SGD — controlled by config["optimizer"] ─────
        opt_name = config.get("optimizer", "adamw").lower()
        if opt_name == "sgd":
            lr = config.get("lr_sgd", 3e-3)
            self.optimizer = torch.optim.SGD(
                self.encoder.parameters(),
                lr=lr,
                weight_decay=config["weight_decay"],
                momentum=config.get("momentum", 0.9),
                nesterov=True,
            )
        else:
            lr = config.get("lr_adamw", 5e-4)
            self.optimizer = torch.optim.AdamW(
                self.encoder.parameters(),
                lr=lr,
                weight_decay=config["weight_decay"],
            )

        # ── Scheduler: linear warmup + cosine annealing (OneCycleLR) ─────────
        steps_per_epoch    = len(train_loader) if train_loader is not None else 1
        self.total_steps   = config["num_epochs"] * steps_per_epoch
        self.warmup_epochs = max(1, int(config.get("warmup_ratio", 0.05) * config["num_epochs"]))

        if train_loader is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=config["num_epochs"],
                steps_per_epoch=steps_per_epoch,
                pct_start=config.get("warmup_ratio", 0.05),
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0,
            )
        else:
            self.scheduler = None

        self.path_save    = config["path_save"]
        self.best_encoder = None

        # Forecasting attributes — set by run_lejepa before calling forcasting_zeroshot
        self.forcast_train = None
        self.forcast_val   = None
        self.forcast_test  = None
        self.epoch_t       = config.get("epoch_t", 70)


# Bind methods
LeJEPA.compute_lejepa_loss = compute_lejepa_loss
LeJEPA.evaluate            = evaluate
LeJEPA.save_model          = save_model
LeJEPA.train_and_evaluate  = train_and_evaluate
LeJEPA.forcasting_zeroshot    = forcasting_zeroshot
LeJEPA.classification_zeroshot = classification_zeroshot
