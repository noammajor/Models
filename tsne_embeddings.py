#!/usr/bin/env python3
"""
tsne_embeddings.py — t-SNE visualization of time-series classification embeddings.

For each model (default: all 6), loads the pretrained backbone from the
classification checkpoint (trained with num_patches=72), encodes all samples
from the specified dataset(s), and produces a t-SNE scatter plot colored by
class label. One figure per model, one subplot per dataset.

Usage:
    python tsne_embeddings.py --datasets EthanolConcentration Heartbeat
    python tsne_embeddings.py --datasets JapaneseVowels --models lejepa npt
    python tsne_embeddings.py --datasets Handwriting --seed 2003 --encoder_layers 8
    python tsne_embeddings.py --datasets FaceDetection PEMS-SF --output_dir plots/
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

ROOT        = Path(__file__).parent.resolve()
PATCH_SIZE  = 16
NUM_PATCHES = 72       # same as CLS_NUM_PATCHES in run_seed_analysis.py
CW          = NUM_PATCHES * PATCH_SIZE  # 1152

ALL_MODELS = ["dino", "jepa_simple", "lejepa", "patchtst", "npt", "timedart"]

DEFAULT_CLS_DIR = "/home/shared/datasets/Classification_TS"


# ── helpers ───────────────────────────────────────────────────────────────────

def _add_path(*dirs):
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load_config(path):
    spec = importlib.util.spec_from_file_location("_cfg", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.config)


# ── data loading ──────────────────────────────────────────────────────────────

def _get_combined_loader(dataset_name: str, cls_dir: str, patch_size: int,
                         num_patches: int, batch_size: int = 64):
    """Build a combined train+test DataLoader. Returns (loader, n_classes)."""
    _add_path(ROOT / "Discrete_JEPA")
    from data_loaders.data_puller import make_uea_dataloaders, ClassificationDataPuller
    import torch.utils.data
    import torch.nn.functional as F

    target_T = num_patches * patch_size

    def _patch_collate(batch):
        """UEA batches: (x [T,C], label, orig_len) -> (patches [B,P,PL,C], labels, mask)"""
        xs, ys, orig_lens = zip(*batch)
        orig_lens = torch.stack(orig_lens)
        max_t = max(x.shape[0] for x in xs)
        xs = torch.stack([F.pad(x, (0, 0, 0, max_t - x.shape[0])) for x in xs])
        B_, T_, C_ = xs.shape
        if T_ != target_T:
            idx = torch.linspace(0, T_ - 1, target_T).long()
            xs  = xs[:, idx, :]
            patch_idx    = idx[torch.arange(num_patches) * patch_size]
            padding_mask = patch_idx.unsqueeze(0) < orig_lens.unsqueeze(1)
        else:
            patch_starts = torch.arange(num_patches) * patch_size
            padding_mask = patch_starts.unsqueeze(0) < orig_lens.unsqueeze(1)
        xs = xs.reshape(B_, num_patches, patch_size, C_)
        return xs, torch.stack(ys), padding_mask

    dataset_path = Path(cls_dir) / dataset_name
    if list(dataset_path.glob("*_TRAIN.ts")):
        tr_raw, _, te_raw, n_classes = make_uea_dataloaders(cls_dir, dataset_name, batch_size)
        combined = torch.utils.data.ConcatDataset([tr_raw.dataset, te_raw.dataset])
        loader   = torch.utils.data.DataLoader(
            combined, batch_size=batch_size, shuffle=False,
            collate_fn=_patch_collate)
    else:
        tr = ClassificationDataPuller(cls_dir, dataset_name, patch_size, which="train")
        te = ClassificationDataPuller(cls_dir, dataset_name, patch_size, which="test")
        n_classes = tr.n_classes
        combined  = torch.utils.data.ConcatDataset([tr, te])
        loader    = torch.utils.data.DataLoader(combined, batch_size=batch_size, shuffle=False)

    return loader, n_classes


# ── checkpoint path resolution ────────────────────────────────────────────────

def _ckpt_path(model: str, encoder_layers: int, seed: int, pretrain_source: str) -> Path:
    src       = pretrain_source
    _src_tag  = f"_{src.replace('+', '_')}" if src != 'monash' else ''
    _seed_tag = f'_seed{seed}' if seed is not None else ''

    if model == "dino":
        return (ROOT / "classification" /
                f"checkpoints{_src_tag}_layers{encoder_layers}{_seed_tag}_cw{CW}" /
                "checkpoint_best.pth")

    if model == "jepa_simple":
        return (ROOT / "output_model" / "classification" /
                f"JEPA{_src_tag}_layers{encoder_layers}{_seed_tag}_cw{CW}" /
                "best_model.pt")

    if model == "lejepa":
        return (ROOT / "output_model" / "classification" /
                f"LE-JEPA{_src_tag}_layers{encoder_layers}{_seed_tag}_cw{CW}" /
                "best_model.pt")

    if model == "npt":
        npt_dir = ROOT / "NPT"
        _add_path(npt_dir)
        cfg = _load_config(npt_dir / "config_ntp.py")
        cfg['n_layers']            = encoder_layers
        cfg['pretrained_model_id'] = encoder_layers
        cfg['context_patches']     = NUM_PATCHES
        cfg['ratio_patches']       = NUM_PATCHES + cfg.get('horizon_t', 6)
        spec2 = importlib.util.spec_from_file_location("ntp_pretrain", npt_dir / "ntp_pretrain.py")
        mod2  = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(mod2)
        fname = mod2._model_fname(cfg, src)
        return (npt_dir / "saved_models" / "classification" / src / "ntp" /
                f"layers{encoder_layers}_cw{CW}{_seed_tag}" / f"{fname}.pt")

    if model == "patchtst":
        patchtst_dir = ROOT / "PatchTST_self_supervised"
        _add_path(patchtst_dir)
        cfg = _load_config(patchtst_dir / "config_patchtst.py")
        cfg['n_layers']            = encoder_layers
        cfg['pretrained_model_id'] = encoder_layers
        cfg['context_points']      = NUM_PATCHES * cfg.get('patch_len', PATCH_SIZE)
        n_ep    = cfg.get("n_epochs_pretrain", 10)
        ctx     = cfg.get("context_points", 512)
        p_len   = cfg.get("patch_len", 12)
        stride  = cfg.get("stride", 12)
        m_ratio = cfg.get("mask_ratio", 0.4)
        m_id    = cfg.get("pretrained_model_id", 1)
        fname   = (f"patchtst_pretrained_cw{ctx}_patch{p_len}_stride{stride}"
                   f"_epochs-pretrain{n_ep}_mask{m_ratio}_model{m_id}.pth")
        mtype   = cfg.get("model_type", "based_model")
        return (patchtst_dir / "saved_models" / "classification" / src /
                "masked_patchtst" / mtype / f"layers{encoder_layers}_cw{CW}" / fname)

    if model == "timedart":
        cfg = _load_config(ROOT / "TimeDART-main" / "config_timedart.py")
        return (ROOT / f"outputs/timedart_pretrain{_src_tag}_layers{cfg['e_layers']}" /
                f"monash{_src_tag}" / "ckpt_best.pth")

    raise ValueError(f"Unknown model: {model}")


# ── per-model embedding extraction ────────────────────────────────────────────

@torch.no_grad()
def _extract_dino(ckpt: Path, loader, device) -> tuple:
    """DINO encoder. Returns (embeddings [N, d], labels [N])."""
    _add_path(ROOT / "TSDiNO", ROOT / "TSDiNO" / "models")

    sample_patches, _, _ = next(iter(loader))
    n_v = sample_patches.shape[-1]
    n_p = sample_patches.shape[1]

    from models.patchTST import PatchTST as DinoPatchTST
    cfg   = _load_config(ROOT / "TSDiNO" / "config.py")
    model = DinoPatchTST(
        c_in=n_v,
        target_dim=cfg.get("out_dim", 256),
        patch_len=PATCH_SIZE,
        stride=PATCH_SIZE,
        num_patch=n_p,
        n_layers=cfg.get("n_layers", 3),
        n_heads=cfg.get("n_heads", 16),
        d_model=cfg.get("embed_dim", 128),
        shared_embedding=True,
        d_ff=cfg.get("d_ff", 256),
        dropout=cfg.get("dropout", 0.0),
        head_dropout=cfg.get("head_dropout", 0.0),
        act="gelu",
        head_type="Dino",
        res_attention=False,
        step_size=PATCH_SIZE,
    ).to(device)

    if ckpt.exists():
        raw    = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd     = raw.get("student", raw)
        md     = model.state_dict()
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("module.", "")
            if nk.startswith("backbone."):
                nk = nk[len("backbone."):]
            if "head" in k.split(".")[-2:]:
                continue
            if nk in md and md[nk].shape == v.shape:
                new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
        print(f"  [dino] loaded {len(new_sd)}/{len(md)} params from {ckpt.name}")
    else:
        print(f"  [dino] WARNING: checkpoint not found at {ckpt}")

    model.eval()

    all_embs, all_labels = [], []
    for patches, labels, padding_mask in loader:
        B, P, PL, C = patches.shape
        x  = patches.permute(0, 1, 3, 2).float().to(device)  # [B, P, C, PL]
        pm = padding_mask.to(device)
        z = model.backbone(x, padding_mask=pm)   # [B, C, d_model, P]
        B_, C_, D_, P_ = z.shape
        z = z.reshape(B_, C_ * D_ * P_)          # [B, C*d_model*P]
        all_embs.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


@torch.no_grad()
def _extract_jepa(ckpt: Path, loader, encoder_layers: int, device,
                  model_name: str = "jepa_simple") -> tuple:
    """JEPA / jepa_simple encoder. Returns (embeddings [N, d], labels [N])."""
    jepa_dir  = ROOT / "JEPA"
    djepa_dir = ROOT / "Discrete_JEPA"
    _add_path(str(jepa_dir / "JEPA"), str(jepa_dir), str(djepa_dir))

    # Load by absolute path to avoid module-cache conflicts with LE-JEPA's Encoder.py
    _enc_spec = importlib.util.spec_from_file_location(
        "jepa_encoder", jepa_dir / "JEPA" / "Encoder.py")
    _enc_mod  = importlib.util.module_from_spec(_enc_spec)
    _enc_spec.loader.exec_module(_enc_mod)
    JepaEncoder = _enc_mod.Encoder

    cfg = _load_config(jepa_dir / "config_files" / "config_jepa.py")
    cfg['num_encoder_layers'] = encoder_layers
    embed_dim = cfg["encoder_embed_dim"]

    encoder = JepaEncoder(
        num_patches   = NUM_PATCHES,
        dim_in        = PATCH_SIZE,
        embed_dim     = embed_dim,
        nhead         = cfg["nhead"],
        num_layers    = encoder_layers,
        mlp_ratio     = cfg["mlp_ratio"],
        drop_rate     = cfg["drop_rate"],
        attn_drop_rate = cfg["attn_drop_rate"],
        pe            = 'sincos',
        learn_pe      = False,
        res_attention = True,
    ).to(device)

    if ckpt.exists():
        raw    = torch.load(ckpt, map_location="cpu")
        enc_sd = raw["target_encoder"]
        encoder.load_state_dict(enc_sd, strict=False)
        print(f"  [{model_name}] loaded encoder from {ckpt.name}")
    else:
        print(f"  [{model_name}] WARNING: checkpoint not found at {ckpt}")

    encoder.eval()

    def _instance_norm(x, eps=1e-6):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2),  keepdim=True) + eps
        return (x - mean) / std, mean, std

    all_embs, all_labels = [], []
    for patches, labels, padding_mask in loader:
        patches      = patches.float().to(device)
        padding_mask = padding_mask.to(device)
        B, P, PL, C  = patches.shape
        ctx_norm, _, _ = _instance_norm(patches)
        out = encoder(ctx_norm, padding_mask=padding_mask)
        enc = out["data_patches"]            # [B*C, P, embed_dim]
        enc = enc.reshape(B, C * P * embed_dim)  # [B, C*P*embed_dim]
        all_embs.append(enc.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


@torch.no_grad()
def _extract_lejepa(ckpt: Path, loader, encoder_layers: int, device) -> tuple:
    """LE-JEPA encoder. Returns (embeddings [N, d], labels [N])."""
    lejepa_dir = ROOT / "LE-JEPA"
    djepa_dir  = ROOT / "Discrete_JEPA"
    _add_path(str(lejepa_dir), str(djepa_dir))

    # Load by absolute path to avoid module-cache conflicts with JEPA's Encoder.py
    _enc_spec = importlib.util.spec_from_file_location(
        "lejepa_encoder", lejepa_dir / "Encoder.py")
    _enc_mod  = importlib.util.module_from_spec(_enc_spec)
    _enc_spec.loader.exec_module(_enc_mod)
    LeJepaEncoder = _enc_mod.Encoder

    cfg = _load_config(lejepa_dir / "config_lejepa.py")
    cfg['num_encoder_layers'] = encoder_layers
    embed_dim = cfg["encoder_embed_dim"]

    encoder = LeJepaEncoder(
        num_patches   = NUM_PATCHES,
        dim_in        = PATCH_SIZE,
        embed_dim     = embed_dim,
        nhead         = cfg["nhead"],
        num_layers    = encoder_layers,
        mlp_ratio     = cfg["mlp_ratio"],
        drop_rate     = cfg["drop_rate"],
        attn_drop_rate = cfg["attn_drop_rate"],
        pe            = 'sincos',
        learn_pe      = False,
        res_attention = True,
    ).to(device)

    if ckpt.exists():
        raw   = torch.load(ckpt, map_location="cpu")
        enc_sd = raw["encoder"]
        encoder.load_state_dict(enc_sd, strict=False)
        print(f"  [lejepa] loaded encoder from {ckpt.name}")
    else:
        print(f"  [lejepa] WARNING: checkpoint not found at {ckpt}")

    encoder.eval()

    def _instance_norm(x, eps=1e-6):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2),  keepdim=True) + eps
        return (x - mean) / std, mean, std

    all_embs, all_labels = [], []
    for patches, labels, padding_mask in loader:
        patches      = patches.float().to(device)
        padding_mask = padding_mask.to(device)
        B, P, PL, C  = patches.shape
        ctx_norm, _, _ = _instance_norm(patches)
        out = encoder(ctx_norm, padding_mask=padding_mask)
        enc = out["data_patches"]            # [B*C, P, embed_dim]
        enc = enc.reshape(B, C * P * embed_dim)  # [B, C*P*embed_dim]
        all_embs.append(enc.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


@torch.no_grad()
def _extract_npt(ckpt: Path, loader, encoder_layers: int, device) -> tuple:
    """NPT (PatchTST causal) encoder. Returns (embeddings [N, d], labels [N])."""
    npt_dir   = ROOT / "NPT"
    djepa_dir = ROOT / "Discrete_JEPA"
    _add_path(str(npt_dir), str(djepa_dir))

    from models.patchTST import PatchTST

    cfg     = _load_config(npt_dir / "config_ntp.py")
    d_model = cfg["d_model"]

    sample_patches, _, _ = next(iter(loader))
    num_patch = sample_patches.shape[1]
    n_v       = sample_patches.shape[-1]

    backbone = PatchTST(
        c_in=n_v,
        target_dim=cfg["patch_size"],
        patch_len=cfg["patch_size"],
        stride=cfg["patch_size"],
        num_patch=num_patch,
        n_layers=encoder_layers,
        n_heads=cfg["n_heads"],
        d_model=d_model,
        shared_embedding=True,
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        head_dropout=cfg["head_dropout"],
        act=cfg["act"],
        head_type="pretrain",
        causal=True,
        res_attention=False,
    ).to(device)

    if ckpt.exists():
        raw      = torch.load(ckpt, map_location="cpu")
        state    = raw.get("encoder", raw.get("model", raw))
        enc_dict = backbone.backbone.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in enc_dict and enc_dict[k].shape == v.shape}
        enc_dict.update(filtered)
        backbone.backbone.load_state_dict(enc_dict)
        print(f"  [npt] loaded {len(filtered)}/{len(enc_dict)} params from {ckpt.name}")
    else:
        print(f"  [npt] WARNING: checkpoint not found at {ckpt}")

    backbone.eval()

    all_embs, all_labels = [], []
    for patches, labels, padding_mask in loader:
        x            = patches.permute(0, 1, 3, 2).float().to(device)  # [B,P,C,PL]
        padding_mask = padding_mask.to(device)
        z = backbone.backbone(x, padding_mask=padding_mask)  # [B, C, d_model, P]
        B_, C_, D_, P_ = z.shape
        z = z.reshape(B_, C_ * D_ * P_)                      # [B, C*d_model*P]
        all_embs.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


@torch.no_grad()
def _extract_patchtst(ckpt: Path, loader, encoder_layers: int, device) -> tuple:
    """PatchTST (self-supervised) encoder. Returns (embeddings [N, d], labels [N])."""
    patchtst_dir = ROOT / "PatchTST_self_supervised"
    djepa_dir    = ROOT / "Discrete_JEPA"
    _add_path(str(patchtst_dir), str(djepa_dir))

    from src.models.patchTST import PatchTST

    cfg = _load_config(patchtst_dir / "config_patchtst.py")

    sample_patches, _, _ = next(iter(loader))
    num_patch = sample_patches.shape[1]
    n_v       = sample_patches.shape[-1]
    patch_len = sample_patches.shape[2]

    model = PatchTST(
        c_in=n_v,
        target_dim=1,
        patch_len=patch_len,
        stride=patch_len,
        num_patch=num_patch,
        n_layers=encoder_layers,
        n_heads=cfg.get("n_heads", 16),
        d_model=cfg.get("d_model", 128),
        shared_embedding=True,
        d_ff=cfg.get("d_ff", 512),
        dropout=cfg.get("dropout", 0.2),
        head_dropout=cfg.get("head_dropout", 0.2),
        act="gelu",
        head_type="pretrain",
        res_attention=False,
    ).to(device)

    if ckpt.exists():
        raw      = torch.load(ckpt, map_location="cpu")
        state    = raw.get("model", raw)
        md       = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in md and md[k].shape == v.shape and "head" not in k}
        md.update(filtered)
        model.load_state_dict(md)
        print(f"  [patchtst] loaded {len(filtered)}/{len(md)} params from {ckpt.name}")
    else:
        print(f"  [patchtst] WARNING: checkpoint not found at {ckpt}")

    model.eval()

    all_embs, all_labels = [], []
    for patches, labels, padding_mask in loader:
        x            = patches.permute(0, 1, 3, 2).float().to(device)  # [B,P,C,PL]
        padding_mask = padding_mask.to(device)
        z = model.backbone(x, padding_mask=padding_mask)  # [B, C, d_model, P]
        B_, C_, D_, P_ = z.shape
        z = z.reshape(B_, C_ * D_ * P_)                    # [B, C*d_model*P]
        all_embs.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


@torch.no_grad()
def _extract_timedart(ckpt: Path, loader, encoder_layers: int, device) -> tuple:
    """TimeDaRT encoder. Returns (embeddings [N, d], labels [N])."""
    td_dir    = ROOT / "TimeDART-main"
    djepa_dir = ROOT / "Discrete_JEPA"
    _add_path(str(td_dir), str(td_dir / "models"), str(djepa_dir))
    # Remove any cached 'layers' / 'models' from other models so TimeDaRT's are used
    import sys as _sys
    for _k in list(_sys.modules.keys()):
        if _k == "layers" or _k.startswith("layers.") or _k == "models" or _k.startswith("models."):
            del _sys.modules[_k]

    _td_model_spec = importlib.util.spec_from_file_location(
        "timedart_model", td_dir / "models" / "TimeDART.py")
    _td_model_mod  = importlib.util.module_from_spec(_td_model_spec)
    _td_model_spec.loader.exec_module(_td_model_mod)
    Model = _td_model_mod.Model

    _td_tools_spec = importlib.util.spec_from_file_location(
        "timedart_tools", td_dir / "utils" / "tools.py")
    _td_tools_mod  = importlib.util.module_from_spec(_td_tools_spec)
    _td_tools_spec.loader.exec_module(_td_tools_mod)
    transfer_weights = _td_tools_mod.transfer_weights

    cfg = _load_config(td_dir / "config_timedart.py")
    from types import SimpleNamespace

    sample_patches, _, _ = next(iter(loader))
    n_v = sample_patches.shape[-1]

    patch_len = cfg.get("patch_len", PATCH_SIZE)
    stride    = cfg.get("stride", patch_len)

    args = SimpleNamespace(
        input_len    = 512,
        d_model      = cfg.get("d_model", 256),
        n_heads      = cfg.get("n_heads", 8),
        d_ff         = cfg.get("d_ff", 512),
        dropout      = cfg.get("dropout", 0.1),
        head_dropout = cfg.get("head_dropout", 0.1),
        device       = device,
        task_name    = "pretrain",
        pred_len     = 0,
        use_norm     = False,
        patch_len    = patch_len,
        stride       = stride,
        time_steps   = cfg.get("time_steps", 1000),
        scheduler    = cfg.get("scheduler", "cosine"),
        mask_ratio   = cfg.get("mask_ratio", 1.0),
        e_layers     = encoder_layers,
        d_layers     = cfg.get("d_layers", 1),
        enc_in       = n_v,
        dec_in       = n_v,
        c_out        = n_v,
    )

    model = Model(args).float().to(device)

    if ckpt.exists():
        model = transfer_weights(str(ckpt), model, exclude_head=True, device=str(device))
        print(f"  [timedart] loaded checkpoint from {ckpt.name}")
    else:
        print(f"  [timedart] WARNING: checkpoint not found at {ckpt}")

    model.eval()
    d_model = cfg.get("d_model", 256)

    def _encode(x, padding_mask=None):
        """x: [B,T,C] -> [B, C, d_model, P]"""
        B, T, C = x.shape
        xc  = model.channel_independence(x)          # [B*C, T, 1]
        xc  = model.patch(xc)                        # [B*C, P, patch_len]
        P   = xc.shape[1]
        xc  = model.enc_embedding(xc)                # [B*C, P, d_model]
        xc  = model.positional_encoding(xc)          # [B*C, P, d_model]
        kpm = None
        if padding_mask is not None:
            pm  = padding_mask.bool()
            kpm = (~pm).unsqueeze(1).expand(-1, C, -1).reshape(B * C, P)
        xc  = model.encoder(xc, is_mask=False, key_padding_mask=kpm)  # [B*C, P, d_model]
        xc  = xc.reshape(B, C, P, d_model)
        return xc.permute(0, 1, 3, 2)               # [B, C, d_model, P]

    all_embs, all_labels = [], []
    for patches, labels, padding_mask in loader:
        B, P, PL, C = patches.shape
        x            = patches.reshape(B, P * PL, C).float().to(device)
        padding_mask = padding_mask.to(device)
        z = _encode(x, padding_mask)          # [B, C, d_model, P]
        B_, C_, D_, P_ = z.shape
        z = z.reshape(B_, C_ * D_ * P_)       # [B, C*d_model*P]
        all_embs.append(z.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs), np.concatenate(all_labels)


# ── dispatch table ─────────────────────────────────────────────────────────────

_EXTRACTORS = {
    "dino":        _extract_dino,
    "jepa_simple": _extract_jepa,
    "lejepa":      _extract_lejepa,
    "npt":         _extract_npt,
    "patchtst":    _extract_patchtst,
    "timedart":    _extract_timedart,
}


# ── t-SNE + plotting ──────────────────────────────────────────────────────────

def _subsample(embeddings: np.ndarray, labels: np.ndarray, max_points: int):
    """Stratified subsample to max_points (equal per class)."""
    if len(embeddings) <= max_points:
        return embeddings, labels
    classes   = np.unique(labels)
    per_class = max(1, max_points // len(classes))
    idx = []
    rng = np.random.default_rng(42)
    for c in classes:
        c_idx   = np.where(labels == c)[0]
        chosen  = rng.choice(c_idx, min(per_class, len(c_idx)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    return embeddings[idx], labels[idx]


def _reduce(embeddings: np.ndarray, method: str = "tsne",
            perplexity: float = 30.0, n_iter: int = 1000,
            pca_components: int = 50) -> np.ndarray:
    """L2-normalize → optional PCA → tsne or umap → xy [N, 2]."""
    embs = normalize(embeddings, norm="l2")
    if pca_components > 0:
        n_components = min(pca_components, embs.shape[0] - 1, embs.shape[1])
        if n_components > 2:
            embs = PCA(n_components=n_components, random_state=42).fit_transform(embs)

    if method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=min(15, len(embs) - 1))
        return reducer.fit_transform(embs)
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                    random_state=42, init="pca")
        return tsne.fit_transform(embs)


MODEL_DISPLAY = {
    "jepa_simple": "JEPA",
    "lejepa":      "LE-JEPA",
    "dino":        "TSDiNO",
    "patchtst":    "PatchTST",
    "npt":         "NPT",
    "timedart":    "TimeDart",
}


def plot_combined(all_results: dict, output_dir: Path, datasets: list,
                  max_points: int = 500, method: str = "tsne",
                  pca_components: int = 50, perplexity: float = 50.0):
    """
    all_results: {model_name: {dataset_name: (embeddings, labels)}}
    Produces one figure per dataset, each with 6 model subplots in a row.
    """
    models = [m for m in MODEL_DISPLAY if m in all_results]

    pca_tag  = f"_pca{pca_components}" if pca_components > 0 else "_nopca"
    perp_tag = f"_perp{int(perplexity)}"

    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          12,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,
    })

    n_cols = 3
    n_rows = 2  # always 2 rows of 3

    for ds_name in datasets:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 5 * n_rows),
                                 squeeze=False)

        for idx, model_name in enumerate(models):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            results = all_results.get(model_name, {})
            if ds_name not in results:
                ax.set_visible(False)
                continue

            embs, labels = results[ds_name]
            n_cls = len(np.unique(labels))
            cmap  = plt.get_cmap("tab10" if n_cls <= 10 else "tab20")

            embs, labels = _subsample(embs, labels, max_points)
            print(f"  [combined] {method.upper()} {model_name}/{ds_name} …")
            xy = _reduce(embs, method=method, perplexity=perplexity,
                         pca_components=pca_components)

            for cls_id in np.unique(labels):
                mask = labels == cls_id
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           s=15, alpha=0.75, linewidths=0,
                           color=cmap(int(cls_id) % 20),
                           label=f"Class {int(cls_id)}")

            ax.set_title(MODEL_DISPLAY.get(model_name, model_name),
                         fontsize=14, fontweight="bold", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            # border around each subplot so models are clearly separated
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#cccccc")
            if n_cls <= 10:
                ax.legend(markerscale=1.2, fontsize=7, framealpha=0.8,
                          loc="best", ncol=2 if n_cls > 6 else 1,
                          handletextpad=0.3, borderpad=0.4)

        # hide any unused axes (if fewer than 6 models)
        for idx in range(len(models), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(ds_name, fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout(pad=2.0)
        out_file = output_dir / f"{method}{pca_tag}{perp_tag}_combined_{ds_name}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        print(f"\n  [combined] Saved → {out_file}")
        plt.close(fig)

    plt.rcParams.update(plt.rcParamsDefault)


def plot_model(model_name: str, dataset_results: dict, output_dir: Path,
               max_points: int = 500, method: str = "tsne", pca_components: int = 50,
               perplexity: float = 50.0):
    """
    dataset_results: {dataset_name: (embeddings [N,d], labels [N])}
    One figure per (model, dataset) — paper-ready styling.
    """
    pca_tag  = f"_pca{pca_components}" if pca_components > 0 else "_nopca"
    perp_tag = f"_perp{int(perplexity)}"

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      13,
        "axes.linewidth": 0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,
    })

    for ds_name, (embs, labels) in dataset_results.items():
        n_cls = len(np.unique(labels))
        cmap  = plt.get_cmap("tab10" if n_cls <= 10 else "tab20")

        embs, labels = _subsample(embs, labels, max_points)
        print(f"  [{model_name}] {method.upper()} on {ds_name}: {len(embs)} points, {n_cls} classes …")
        xy = _reduce(embs, method=method, perplexity=perplexity, pca_components=pca_components)

        fig, ax = plt.subplots(figsize=(5, 5))
        for cls_id in np.unique(labels):
            mask = labels == cls_id
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       s=20, alpha=0.75, linewidths=0,
                       color=cmap(int(cls_id) % 20),
                       label=f"Class {int(cls_id)}")

        ax.set_title(f"{ds_name}", fontsize=15, fontweight="bold", pad=10)
        ax.set_xlabel(f"{method.upper()} dim 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} dim 2", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        legend = ax.legend(markerscale=1.5, fontsize=9, framealpha=0.9,
                           loc="best", ncol=2 if n_cls > 6 else 1)
        legend.get_frame().set_linewidth(0.5)

        fig.tight_layout()
        out_file = output_dir / f"{method}{pca_tag}{perp_tag}_{model_name}_{ds_name}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        print(f"  [{model_name}] Saved → {out_file}")
        plt.close(fig)

    plt.rcParams.update(plt.rcParamsDefault)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE of frozen backbone embeddings per model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--datasets",        nargs="+", required=True,
                        help="Classification dataset names")
    parser.add_argument("--models",          nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,  metavar="MODEL",
                        help="Models to plot (default: all)")
    parser.add_argument("--encoder_layers",  type=int, default=8)
    parser.add_argument("--pretrain_source", type=str, default="monash+synthetic",
                        choices=["monash", "synthetic", "monash+synthetic"])
    parser.add_argument("--cls_dir",         type=str, default=DEFAULT_CLS_DIR,
                        help="Root directory for classification datasets")
    parser.add_argument("--output_dir",      type=str, default="plots",
                        help="Directory for output figures (default: plots/)")
    parser.add_argument("--max_points",      type=int, default=500,
                        help="Max points per dataset (stratified, default: 500)")
    parser.add_argument("--method",          type=str, default="tsne",
                        choices=["tsne", "umap"],
                        help="Dimensionality reduction method (default: tsne)")
    parser.add_argument("--pca_components",  type=int, default=50,
                        help="PCA dims before reduction (0 = skip PCA, default: 50)")
    parser.add_argument("--perplexity",      type=float, default=50.0,
                        help="t-SNE perplexity (default: 50)")
    parser.add_argument("--batch_size",      type=int, default=64)
    parser.add_argument("--gpu",             type=int, default=0)
    parser.add_argument("--combined",        action="store_true",
                        help="Save all models in one figure (rows=datasets, cols=models)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models:   {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Layers: {args.encoder_layers}  Source: {args.pretrain_source}\n")

    all_results = {}  # {model_name: {ds_name: (embs, labels)}}

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        ckpt = _ckpt_path(model_name, args.encoder_layers, None, args.pretrain_source)
        print(f"  Checkpoint: {ckpt}")

        extractor      = _EXTRACTORS[model_name]
        dataset_results = {}

        for ds_name in args.datasets:
            print(f"\n  → Dataset: {ds_name}")
            try:
                loader, n_classes = _get_combined_loader(
                    ds_name, args.cls_dir, PATCH_SIZE, NUM_PATCHES, args.batch_size)
                print(f"    {len(loader.dataset)} samples, {n_classes} classes")

                # call extractor with correct signature
                if model_name in ("dino",):
                    embs, labels = extractor(ckpt, loader, device)
                elif model_name == "timedart":
                    embs, labels = extractor(ckpt, loader, args.encoder_layers, device)
                elif model_name == "jepa_simple":
                    embs, labels = extractor(ckpt, loader, args.encoder_layers, device,
                                             model_name="jepa_simple")
                else:
                    embs, labels = extractor(ckpt, loader, args.encoder_layers, device)

                dataset_results[ds_name] = (embs, labels)
                print(f"    Embeddings: {embs.shape}")
            except Exception as e:
                import traceback
                print(f"    ERROR on {ds_name}: {e}")
                traceback.print_exc()

        if dataset_results:
            all_results[model_name] = dataset_results
            if not args.combined:
                plot_model(model_name, dataset_results, output_dir,
                           max_points=args.max_points, method=args.method,
                           pca_components=args.pca_components,
                           perplexity=args.perplexity)
        else:
            print(f"  [{model_name}] No datasets succeeded — skipping plot.")

    if args.combined and all_results:
        plot_combined(all_results, output_dir, args.datasets,
                      max_points=args.max_points, method=args.method,
                      pca_components=args.pca_components,
                      perplexity=args.perplexity)

    print(f"\nDone. Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
