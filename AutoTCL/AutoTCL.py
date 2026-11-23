"""
AutoTCL pretrain helper.

Parametric augmentation for time-series contrastive learning (Zheng et al.,
ICLR 2024), re-implemented on the shared PatchTST transformer so it can be
compared with the other objectives under one protocol. The reference code is
in ../AutoTCL-main (AutoTCL_CoST.py, models/losses.py); this file keeps its
training procedure and losses and swaps only the encoder:

  encoder  f : (B, T, C) -> (B, P, C*D)   PatchTransformerWrapper (SoftCLT's)
  augmenter  : (B, T, C) -> h(x), g(x)    small dilated-conv net, per timestep
  view       : x' = g(x) * h(x) * x

Encoder step   : InfoNCE(maxpool f(x), maxpool f(x')) + local_weight * local InfoNCE
Augmenter step : MMD(f(x), f(x')) + reg_weight * relu(mean gate mass - thres)
                 + regular_weight * smoothness(h)          ("PRI" objective)
The two steps alternate (one augmenter update every `ratio_step` encoder
updates); an EMA copy of the encoder is what gets exported, as in the paper.

Handles, like SoftCLT.py:
  - data loading from Monash, synthetic, or CSV sources
  - pretraining
  - saving the PatchTSTEncoder backbone
"""

import sys
import os
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ── path wiring ───────────────────────────────────────────────────────────────
_AUTOTCL_DIR = Path(__file__).parent.resolve()
_REPO_ROOT   = _AUTOTCL_DIR.parent
_SHARED_DIR  = _REPO_ROOT / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

# The transformer wrapper lives in SoftCLT's tree. Load it by file so neither
# SoftCLT's nor DINO's top-level `models` package has to be importable here.
_WRAPPER_FILE = _REPO_ROOT / "SoftCLT" / "softclt_ts2vec" / "models" / "patch_transformer.py"
_spec = importlib.util.spec_from_file_location("_autotcl_patch_transformer", _WRAPPER_FILE)
_pt_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pt_mod)
PatchTransformerWrapper = _pt_mod.PatchTransformerWrapper


# ── data loading helpers ──────────────────────────────────────────────────────

def load_csv_train_data(csv_path: str, timestamp_col: str = "date",
                        seq_len: int = 336) -> np.ndarray:
    """Training split of a CSV forecasting set as (N, seq_len, C) windows, 50 % overlap."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df   = pd.read_csv(csv_path)
    df   = df.drop(columns=[timestamp_col], errors="ignore")
    data = df.values.astype(np.float32)

    T, C      = data.shape
    train_len = int(T * 0.6)
    scaler    = StandardScaler()
    train     = scaler.fit_transform(data[:train_len])

    stride  = max(1, seq_len // 2)
    windows = [train[i:i + seq_len] for i in range(0, len(train) - seq_len + 1, stride)]
    if not windows:
        windows = [np.pad(train, ((0, seq_len - len(train)), (0, 0)), mode="edge")]
    arr = np.stack(windows).astype(np.float32)
    print(f"[AutoTCL] CSV windows: {arr.shape}")
    return arr


def load_monash_windows(monash_dir: str, seq_len: int, min_len: int = 512) -> np.ndarray:
    """All Monash training windows as (N, seq_len, 1) float32."""
    from data_loaders.data_puller import MonashWindowDatasetTimeDart
    ds      = MonashWindowDatasetTimeDart(monash_dir, seq_len=seq_len, which="train", min_len=min_len)
    arr     = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    print(f"[AutoTCL] Monash windows: {arr.shape}")
    return arr


def load_synthetic_windows(synth_dir: str, seq_len: int, min_len: int = 512) -> np.ndarray:
    """All synthetic training windows as (N, seq_len, C) float32."""
    from data_loaders.data_puller import SyntheticWindowDatasetTimeDart
    ds      = SyntheticWindowDatasetTimeDart(synth_dir, seq_len=seq_len, which="train", min_len=min_len)
    arr     = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    print(f"[AutoTCL] Synthetic windows: {arr.shape}")
    return arr


# ── input masks (TS2Vec) ──────────────────────────────────────────────────────

def _binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


def _continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    n = max(min(int(n * T) if isinstance(n, float) else n, T // 2), 1)
    l = max(int(l * T) if isinstance(l, float) else l, 1)
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T - l + 1)
            res[i, t:t + l] = False
    return res


def _make_mask(mode: str, B: int, T: int, device):
    if mode == "binomial":
        m = _binomial_mask(B, T)
    elif mode == "continuous":
        m = _continuous_mask(B, T)
    elif mode == "mask_last":
        m = torch.ones(B, T, dtype=torch.bool); m[:, -1] = False
    elif mode == "all_false":
        m = torch.zeros(B, T, dtype=torch.bool)
    else:  # all_true
        m = torch.ones(B, T, dtype=torch.bool)
    return m.to(device)


# ── augmentation network (TS2Vec-style dilated conv, as in AutoTCL) ──────────

class _SamePadConv(nn.Module):
    def __init__(self, cin, cout, k, dilation=1):
        super().__init__()
        rf = (k - 1) * dilation + 1
        self.conv = nn.Conv1d(cin, cout, k, padding=rf // 2, dilation=dilation)
        self.remove = 1 if rf % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.remove] if self.remove else out


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, final=False):
        super().__init__()
        self.conv1 = _SamePadConv(cin, cout, k, dilation)
        self.conv2 = _SamePadConv(cout, cout, k, dilation)
        self.proj  = nn.Conv1d(cin, cout, 1) if cin != cout or final else None

    def forward(self, x):
        res = x if self.proj is None else self.proj(x)
        x = self.conv2(F.gelu(self.conv1(F.gelu(x))))
        return x + res


class AugEncoder(nn.Module):
    """(B, T, C) -> (B, T, aug_dim). Shared trunk of the gate h(x) and scale g(x)."""

    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=2, mask_mode="binomial"):
        super().__init__()
        self.mask_mode = mask_mode
        self.input_fc  = nn.Linear(input_dims, hidden_dims)
        chans = [hidden_dims] * depth + [output_dims]
        self.blocks = nn.Sequential(*[
            _ConvBlock(chans[i - 1] if i > 0 else hidden_dims, chans[i], 3,
                       dilation=2 ** i, final=(i == len(chans) - 1))
            for i in range(len(chans))])

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        x = self.input_fc(x)
        mode = mask if mask is not None else (self.mask_mode if self.training else "all_true")
        m = _make_mask(mode, B, T, x.device).float().unsqueeze(2)
        x = x * m
        return self.blocks(x.transpose(1, 2)).transpose(1, 2)


# ── losses (models/losses.py of the reference code, device-safe) ──────────────

def _maxpool_time(z):                      # (B, T, D) -> (B, D)
    return F.max_pool1d(z.transpose(1, 2).contiguous(), kernel_size=z.size(1)).squeeze(2)


def info_nce(z1, z2, temperature=1.0):
    """InfoNCE between the two views' max-pooled embeddings; positives on the diagonal."""
    p1 = F.normalize(_maxpool_time(z1), dim=1)
    p2 = F.normalize(_maxpool_time(z2), dim=1)
    logits = p1 @ p2.T / temperature
    return F.cross_entropy(logits, torch.arange(p1.size(0), device=p1.device))


def local_info_nce(z1, k=16, temperature=1.0):
    """
    Local InfoNCE (InfoTS) on one view: k equal crops of the window, max-pooled;
    crop i is positive with crop i+1 (the last with its predecessor), negatives
    are the crops that are not immediate neighbours.
    """
    z1 = F.normalize(z1, dim=2)
    B, T, D = z1.shape
    crop_size = T // k
    if crop_size < 1:
        return z1.new_tensor(0.)
    crop_len = crop_size * k
    start = np.random.randint(0, T - crop_len + 1)
    crops = z1[:, start:start + crop_len, :].reshape(B * k, crop_size, D)
    pooled = F.max_pool1d(crops.transpose(1, 2).contiguous(), kernel_size=crop_size).reshape(B, k, D)
    sim = torch.bmm(pooled, pooled.transpose(1, 2))          # (B, k, k)

    eye = torch.eye(k, device=z1.device)
    upper = torch.diag(torch.ones(k - 1, device=z1.device), 1)   # (i, i+1)
    pos = upper.clone(); pos[k - 1, k - 2] = 1.0
    neg_excl = upper + upper.T + eye
    neg_excl[0, 2] = 1.0; neg_excl[-1, -3] = 1.0

    sim = sim[0]                       # reference code scores the first window only
    positives = sim[pos.bool()].view(k, -1)
    negatives = sim[~neg_excl.bool()].view(k, -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    return (-F.log_softmax(logits, dim=-1)[:, 0]).mean()


def mmd(z1, z2, kernel_mul=2.0, kernel_num=5):
    """Multi-bandwidth Gaussian MMD between the two views' pooled embeddings."""
    s, t = _maxpool_time(z1), _maxpool_time(z2)
    n = s.size(0)
    total = torch.cat([s, t], 0)
    d2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    bw = d2.detach().sum() / (total.size(0) ** 2 - total.size(0))
    bw = bw / kernel_mul ** (kernel_num // 2)
    K = sum(torch.exp(-d2 / (bw * kernel_mul ** i)) for i in range(kernel_num))
    return (K[:n, :n] + K[n:, n:] - K[:n, n:] - K[n:, :n]).mean()


def gate_smoothness(w):
    """AutoTCL's regular_consistency: neighbouring gates alike, distant ones free."""
    B, T, _ = w.shape
    s0 = torch.randint(1, T - 2, (B,)); s1 = torch.randint(1, T - 2, (B,))
    far = ((s1 - s0) > 1).float().to(w.device).reshape(1, B, 1)
    diff = (far * (w[:, s0] - w[:, s1]).abs()
            + (w[:, s0] - w[:, s0 - 1]).abs() + (w[:, s0] - w[:, s0 + 1]).abs()
            + (1 - far) * (1 - (w[:, s0] - w[:, s1]).abs()))
    return diff.mean()


# ── the model ─────────────────────────────────────────────────────────────────

class AutoTCL:
    """AutoTCL with the PatchTST transformer as encoder. Mirrors AutoTCL_CoST.AutoTCL."""

    def __init__(self, cfg: dict, c_in: int, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device
        self.lr          = cfg["lr"]
        self.meta_lr     = cfg.get("meta_lr", 1e-2)
        self.batch_size  = cfg["batch_size"]
        self.temperature = cfg.get("temperature", 1.0)
        self.local_weight   = cfg.get("local_weight", 0.1)
        self.local_k        = cfg.get("local_k", 16)
        self.reg_weight     = cfg.get("reg_weight", 0.1)
        self.regular_weight = cfg.get("regular_weight", 0.0)
        self.reg_thres      = cfg.get("reg_thres", 0.4)
        self.ratio_step     = max(1, int(cfg.get("ratio_step", 1)))
        self.hard_mask      = bool(cfg.get("hard_mask", True))
        self.gumbel_bias    = cfg.get("gumbel_bias", 0.05)
        self.gamma_zeta     = -abs(cfg.get("gamma_zeta", 0.05))
        self.zeta           = 1.0
        self.mask_mode      = cfg.get("mask_mode", "all_true")

        self._net = PatchTransformerWrapper(
            input_dims=c_in, output_dims=cfg["embed_dim"], patch_len=cfg["patch_len"],
            max_num_patches=cfg["num_patches"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], d_ff=cfg["d_ff"], dropout=cfg.get("dropout", 0.1),
        ).to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        aug_dim = cfg.get("aug_dim", 16)
        self.augsharenet = AugEncoder(
            c_in, aug_dim, hidden_dims=cfg.get("aug_hidden_dims", 64),
            depth=cfg.get("aug_depth", 2), mask_mode=cfg.get("augmask_mode", "binomial"),
        ).to(device)
        self.factor_augnet = nn.Sequential(nn.Linear(aug_dim, 1), nn.Sigmoid()).to(device)   # h(x)
        self.augmentation_projector = nn.Sequential(nn.Linear(aug_dim, 1), nn.Sigmoid()).to(device)  # g(x)

        self.n_epochs = 0
        self.n_iters  = 0

    # -- learned augmentation ------------------------------------------------
    def _sample_gate(self, logits, training=True):
        if training:
            bias = self.gumbel_bias
            eps = (bias - (1 - bias)) * torch.rand_like(logits) + (1 - bias)
            g = torch.sigmoid(torch.log(eps) - torch.log(1 - eps) + logits)
        else:
            g = torch.sigmoid(logits)
        return torch.clip(g * (self.zeta - self.gamma_zeta) + self.gamma_zeta, 0.0, 1.0)

    def get_features(self, x, training=True, aug_mask=None):
        emb      = self.augsharenet(x, mask=aug_mask)
        weight_h = self.factor_augnet(emb)                 # (B, T, 1) gate
        weight_s = self.augmentation_projector(emb)        # (B, T, 1) scale
        mask_h   = self._sample_gate(weight_h, training)
        if self.hard_mask:
            hard = (torch.sign(mask_h - 0.5) + 1) / 2
            mask_h = (hard - mask_h).detach() + mask_h     # straight-through
        ax = weight_s * mask_h * x
        if self.mask_mode != "all_true" and self._net.training:
            m = _make_mask(self.mask_mode, x.size(0), x.size(1), x.device).float().unsqueeze(2)
            x, ax = x * m, ax * m
        return x, ax, self._net(x), self._net(ax), weight_h

    # -- training --------------------------------------------------------------
    def fit(self, train_data: np.ndarray, n_epochs: int, verbose: bool = True):
        assert train_data.ndim == 3
        train_data = np.nan_to_num(train_data[~np.isnan(train_data).all(axis=2).all(axis=1)])
        loader = DataLoader(TensorDataset(torch.from_numpy(train_data).float()),
                            batch_size=min(self.batch_size, len(train_data)),
                            shuffle=True, drop_last=True)

        import itertools
        meta_opt = torch.optim.AdamW(itertools.chain(
            self.augsharenet.parameters(), self.factor_augnet.parameters(),
            self.augmentation_projector.parameters()), lr=self.meta_lr)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        loss_log = []
        while self.n_epochs < n_epochs:
            cum_loss, cum_aug, n_batches = 0.0, 0.0, 0
            for (x,) in loader:
                x = x.to(self.device)

                # augmentation-network step (PRI): encoder frozen for this pass
                if self.n_iters % self.ratio_step == 0:
                    self._net.eval(); self.augsharenet.train()
                    meta_opt.zero_grad()
                    _, _, outx, outv, weight_h = self.get_features(x, aug_mask="all_true")
                    reg      = F.relu(weight_h.sum(dim=-1).mean() - self.reg_thres)
                    aug_loss = mmd(outx, outv) + self.reg_weight * reg
                    if self.regular_weight:
                        aug_loss = aug_loss + self.regular_weight * gate_smoothness(weight_h)
                    aug_loss.backward()
                    meta_opt.step()
                    cum_aug += aug_loss.item()

                # encoder step: contrastive agreement between x and its learned view
                self._net.train(); self.augsharenet.eval()
                opt.zero_grad()
                _, _, out1, out2, _ = self.get_features(x)
                loss = info_nce(out1, out2, self.temperature)
                if self.local_weight:
                    loss = loss + self.local_weight * local_info_nce(out1, self.local_k, self.temperature)
                loss.backward()
                opt.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item(); n_batches += 1; self.n_iters += 1

            cum_loss /= max(n_batches, 1)
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss:.6f}  aug_loss={cum_aug / max(n_batches, 1):.6f}")
            self.n_epochs += 1
        return loss_log

    # -- export ----------------------------------------------------------------
    def backbone_state_dict(self):
        """The EMA encoder's PatchTSTEncoder weights, what AutoTCL evaluates with."""
        return self.net.module.backbone.state_dict()


# ── checkpoint helper ─────────────────────────────────────────────────────────

def save_backbone(model: AutoTCL, path: str, epoch: int, cfg: dict):
    torch.save({"backbone": model.backbone_state_dict(), "epoch": epoch, "config": cfg}, path)
    print(f"[AutoTCL] Backbone saved → {path}")


# ── model builder ─────────────────────────────────────────────────────────────

def build_model(cfg: dict, c_in: int, device: str = "cuda") -> AutoTCL:
    return AutoTCL(cfg, c_in, device)
