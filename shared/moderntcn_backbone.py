"""
ModernTCN backbone (Luo & Wang, ICLR 2024, "ModernTCN: A Modern Pure Convolution
Structure for General Time Series Analysis") as a drop-in replacement for the
patch transformer of DINO, JEPA and LE-JEPA.

The block below is the reference ModernTCN block (../ModernTCN-main,
ModernTCN-Long-term-forecasting/models/ModernTCN.py: ReparamLargeKernelConv +
Block): a depthwise large-kernel convolution with a parallel small kernel, each
followed by BatchNorm, then two grouped 1x1-conv FFNs, with a residual over the
whole block. What changes is only how it is wired into the benchmark:

  * Patching stays the benchmark's: patch length 16, linear patch embedding to
    d_model. With stride == patch length this equals ModernTCN's conv stem.
  * One stage, no down-sampling, so the encoder keeps one token per patch, the
    same token grid the transformer produces. Every predictor, probe and head
    reads the encoder unchanged.
  * Channel-independent, like the transformer it replaces: each variable is
    encoded on its own (nvars = 1 inside the block). Pre-training is on
    univariate Monash series, so a block with cross-variable mixing could not be
    transferred to 7- or 862-channel downstream data. ModernTCN's ConvFFN2
    (the cross-variable FFN) is kept, and with one variable it reduces to a
    per-feature 1x1 expansion.
  * No positional encoding, as in ModernTCN: convolution is translation
    equivariant and the zero padding at the edges carries position.
  * JEPA follows CNN-JEPA (Kalapos & Gyires-Toth, 2024): a sparse student
    encoder (masked positions zeroed at every block, sparse BatchNorm) and a
    fully convolutional, depthwise-separable predictor that places mask tokens
    on the patch grid (CNNJEPAPredictor).
  * Depth = number of blocks = the transformer's layer count, and the FFN ratio
    is d_ff / d_model, so the depth and width knobs of every script still apply.

Kernel sizes default to large 13 / small 5. ModernTCN's long-term forecasting
configs use a large kernel of ~0.6 x the token count (51 for 84 tokens); 13 is
that ratio for the 21-token context used here. TS_MTCN_LARGE and TS_MTCN_SMALL
override them (both must be odd).

Which backbone the objectives build is read from TS_BACKBONE ("transformer",
the default, or "moderntcn"). Train_and_downstream.py sets it from --backbone.
"""

import os

import torch
import torch.nn as nn


def backbone_name() -> str:
    name = os.environ.get("TS_BACKBONE", "transformer").strip().lower() or "transformer"
    if name not in ("transformer", "moderntcn"):
        raise ValueError(f"TS_BACKBONE={name!r}: expected 'transformer' or 'moderntcn'")
    return name


def use_moderntcn() -> bool:
    return backbone_name() == "moderntcn"


def _kernel_sizes():
    large = int(os.environ.get("TS_MTCN_LARGE", 13))
    small = int(os.environ.get("TS_MTCN_SMALL", 5))
    if large % 2 == 0 or small % 2 == 0:
        raise ValueError(f"ModernTCN kernels must be odd (large={large}, small={small})")
    if small > large:
        raise ValueError(f"small kernel {small} > large kernel {large}")
    return large, small


# ── sparse (masked) BatchNorm ─────────────────────────────────────────────────

def _bn(bn, x, keep=None):
    """BatchNorm over [N, D, L]. With keep ([N, L] bool) in training, the batch
    statistics come from the kept tokens only and the rest come back as zero:
    SparK's sparse BatchNorm, as used by CNN-JEPA, so masked (all-zero) positions
    never shift the statistics. In eval the running statistics are used as usual,
    so the same weights run densely downstream."""
    if keep is None or not bn.training:
        return bn(x)
    xt = x.transpose(1, 2)                                     # [N, L, D]
    out = xt.new_zeros(xt.shape)
    out[keep] = bn(xt[keep])                                   # BatchNorm1d on [M, D]
    return out.transpose(1, 2)


# ── reference ModernTCN block (nvars = 1) ─────────────────────────────────────

def _conv_bn(in_ch, out_ch, kernel_size, groups):
    seq = nn.Sequential()
    seq.add_module("conv", nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1,
                                     padding=kernel_size // 2, groups=groups, bias=False))
    seq.add_module("bn", nn.BatchNorm1d(out_ch))
    return seq


class ReparamLargeKernelConv(nn.Module):
    """Depthwise large kernel + parallel small kernel (re-parameterisable)."""

    def __init__(self, channels, large_size, small_size):
        super().__init__()
        self.lkb_origin = _conv_bn(channels, channels, large_size, groups=channels)
        self.small_conv = _conv_bn(channels, channels, small_size, groups=channels)

    def forward(self, x, keep=None):
        return (_bn(self.lkb_origin.bn, self.lkb_origin.conv(x), keep)
                + _bn(self.small_conv.bn, self.small_conv.conv(x), keep))


class ModernTCNBlock(nn.Module):
    """ModernTCN Block with nvars = 1. x: [N, D, L] -> [N, D, L]."""

    def __init__(self, d_model, d_ff, large_size, small_size, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(d_model, large_size, small_size)
        self.norm = nn.BatchNorm1d(d_model)
        # ConvFFN1: per-variable feature mixing (groups = nvars = 1)
        self.ffn1pw1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)
        # ConvFFN2: cross-variable mixing (groups = d_model); one variable here
        self.ffn2pw1 = nn.Conv1d(d_model, d_ff, kernel_size=1, groups=d_model)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(d_ff, d_model, kernel_size=1, groups=d_model)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x, keep=None):
        res = x
        x = _bn(self.norm, self.dw(x, keep), keep)
        x = self.ffn1drop2(self.ffn1pw2(self.ffn1act(self.ffn1drop1(self.ffn1pw1(x)))))
        x = self.ffn2drop2(self.ffn2pw2(self.ffn2act(self.ffn2drop1(self.ffn2pw1(x)))))
        return res + x


class ModernTCNStack(nn.Module):
    """n_layers ModernTCN blocks over a token sequence. [N, L, D] -> [N, L, D].

    keep: optional [N, L] bool, True = token present (visible context patch, or
    real data under classification padding). Absent tokens are zeroed before and
    after every block and excluded from BatchNorm statistics, so the network
    behaves as a sparse CNN over the present tokens (SparK / CNN-JEPA). This is
    the convolutional counterpart of dropping tokens or masking attention keys.
    """

    def __init__(self, d_model, n_layers, d_ff, dropout=0.1):
        super().__init__()
        large, small = _kernel_sizes()
        self.large_size, self.small_size = large, small
        if d_ff % d_model != 0:
            raise ValueError(f"ModernTCN needs d_ff ({d_ff}) to be a multiple of d_model ({d_model})")
        self.blocks = nn.ModuleList([
            ModernTCNBlock(d_model, d_ff, large, small, drop=dropout) for _ in range(n_layers)
        ])

    def forward(self, x, keep=None):
        x = x.transpose(1, 2)                                   # [N, D, L]
        keep = None if keep is None else keep.to(x.device).bool()
        k = None if keep is None else keep.to(x.device).unsqueeze(1).to(x.dtype)
        for blk in self.blocks:
            if k is not None:
                x = x * k
            x = blk(x, keep)
        if k is not None:
            x = x * k
        return x.transpose(1, 2)                                # [N, L, D]


# ── JEPA / LE-JEPA encoder ────────────────────────────────────────────────────

class ModernTCNPatchEncoder(nn.Module):
    """Same constructor and forward contract as JEPA/LeJEPA Encoder.

    Input  x: [B, P, P_L, F];  mask: [B, n_ctx] context indices or None;
           padding_mask: [B, P] bool, True = real.
    Output dict(data_patches=[B*F, n_ctx_or_P, D], orig_B=B, orig_F=F).

    The transformer student drops the masked patches before attention. Here the
    student is a sparse CNN, as in CNN-JEPA (Kalapos & Gyires-Toth, 2024): the
    non-context patches are zero, are re-zeroed before every block and are left
    out of the BatchNorm statistics, so the context features are computed from
    the context patches alone. The context positions are returned. Mask tokens
    belong to the predictor (CNNJEPAPredictor), not the encoder.
    """

    def __init__(self, num_patches, dim_in, embed_dim, nhead=None, num_layers=8,
                 mlp_ratio=4.0, drop_rate=0.0, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.W_P = nn.Linear(dim_in, embed_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.tcn = ModernTCNStack(embed_dim, num_layers, int(embed_dim * mlp_ratio), dropout=drop_rate)
        self.encoder_norm = nn.LayerNorm(embed_dim)
        # Not a positional encoding: the probes read W_pos.shape[0] as the number
        # of patches the encoder was built for. Not saved in checkpoints.
        self.register_buffer("W_pos", torch.zeros(num_patches, embed_dim), persistent=False)

    def forward(self, x, mask=None, padding_mask=None, **kwargs):
        B, P, P_L, F = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B * F, P, P_L)
        x = self.dropout(self.W_P(x))                            # [B*F, P, D]

        keep = None
        if padding_mask is not None:
            keep = padding_mask.to(x.device).bool().unsqueeze(1).expand(-1, F, -1).reshape(B * F, P)

        m = None
        if mask is not None:
            mask = mask.to(x.device)
            n_ctx = mask.shape[1]
            m = mask.unsqueeze(1).expand(-1, F, -1).reshape(B * F, n_ctx)
            visible = torch.zeros(B * F, P, dtype=torch.bool, device=x.device)
            visible.scatter_(1, m, True)
            keep = visible if keep is None else (keep & visible)

        x = self.tcn(x, keep=keep)
        x = self.encoder_norm(x)
        if m is not None:
            x = torch.gather(x, 1, m.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        elif keep is not None and not bool(keep.all()):
            # Padded positions leave the stack as zeros, so after encoder_norm they
            # are the LayerNorm bias: the same vector for every sample. The JEPA /
            # LeJEPA classification heads read the LAST patch, which is padding for
            # every sample shorter than its batch's longest (variable-length UEA
            # sets: JapaneseVowels, SpokenArabicDigits), so the probe saw a
            # constant and predicted one class. A transformer still yields an
            # informative output there (padded queries attend to the real keys).
            # Give each padded position the embedding of the last real patch, so
            # "last patch" means the last real one; unpadded inputs are unchanged.
            pos = torch.arange(P, device=x.device).unsqueeze(0).expand(B * F, -1)
            last = torch.where(keep, pos, torch.zeros_like(pos)).max(dim=1).values   # [B*F]
            src = torch.where(keep, pos, last.unsqueeze(1))                           # [B*F, P]
            x = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, self.embed_dim))
        return {"data_patches": x, "orig_B": B, "orig_F": F}


# ── DINO encoder ──────────────────────────────────────────────────────────────

class ModernTCNPatchTSTEncoder(nn.Module):
    """Same constructor and forward contract as DINO's PatchTSTEncoder.

    Input  x: [bs, num_patch, nvars, patch_len]; padding_mask: [bs, num_patch], True = real.
    Output   [bs, num_patch + 1, nvars, d_model]. Slot 0 is the global token DINO
    reads (the transformer's CLS): here the mean of the real patch tokens, since a
    convolution has no token that attends to the whole sequence.
    """

    def __init__(self, c_in, num_patch, patch_len, n_layers=3, d_model=128,
                 n_heads=None, shared_embedding=True, d_ff=256, dropout=0., **kwargs):
        super().__init__()
        if not shared_embedding:
            raise ValueError("ModernTCN backbone supports shared_embedding=True only")
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = True
        self.W_P = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.tcn = ModernTCNStack(d_model, n_layers, d_ff, dropout=dropout)
        self.register_buffer("W_pos", torch.zeros(num_patch + 1, d_model), persistent=False)

    def forward(self, x, padding_mask=None):
        bs, num_patch, n_vars, _ = x.shape
        x = self.W_P(x).transpose(1, 2).reshape(bs * n_vars, num_patch, self.d_model)
        x = self.dropout(x)

        keep = None
        if padding_mask is not None:
            keep = padding_mask.to(x.device).bool().unsqueeze(1).expand(-1, n_vars, -1).reshape(bs * n_vars, num_patch)

        z = self.tcn(x, keep=keep)                               # [bs*nvars, num_patch, D]
        if keep is None:
            g = z.mean(dim=1, keepdim=True)
        else:
            w = keep.to(z.dtype).unsqueeze(-1)
            g = (z * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1.0)
        z = torch.cat([g, z], dim=1)                             # [bs*nvars, num_patch+1, D]
        z = z.reshape(bs, n_vars, num_patch + 1, self.d_model)
        return z.permute(0, 2, 1, 3)                             # [bs, num_patch+1, nvars, D]


# ── CNN-JEPA predictor ────────────────────────────────────────────────────────

class _DWSepBlock(nn.Module):
    """Depthwise-separable residual conv block: x + PW(GELU(BN(DW(x))))."""

    def __init__(self, dim, kernel_size, drop=0.0):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.pw(self.act(self.bn(self.dw(x)))))


class CNNJEPAPredictor(nn.Module):
    """Fully convolutional JEPA predictor, as in CNN-JEPA. Same call contract as
    JEPA/JEPA/Predictors.py: (context [B*F, N_c, D], target_mask [B, N_t],
    context_mask [B, N_c]) -> predicted target embeddings [B*F, N_t, D].

    The context embeddings go back on the full patch grid, every other position
    gets a learned mask token, a stack of depthwise-separable convolutions runs
    over the grid and the target positions are read out. The convolutions see
    where each token sits, so no positional encoding is needed.

    Width and depth follow the transformer predictor's config
    (predictor_embed_dim, predictor_num_layers); the kernel is the encoder's
    large kernel, so the predictor's receptive field spans the context.
    """

    def __init__(self, num_patches, embed_dim, config=None, **kwargs):
        super().__init__()
        config = config or {}
        pred_dim = config.get("predictor_embed_dim", embed_dim // 2)
        n_layers = config.get("predictor_num_layers", 2)
        drop = config.get("predictor_drop_rate", 0.1)
        large, _ = _kernel_sizes()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.in_proj = nn.Conv1d(embed_dim, pred_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, pred_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([_DWSepBlock(pred_dim, large, drop) for _ in range(n_layers)])
        self.out_norm = nn.BatchNorm1d(pred_dim)
        self.out_proj = nn.Conv1d(pred_dim, embed_dim, kernel_size=1)

    def forward(self, x_input, target_mask=None, context_mask=None, **kwargs):
        if context_mask is None or target_mask is None:
            raise ValueError("CNNJEPAPredictor needs both context_mask and target_mask")
        BF, N_c, _ = x_input.shape
        B = context_mask.shape[0]
        F = BF // B
        P = self.num_patches
        dev = x_input.device
        ctx = context_mask.to(dev).unsqueeze(1).expand(-1, F, -1).reshape(BF, N_c)
        tgt = target_mask.to(dev)
        N_t = tgt.shape[1]
        tgt = tgt.unsqueeze(1).expand(-1, F, -1).reshape(BF, N_t)

        c = self.in_proj(x_input.transpose(1, 2))                    # [BF, pd, N_c]
        grid = self.mask_token.to(c.dtype).expand(BF, -1, P).clone()  # [BF, pd, P]
        grid.scatter_(2, ctx.unsqueeze(1).expand(-1, c.shape[1], -1), c)
        for blk in self.blocks:
            grid = blk(grid)
        out = self.out_proj(self.out_norm(grid))                     # [BF, D, P]
        out = torch.gather(out, 2, tgt.unsqueeze(1).expand(-1, self.embed_dim, -1))
        return out.transpose(1, 2)                                   # [BF, N_t, D]
