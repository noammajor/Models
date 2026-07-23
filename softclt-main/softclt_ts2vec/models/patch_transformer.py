import sys
from pathlib import Path

import torch
import torch.nn as nn

# Resolve TSDiNO so we can import PatchTSTEncoder without installing the package.
#
# TSDiNO/models/patchTST.py internally does `from models.layers... import *` and
# `from utils.util import ...`, so while it is being imported the top-level names
# `models` and `utils` must resolve to TSDiNO's packages — not to SoftCLT's
# identically-named ones (this file lives inside SoftCLT's `models` package).
# Swap TSDiNO in for the duration of the import, then restore SoftCLT's.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TSDINO = str(_REPO_ROOT / "TSDiNO")

_saved = {k: v for k, v in sys.modules.items() if k.split('.')[0] in ('models', 'utils')}
for _k in list(_saved):
    del sys.modules[_k]
# TSDiNO may already be on sys.path but *behind* SoftCLT's dirs, so force it to
# the front for the duration of the import, then put it back where it was.
_orig_idx = sys.path.index(_TSDINO) if _TSDINO in sys.path else None
if _orig_idx is not None:
    sys.path.remove(_TSDINO)
sys.path.insert(0, _TSDINO)
try:
    from models.patchTST import PatchTSTEncoder
finally:
    for _k in [k for k in list(sys.modules) if k.split('.')[0] in ('models', 'utils')]:
        del sys.modules[_k]
    sys.modules.update(_saved)
    if _TSDINO in sys.path:
        sys.path.remove(_TSDINO)
    if _orig_idx is not None:
        sys.path.insert(_orig_idx, _TSDINO)


class PatchTransformerWrapper(nn.Module):
    """
    Drop-in replacement for TSEncoder using the PatchTST transformer backbone.

    Input:  (B, T, C)  — same as TSEncoder
    Output: (B, P, C * d_model)  where P = T // patch_len

    The temporal dimension T becomes P (patch count).  The caller (soft_ts2vec.py)
    must do its crop arithmetic in patch units when using this encoder — see the
    patch_len argument on TS2Vec.

    Mask argument is accepted for interface compatibility but unused.
    """

    def __init__(
        self,
        input_dims: int,           # n_vars / C
        output_dims: int,          # d_model per variable
        patch_len: int = 16,
        max_num_patches: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.n_vars = input_dims
        self.d_model = output_dims

        self.backbone = PatchTSTEncoder(
            c_in=input_dims,
            num_patch=max_num_patches,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=output_dims,
            n_heads=n_heads,
            shared_embedding=True,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act="gelu",
            res_attention=False,
            pre_norm=False,
            pe="zeros",
            learn_pe=True,
        )

    def forward(self, x, mask=None):  # x: (B, T, C)
        B, T, C = x.shape
        P = T // self.patch_len

        # Trim to patch boundary and patchify
        # unfold(dim=1, size, step) on (B, T, C) → (B, P, C, patch_len)
        x_trimmed = x[:, : P * self.patch_len, :]
        patches = x_trimmed.unfold(1, self.patch_len, self.patch_len)  # (B, P, C, patch_len)

        # PatchTSTEncoder expects (B, num_patch, n_vars, patch_len) ✓
        z = self.backbone(patches)      # (B, P+1, C, d_model)  — CLS at index 0
        z = z[:, 1:, :, :]             # drop CLS → (B, P, C, d_model)
        z = z.reshape(B, P, -1)        # flatten channels → (B, P, C * d_model)
        return z
