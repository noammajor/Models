import sys
from pathlib import Path

import torch
import torch.nn as nn

# Resolve TSDiNO so we can import PatchTSTEncoder without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TSDINO_MODELS = str(_REPO_ROOT / "TSDiNO" / "models")
if _TSDINO_MODELS not in sys.path:
    sys.path.insert(0, _TSDINO_MODELS)

from patchTST import PatchTSTEncoder


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
