import math
import torch
import torch.nn as nn

from JEPA.Encoder import use_moderntcn
from moderntcn_backbone import CNNJEPAPredictor


# ─── Cross-Attention Block ─────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block.

    queries attend to context (keys / values); used for mask-token prediction.
    """

    def __init__(self, embed_dim: int, nhead: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )
        self.drop    = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q:  [B, N_q,  D]
        # kv: [B, N_kv, D]
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop(attn_out)
        q = q + self.ff(self.norm_ff(q))
        return q


# ─── Predictor ─────────────────────────────────────────────────────────────────

class TransformerJEPAPredictor(nn.Module):
    """
    Bottleneck transformer predictor for JEPA (P2P only).

    Context patch embeddings → predicted target patch embeddings at masked positions.

    Encoder embeddings are projected down to predictor_embed_dim for
    cross-attention, then projected back up to encoder embed_dim for output.
    This bottleneck prevents trivial identity solutions.
    """

    def __init__(
        self,
        embed_dim: int,
        config: dict,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        num_patches   = config["ratio_patches"]
        pred_dim      = config.get("predictor_embed_dim", embed_dim // 2)
        nhead         = config.get("predictor_nhead",     4)
        num_layers    = config.get("predictor_num_layers", 2)
        dropout       = config.get("predictor_drop_rate",  0.1)
        self.pred_dim = pred_dim

        # ── Input projections: encoder_dim → predictor_dim ────────────────────
        self.kv_proj = nn.Linear(embed_dim, pred_dim)   # projects context kv
        self.pe_proj = nn.Linear(embed_dim, pred_dim)   # projects sincos PE

        # ── Learnable tokens (at predictor_dim) ───────────────────────────────
        self.mask_token     = nn.Parameter(torch.zeros(1, 1, pred_dim))
       
        # ── Positional encoding (fixed sincos at encoder embed_dim) ───────────
        self.register_buffer("pos_embed", self._make_sincos_pe(num_patches, embed_dim))

        # ── Cross-attention stacks (at predictor_dim) ─────────────────────────
        self.predictor_layers = nn.ModuleList([
            CrossAttentionBlock(pred_dim, nhead, mlp_ratio=2.0, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── Output projections: predictor_dim → encoder_dim ───────────────────
        # LayerNorm at pred_dim then project back up to encoder embed_dim.
        # This completes the bottleneck: encoder_dim → pred_dim → encoder_dim.
        self.out_proj = nn.Sequential(nn.LayerNorm(pred_dim), nn.Linear(pred_dim, embed_dim))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_sincos_pe(num_patches: int, d_model: int) -> torch.Tensor:
        """Normalized sincos PE matching the encoder's positional_encoding('sincos')."""
        pe = torch.zeros(num_patches, d_model)
        pos = torch.arange(0, num_patches).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
        return pe  # [num_patches, encoder_dim]

    def _build_queries(self, target_mask: torch.Tensor, B_total: int) -> torch.Tensor:
        """
        Construct mask-token queries for target positions at predictor_dim.

        target_mask: [B, N_t]  — integer patch indices in [0, num_patches)
        B_total:     B*F       — channel-independent batch size

        Returns: [B*F, N_t, pred_dim]
        """
        B_orig, N_t = target_mask.shape
        F = B_total // B_orig

        pe = self.pos_embed[target_mask]                             # [B, N_t, encoder_dim]
        pe = pe.unsqueeze(1).expand(-1, F, -1, -1)                  # [B, F, N_t, encoder_dim]
        pe = pe.reshape(B_total, N_t, self.embed_dim)               # [B*F, N_t, encoder_dim]
        pe = self.pe_proj(pe)                                        # [B*F, N_t, pred_dim]

        queries = self.mask_token.expand(B_total, N_t, -1) + pe    # [B*F, N_t, pred_dim]
        return queries

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_input: torch.Tensor,
        target_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        x_input:     [B*F, N_ctx, encoder_dim]  context patch embeddings
        target_mask: [B, N_t]                   target patch indices

        Returns: [B*F, N_t, encoder_dim]
        """
        B  = x_input.shape[0]
        kv = self.kv_proj(x_input)   # [B*F, N_kv, pred_dim]
        q = self._build_queries(target_mask, B)
        for layer in self.predictor_layers:
            q = layer(q, kv)
        return self.out_proj(q)   # [B*F, N_t, encoder_dim]


def JEPAPredictor(*args, **kwargs):
    """The predictor this run uses: the cross-attention transformer above, or the
    fully convolutional CNN-JEPA predictor when TS_BACKBONE=moderntcn. Both map
    context embeddings to predicted target embeddings with the same call."""
    if use_moderntcn():
        return CNNJEPAPredictor(*args, **kwargs)
    return TransformerJEPAPredictor(*args, **kwargs)
