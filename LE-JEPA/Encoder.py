import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


# ─── Positional Encoding ──────────────────────────────────────────────────────

def positional_encoding(pe, learn_pe, q_len, d_model) -> nn.Parameter:
    """Return a positional encoding as nn.Parameter [q_len, d_model]."""
    if pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe in ('normal', 'gauss'):
        W_pos = torch.zeros((q_len, 1))
        nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'sincos':
        pe_table = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe_table[:, 0::2] = torch.sin(position * div_term)
        pe_table[:, 1::2] = torch.cos(position * div_term)
        pe_table = pe_table - pe_table.mean()
        pe_table = pe_table / (pe_table.std()*10)
        W_pos = pe_table
        learn_pe = False
    else:
        raise ValueError(f"'{pe}' is not a valid pe.")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# ─── Attention ────────────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)

    def forward(self, q, k, v, prev: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_scores = attn_scores + attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False,
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout,
            res_attention=res_attention, lsa=lsa
        )
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        if self.res_attention:
            output, _, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, attn_mask=attn_mask)
        else:
            output, _ = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)
            scores = None
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        return output, scores


# ─── Transformer ──────────────────────────────────────────────────────────────

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, attn_dropout=0., dropout=0.,
                 res_attention=True, pre_norm=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.self_attn = MultiheadAttention(
            d_model, n_heads, attn_dropout=attn_dropout,
            proj_dropout=dropout, res_attention=res_attention
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, prev=None, attn_mask=None):
        if self.pre_norm:
            src = self.norm_attn(src)
        src2, scores = self.self_attn(src, src, src, prev=prev, attn_mask=attn_mask)
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src = src + self.dropout_ffn(self.ff(src))
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src, scores


class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, attn_dropout=0., dropout=0.,
                 res_attention=True, n_layers=3, pre_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model, n_heads, d_ff=d_ff, attn_dropout=attn_dropout,
                dropout=dropout, res_attention=res_attention, pre_norm=pre_norm
            )
            for _ in range(n_layers)
        ])

    def forward(self, src, attn_mask=None):
        output, scores = src, None
        for mod in self.layers:
            output, scores = mod(output, prev=scores, attn_mask=attn_mask)
        return output


# ─── Encoder ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    PatchTST-style channel-independent encoder (JEPA variant — no semantic tokens).

    Input:  [B, P, P_L, F]  (batch, num_patches, patch_len, n_vars)
    Output: dict with 'data_patches' [B*F, n_ctx, D]

    Patch embedding : nn.Linear(patch_len → embed_dim)
    Positional enc  : fixed sincos nn.Parameter [num_patches, embed_dim]
    Transformer     : TSTEncoder with residual attention
    """

    def __init__(
        self,
        num_patches,
        dim_in,           # patch_len (P_L)
        embed_dim,
        nhead,
        num_layers,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        pe='sincos',
        learn_pe=False,
        res_attention=True,
        pre_norm=False,
        **kwargs,         # absorb legacy args
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.W_P = nn.Linear(dim_in, embed_dim)
        self.W_pos = positional_encoding(pe, learn_pe, num_patches, embed_dim)  # learnable PE str
        self.dropout = nn.Dropout(drop_rate)
        self.pe_dropout = nn.Dropout(p=0.3)  # randomly zero PE during training

        d_ff = int(embed_dim * mlp_ratio)
        self.transformer = TSTEncoder(
            d_model=embed_dim, n_heads=nhead, d_ff=d_ff,
            attn_dropout=attn_drop_rate, dropout=drop_rate,
            res_attention=res_attention, n_layers=num_layers, pre_norm=pre_norm,
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None, padding_mask=None, **kwargs):
        """
        x:            [B, P, P_L, F]
        mask:         [B, n_ctx] LongTensor — context patch indices (student encoder).
                      Pass None to encode all patches (target/EMA encoder).
        padding_mask: [B, P] bool — True = real data, False = zero-padding.
                      Only used during classification; ignored during pretraining (default None).
        """
        B, P, P_L, F = x.shape

        # Channel independence: [B, P, P_L, F] → [B*F, P, P_L]
        x = x.permute(0, 3, 1, 2).reshape(B * F, P, P_L)

        # Patch projection + positional encoding
        x = self.W_P(x)
        x = self.dropout(x + self.pe_dropout(self.W_pos[:P, :]))

        # Build patch-level attention mask before any context subsampling
        pm_expanded = None
        if padding_mask is not None:
            pm = padding_mask.to(x.device)                                   # (B, P)
            pm_expanded = pm.unsqueeze(1).expand(-1, F, -1).reshape(B * F, P)  # (B*F, P)

        # Context masking (student encoder only)
        if mask is not None:
            mask = mask.to(x.device)
            n_ctx = mask.shape[1]
            m = mask.unsqueeze(1).expand(-1, F, -1).reshape(B * F, n_ctx)
            x = torch.gather(x, 1, m.unsqueeze(-1).expand(-1, -1, self.embed_dim))
            if pm_expanded is not None:
                pm_expanded = torch.gather(pm_expanded, 1, m)               # (B*F, n_ctx)

        # Convert to additive attention mask: True = ignore (padding key)
        attn_mask = None
        if pm_expanded is not None:
            attn_mask = ~pm_expanded[:, None, None, :]                       # (B*F, 1, 1, P_cur)

        x = self.transformer(x, attn_mask=attn_mask)
        x = self.encoder_norm(x)

        return {
            "data_patches": x,   # [B*F, n_ctx_or_P, D]
            "orig_B":       B,
            "orig_F":       F,
        }
