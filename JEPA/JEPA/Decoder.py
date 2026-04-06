"""
    Script for the Decoder
    ---
        Class Decoder contains the decoder achitecture which is based on a
        simple Linear Layer.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Linear classification head for JEPA encoder output.

    Encoder outputs data_patches: [B*n_vars, num_patches, D].
    We take the last patch embedding per variable, flatten across variables,
    and project to n_classes — identical pattern to PatchTST's ClassificationHead.

    Expected input (after reshape): [B, n_vars, D, num_patches]
    Output: [B, n_classes]
    """
    def __init__(self, n_vars: int, d_model: int, n_classes: int, head_dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear  = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        """x: [B, n_vars, D, num_patches]  →  [B, n_classes]"""
        x = x[:, :, :, -1]      # last patch: [B, n_vars, D]
        x = self.flatten(x)      # [B, n_vars * D]
        x = self.dropout(x)
        return self.linear(x)    # [B, n_classes]


class LinearDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Linear(emb_dim, patch_size)

    def forward(self, encoded_patch):
        return self.fc(encoded_patch)
        

class MLPForecastHead(nn.Module):
    """MLP forecast head for P2P and S2P prediction.

    in_dim  : num_ctx_patches * embed_dim  (for P2P)
              OR num_semantic_tokens * embed_dim  (for S2P)
    out_dim : horizon_patches * patch_len  (h_t * P_L)
    """
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B*n_v, in_dim]  →  [B*n_v, out_dim]
        return self.net(x)
class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        head_dim = d_model * num_patch

        if self.individual:
            self.flattens = nn.ModuleList([nn.Flatten(start_dim=-2) for _ in range(n_vars)])
            self.linears = nn.ModuleList([nn.Linear(head_dim, forecast_len) for _ in range(n_vars)])
            self.dropouts = nn.ModuleList([nn.Dropout(head_dropout) for _ in range(n_vars)])
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])   # [bs x d_model * num_patch]
                z = self.linears[i](z)                  # [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)              # [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)                        # [bs x nvars x (d_model * num_patch)]
            x = self.dropout(x)
            x = self.linear(x)                         # [bs x nvars x forecast_len]
        return x.transpose(2, 1)                       # [bs x forecast_len x nvars]
