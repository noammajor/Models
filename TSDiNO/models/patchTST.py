
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from models.layers.pos_encoding import *
from models.layers.basics import *
from utils.util import get_activation_fn
from models.layers.attention import *
from utils.util import DropPath
from torch.nn.init import trunc_normal_
from models.layers.revin import RevIN


            
# Cell
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, num_patch:int, 
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout = 0, 
                 head_type = "prediction", individual = False, 
                 y_range:Optional[tuple]=None, verbose:bool=False,step_size:int=12, head = None,operation='train', **kwargs):

        super().__init__()

        #assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose,step_size=step_size, **kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.patch_len = patch_len
        self.num_patch = num_patch
        self.normalization = RevIN(c_in, affine=True)
        self.step_size = step_size
        self.operation = operation

        if head_type == "Dino":
            self.head = None
        if head_type == "CLS_Prediction":
            self.head = CLSPredictionHead(self.n_vars, d_model, target_dim, head_dropout)
        elif head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(False, self.n_vars, d_model, num_patch + 1, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)


    def forward(self, z, padding_mask=None):
        """
        z: tensor [bs x seq_len x n_vars]  (flat) or [bs x num_patch x n_vars x patch_len] (pre-patched)
        padding_mask: [bs x num_patch] bool, True=real data (only used during classification)
        """
        if self.head_type in ['prediction', 'regression', 'classification', 'CLS_Prediction']:
            z = self.normalization(z, mode='norm')   # instance-normalize before encoder
            patches_tensor = z.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        else:
            patches_tensor = z.unfold(dimension=1, size=self.patch_len, step=self.step_size)
        z = self.backbone(patches_tensor, padding_mask=padding_mask)
        if self.head_type == "Dino":
            z = z[:, 0, :, :]                                                            # z: [bs x nvars x d_model]
            return z
        elif self.head_type == "CLS_Prediction":
            z = z[:, 0, :, :]
            z = self.head(z)
        else:
            z = z  # Keep all tokens: [bs x num_patch+1 x nvars x d_model]
            z = z.permute(0, 2, 3, 1)  # [bs x nvars x d_model x num_patch+1]
            z = self.head(z)
        if self.head_type not in ('classification', 'CLS_Prediction'):
            z = self.normalization(z, mode='denorm')   # undo instance norm on forecast output
        return z

    def forward_recon(self, z):
        """Encode z and return all patch token embeddings (no CLS) for reconstruction.

        z:       [bs, seq_len, n_vars]
        returns: [bs, num_patch, n_vars, d_model]
        Uses non-overlapping patches (step = patch_len). No RevIN applied.
        """
        patches_tensor = z.unfold(dimension=1, size=self.patch_len, step=self.patch_len)
        tokens = self.backbone(patches_tensor)   # [bs, num_patch+1, n_vars, d_model]
        return tokens[:, 1:, :, :]              # drop CLS  [bs, num_patch, n_vars, d_model]


class PatchReconDecoder(nn.Module):
    """Maps patch token embeddings back to raw patch values.

    Input:  [N, num_patch, d_model]  (N = bs * n_vars)
    Output: [N, num_patch, patch_len]
    """
    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, patch_len),
        )

    def forward(self, x):
        return self.decoder(x)


class RegressionHead(nn.Module):
    def __init__(self, n_vars, d_model, output_dim, head_dropout, y_range=None):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, output_dim)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x output_dim]
        """
        x = x[:,:,:,-1]             # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x output_dim
        if self.y_range: y = SigmoidRange(*self.y_range)(y)        
        return y


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars*d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch+1]
        output: [bs x n_classes]
        """
        x = x[:,:,:,0]              # CLS token at position 0 (trained by DINO), x: bs x nvars x d_model
        x = self.flatten(x)         # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)         # y: bs x n_classes
        return y
import torch
import torch.nn as nn
'''
class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model * num_patch
        
        # Define two hidden dimensions to create a bottleneck or expansive structure
        hidden_dim1 = head_dim // 2
        hidden_dim2 = head_dim // 4

        if self.individual:
            self.mlps = nn.ModuleList()
            for i in range(self.n_vars):
                # Individual MLP for each variable
                self.mlps.append(nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(head_dim, hidden_dim1),
                    nn.GELU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(hidden_dim1, hidden_dim2),
                    nn.GELU(),
                    nn.Linear(hidden_dim2, forecast_len)
                ))
        else:
            self.flatten_layer = nn.Flatten(start_dim=-2)
            # Shared MLP for all variables
            self.mlp = nn.Sequential(
                nn.Linear(head_dim, hidden_dim1),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.GELU(),
                nn.Linear(hidden_dim2, forecast_len)
            )

    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.mlps[i](x[:, i, :, :])
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # [bs x nvars x forecast_len]
        else:
            x = self.flatten_layer(x)              # [bs x nvars x head_dim]    
            x = self.mlp(x)                        # [bs x nvars x forecast_len]
            
        return x.transpose(2, 1)                  # [bs x forecast_len x nvars]

'''
class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
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
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)
            x = self.linear(x)      # x: [bs x nvars x forecast_len]
        return x.transpose(2,1)     # [bs x forecast_len x nvars]


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, 
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, step_size=12, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        #CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        # Initialize it with small random values
        trunc_normal_(self.cls_token, std=.02)        

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model) 
        
        #input_len = self.num_patch * self.patch_len 
        #real_num_patches = int((input_len - self.patch_len) / step_size) + 1     

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, self.num_patch+1, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

    def _get_pos_enc(self, seq_len: int) -> Tensor:
        """Return positional encoding for seq_len tokens.

        If seq_len <= W_pos size: slice.
        If seq_len >  W_pos size: interpolate (bilinear) so any input length works.
        """
        W = self.W_pos          # [max_len, d_model]
        if seq_len == W.shape[0]:
            return W
        if seq_len < W.shape[0]:
            return W[:seq_len, :]
        # Interpolate up to seq_len
        W = W.unsqueeze(0).unsqueeze(0)          # [1, 1, max_len, d_model]
        W = F.interpolate(W, size=(seq_len, W.shape[-1]), mode='bilinear', align_corners=False)
        return W.squeeze(0).squeeze(0)           # [seq_len, d_model]

    def forward(self, x, padding_mask=None) -> Tensor:
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        padding_mask: [bs x num_patch] bool, True=real data (optional, only for classification)
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )
        cls_tokens = self.cls_token.expand(u.shape[0], -1, -1)
        u = torch.cat((cls_tokens, u), dim=1)
        u = self.dropout(u + self._get_pos_enc(u.shape[1]))

        # Build key_padding_mask for attention (True = ignore token)
        key_padding_mask = None
        if padding_mask is not None:
            pm = padding_mask.to(u.device).bool()              # [bs, num_patch], True=real
            pm_exp = pm.unsqueeze(1).expand(-1, n_vars, -1).reshape(bs * n_vars, num_patch)
            # CLS token is always real → False (not ignored); patch tokens: ignore if not real
            cls_col = torch.zeros(bs * n_vars, 1, dtype=torch.bool, device=u.device)
            key_padding_mask = torch.cat([cls_col, ~pm_exp], dim=1)  # [bs*n_vars, num_patch+1]

        # Encoder
        z = self.encoder(u, key_padding_mask=key_padding_mask)                  # z: [bs * nvars x num_patch+1 x d_model]
        #z = z[:, 0, :]                                                           # [bs*nvars x d_model]
        #z = torch.reshape(z, (bs, n_vars, self.d_model))
        z = torch.reshape(z, (bs, n_vars, -1, self.d_model))

        z = z.permute(0, 2, 1, 3)  # [bs x num_patch x nvars x d_model]
        return z
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        key_padding_mask: [bs x q_len] bool, True=ignore (optional)
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False,
                drop_path=0.):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn
        #maybe will be used later
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        key_padding_mask: [bs x q_len] bool, True=ignore (optional)
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.drop_path(self.dropout_attn(src2)) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.drop_path(self.dropout_ffn(src2)) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class CLSPredictionHead(nn.Module):
    def __init__(self, n_vars, d_model, forecast_len, head_dropout=0, hidden_dim=512):
        super().__init__()
        self.n_vars = n_vars
        self.layer1 = nn.Linear(d_model, hidden_dim)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(head_dropout)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(head_dropout)
        self.layer3 = nn.Linear(hidden_dim, forecast_len)

    def forward(self, x):                     
        x = self.layer1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        return x.transpose(2, 1)