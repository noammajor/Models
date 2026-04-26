__all__ = ['Transpose', 'LinBnDrop', 'SigmoidRange', 'sigmoid_range', 'get_activation_fn', 'ResidualBlock']
           

import torch
from torch import nn

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class SigmoidRange(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high   
        # self.low, self.high = ranges        
    def forward(self, x):                    
        # return sigmoid_range(x, self.low, self.high)
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm2d(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


def sigmoid_range(x, low, high):
    "Sigmoid function with range `(low, high)`"
    return torch.sigmoid(x) * (high - low) + low

def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    elif activation.lower() in ("silu", "swish"): return nn.SiLU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", "silu", or a callable')


class ResidualBlock(nn.Module):
    """Two-layer MLP with a residual (skip) connection from input to output."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int,
                 act_fn_name: str, dropout_p: float = 0.0,
                 use_layer_norm: bool = False) -> None:
        super().__init__()
        self.dropout       = nn.Dropout(dropout_p)
        self.hidden_layer  = nn.Linear(in_dim, h_dim)
        self.act           = get_activation_fn(act_fn_name)
        self.output_layer  = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        out = out + self.residual_layer(x)
        if self.use_layer_norm:
            out = self.layer_norm(out)
        return out


