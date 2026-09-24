import torch
import torch.nn.functional as F


def _sigreg_loss(x: torch.Tensor, global_step: int, num_slices: int = 256) -> torch.Tensor:
    """
    SIGReg: Gaussianity regularizer via the empirical characteristic function.
    Penalizes representations that deviate from N(0,1) using the Epps-Pulley statistic.

    x           : [B, n_patches, D] or [B, D] — reshaped to [N, D]
    global_step : seeds the random projection matrix (reproducible slices per step)
    num_slices  : number of random 1-D projections

    Always cast to float32 regardless of AMP context.
    Returns scalar loss.
    """
    x = x.reshape(-1, x.size(-1)).float()   # [N, D], force float32
    N, D = x.shape
    dev = dict(device=x.device, dtype=x.dtype)

    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)
    A = torch.randn(D, num_slices, generator=g, **dev)
    A = A / A.norm(p=2, dim=0, keepdim=True)    # unit columns [D, M]

    t = torch.linspace(-5, 5, 17, **dev)        # [T]
    exp_f = torch.exp(-0.5 * t ** 2)            # theoretical CF of N(0,1)

    x_proj = x @ A                              # [N, M]
    x_t    = x_proj.unsqueeze(2) * t            # [N, M, T]
    ecf    = torch.exp(1j * x_t.to(torch.complex64)).mean(0)  # [M, T]

    err  = (ecf - exp_f).abs().square() * exp_f  # [M, T]
    loss = torch.trapz(err.real, t, dim=1) * N   # [M]
    return loss.mean()


def _vicreg_terms(x: torch.Tensor, eps: float = 1e-4):
    """VICReg variance + covariance regularizer (Eq. 9), as used by JEPA.

    Computed on the SAME per-patch embedding population SIGReg operates on —
    x: [B*C, P, D]. This is the drop-in isotropy-vs-VICReg swap: replacing
    _sigreg_loss with these terms (holding the encoder, augmentations, and the
    MSE invariance term fixed) isolates SIGReg's isotropic-Gaussianity
    constraint from Le-JEPA's augmentation stack.

    Returns (var_loss, cov_loss):
      - var_loss: hinge on per-position std across patches (prevents collapse)
      - cov_loss: off-diagonal covariance Frobenius penalty / D (decorrelation)
    """
    x = x.float()
    D = x.shape[-1]
    std_pos  = torch.sqrt(x.var(dim=1, unbiased=True) + eps)   # [B*C, D]
    var_loss = torch.mean(F.relu(1.0 - std_pos))
    x_flat   = x.reshape(-1, D)
    xc       = x_flat - x_flat.mean(dim=0)
    cov      = (xc.T @ xc) / (x_flat.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / D
    return var_loss, cov_loss
