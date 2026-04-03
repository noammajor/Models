import torch
import torch.nn.functional as F


def _sigreg_loss(x: torch.Tensor, global_step: int, num_slices: int = 256) -> torch.Tensor:
    """
    SIGReg: Gaussianity regularizer via the empirical characteristic function.
    Penalizes representations that deviate from N(0,1) using the Epps-Pulley statistic.

    x           : [B, n_patches, D] or [B, D] — will be reshaped to [N, D]
    global_step : used to seed the random projection matrix (reproducible slices)
    num_slices  : number of random 1-D projections

    Returns scalar loss.
    """
    x = x.reshape(-1, x.size(-1))          # [N, D]
    N, D = x.shape
    dev = dict(device=x.device, dtype=x.dtype)

    # Random projection matrix — same across calls with the same global_step
    g = torch.Generator(device=x.device)
    g.manual_seed(global_step)
    A = torch.randn(D, num_slices, generator=g, **dev)
    A = A / A.norm(p=2, dim=0, keepdim=True)   # [D, M] — unit columns

    # Integration points for Epps-Pulley statistic
    t = torch.linspace(-5, 5, 17, **dev)       # [T]

    # Theoretical CF for N(0,1): exp(-0.5 * t^2)
    exp_f = torch.exp(-0.5 * t ** 2)           # [T]

    # Empirical CF: E[exp(i * t * (x @ A))]
    x_proj = x @ A                             # [N, M]
    x_t    = x_proj.unsqueeze(2) * t           # [N, M, T]
    ecf    = torch.exp(1j * x_t).mean(0)       # [M, T]

    # Weighted L2 distance, integrated over t
    err = (ecf - exp_f).abs().square() * exp_f # [M, T]
    loss = torch.trapz(err, t, dim=1) * N      # [M]
    return loss.mean()


def _calculate_vicreg_loss(self, x: torch.Tensor):
    # x: [B*F, N_ctx, embed_dim]
    num_features = x.shape[-1]

    # Variance across batch — prevents inter-sample collapse
    std_batch = torch.sqrt(x.var(dim=0, unbiased=True) + 1e-4)   # [N_ctx, D]
    var_loss_batch = torch.mean(F.relu(1.0 - std_batch))

    # Variance across patch positions — prevents intra-sample positional collapse
    std_pos = torch.sqrt(x.var(dim=1, unbiased=True) + 1e-4)     # [B*F, D]
    var_loss_pos = torch.mean(F.relu(1.0 - std_pos))

    x_flat = x.reshape(-1, num_features)
    x_centered = x_flat - x_flat.mean(dim=0)
    cov = (x_centered.T @ x_centered) / (x_flat.shape[0] - 1)
    cov_loss = (cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()) / num_features

    return var_loss_pos,var_loss_batch, cov_loss