import numpy as np
import torch
import random
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import pywt

class DWTAugmentation:
    def __init__(self, wavelet='db4', level=3, mode='low_pass',
                 soft_threshold_sigma=0.3,
                 zero_out_ratio=0.3,
                 finest_levels=1,
                 high_perturb_noise_range=(0.03, 0.08),
                 band_scale_approx_range=(0.9, 1.1),
                 band_scale_detail_range=(0.6, 1.4),
                 wavelet_pool=None):
        """
        Discrete Wavelet Transform augmentation for time series.

        wavelet_pool: optional list of wavelet names to sample from randomly each call,
                      e.g. ['db4', 'sym4']. When set, overrides `wavelet`.

        mode:
          'low_pass'        – zero all detail coefficients (smooth global view).
          'soft_threshold'  – soft-threshold all detail coefficients; removes
                              small-magnitude high-freq components while keeping
                              dominant structure. Good for stable teacher views.
                              threshold = soft_threshold_sigma * max(|coeffs|) per level.
          'zero_out_detail' – randomly zero out zero_out_ratio fraction of detail
                              coefficients at the finest `finest_levels` levels.
                              Drops fine-scale features stochastically.
          'high_perturb'    – add Gaussian noise to all detail coefficients.
          'band_scale'      – randomly scale each frequency band independently.
        """
        self.wavelet                  = wavelet
        self.wavelet_pool             = wavelet_pool
        self.level                    = level
        self.mode                     = mode
        self.soft_threshold_sigma     = soft_threshold_sigma
        self.zero_out_ratio           = zero_out_ratio
        self.finest_levels            = finest_levels
        self.high_perturb_noise_range = high_perturb_noise_range
        self.band_scale_approx_range  = band_scale_approx_range
        self.band_scale_detail_range  = band_scale_detail_range

    def _pick_wavelet(self):
        if self.wavelet_pool:
            return random.choice(self.wavelet_pool)
        return self.wavelet

    def _soft_thresh(self, c, sigma):
        """Adaptive soft thresholding: threshold = sigma * max(|c|)."""
        threshold = sigma * np.abs(c).max() if c.size > 0 else 0.0
        return np.sign(c) * np.maximum(np.abs(c) - threshold, 0.0)

    def __call__(self, x):
        # x: [seq_len, n_vars] tensor
        device, dtype = x.device, x.dtype
        wavelet = self._pick_wavelet()
        x_np = x.cpu().numpy()
        seq_len, n_vars = x_np.shape
        result = np.zeros_like(x_np)

        for v in range(n_vars):
            coeffs = pywt.wavedec(x_np[:, v], wavelet, level=self.level)
            # coeffs[0]  = approximation (low-freq)
            # coeffs[1:] = detail levels, finest last (coeffs[-1] = finest)

            if self.mode == 'low_pass':
                new_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]

            elif self.mode == 'soft_threshold':
                new_coeffs = [coeffs[0]] + [
                    self._soft_thresh(c, self.soft_threshold_sigma)
                    for c in coeffs[1:]
                ]

            elif self.mode == 'zero_out_detail':
                new_coeffs = list(coeffs)
                for c_idx in range(len(coeffs) - self.finest_levels, len(coeffs)):
                    c = new_coeffs[c_idx].copy()
                    mask = np.random.rand(*c.shape) < self.zero_out_ratio
                    c[mask] = 0.0
                    new_coeffs[c_idx] = c

            elif self.mode == 'high_perturb':
                noise_scale = random.uniform(*self.high_perturb_noise_range)
                new_coeffs = [coeffs[0]] + [
                    c + np.random.randn(*c.shape) * noise_scale
                    for c in coeffs[1:]
                ]

            elif self.mode == 'high_perturb_zero':
                noise_scale = random.uniform(*self.high_perturb_noise_range)
                new_coeffs = [coeffs[0]] + [
                    c + np.random.randn(*c.shape) * noise_scale
                    for c in coeffs[1:]
                ]
                for c_idx in range(len(coeffs) - self.finest_levels, len(coeffs)):
                    c = new_coeffs[c_idx].copy()
                    mask = np.random.rand(*c.shape) < self.zero_out_ratio
                    c[mask] = 0.0
                    new_coeffs[c_idx] = c

            elif self.mode == 'band_scale':
                new_coeffs = [coeffs[0] * random.uniform(*self.band_scale_approx_range)] + [
                    c * random.uniform(*self.band_scale_detail_range) for c in coeffs[1:]
                ]

            else:
                raise ValueError(f"Unknown DWT mode: {self.mode}")

            rec = pywt.waverec(new_coeffs, wavelet)
            result[:, v] = rec[:seq_len]  # waverec may produce 1 extra sample

        return torch.tensor(result, dtype=dtype, device=device)


class polar_transformation:
    def __init__(self, warp_range=(0.7, 1.3)):
        self.warp_range = warp_range

    def __call__(self, x):
        seq_len, n_vars = x.shape
        device = x.device
        warp_factor = random.uniform(*self.warp_range)
        
        t = torch.linspace(0, 1, steps=seq_len).to(device).unsqueeze(1).expand(-1, n_vars)
        r = torch.sqrt(t**2 + x**2)
        theta = torch.atan2(x, t) * warp_factor
        return r * torch.sin(theta)
class galilien_transformation:
    def __init__(self, a_range=(0.8, 1.2)):
        self.a_range = a_range

    def __call__(self, x):
        a = random.uniform(*self.a_range)
        return x * a
        
class rotation_transformation:
    def __init__(self, angle_range=(0, np.pi/8)): # Reduced range for stability
        self.angle_range = angle_range

    def __call__(self, x):
        seq_len, n_vars = x.shape
        angle = random.uniform(*self.angle_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        t = torch.linspace(0, 1, steps=seq_len).to(x.device).unsqueeze(1).expand(-1, n_vars)
        return (t * sin_a) + (x * cos_a)
        
class boost_transformation:
    def __init__(self, b_range=(0.01, 0.3)):
        self.b_range = b_range

    def __call__(self, x):
        seq_len, n_vars = x.shape
        b = random.uniform(*self.b_range)
        t = torch.linspace(0, 1, steps=seq_len).to(x.device).unsqueeze(1).expand(-1, n_vars)
        return x + (b * t)

class lorentz_transformation:
    def __init__(self, v_range=(0.2, 0.6)):
        self.v_range = v_range

    def __call__(self, x):
        seq_len, n_vars = x.shape
        v = random.uniform(*self.v_range)
        gamma = 1 / torch.sqrt(torch.tensor(1 - v**2, device=x.device))
        t = torch.linspace(0, 1, steps=seq_len, device=x.device).unsqueeze(1)
        x_new = gamma * (x - (v * t))
        return x_new

class hyperbolic_amplitude_warp:
    def __init__(self, warp_range=(0.5, 1.5)):
        self.warp_range = warp_range

    def __call__(self, x):
        device = x.device
        warp_factor = random.uniform(*self.warp_range)
        x_new = torch.tanh(x * warp_factor)
        return x_new

class HyperBolicGeometry(nn.Module):
    def __init__(self, shift_magnitude=0.3, eps=1e-8):
        super().__init__()
        self.shift_magnitude = shift_magnitude
        self.eps = eps

    def to_poincare(self, x):
        # x: [seq_len, n_vars] — time on dim 0, channels on dim 1
        seq_len, n_vars = x.shape
        t = torch.linspace(-0.9, 0.9, steps=seq_len, device=x.device).unsqueeze(1).expand(-1, n_vars)
        y_min = x.min(dim=0, keepdim=True)[0]
        y_max = x.max(dim=0, keepdim=True)[0]
        y = 1.8 * (x - y_min) / (y_max - y_min + self.eps) - 0.9
        return t, y

    def mobius_add(self, u, v, u0, v0):
        norm_z0_sq = u0**2 + v0**2
        norm_z_sq = u**2 + v**2
        inner_prod = u0*u + v0*v       
        denom = 1 + 2*inner_prod + norm_z0_sq * norm_z_sq
        num_v = (1 + 2*inner_prod + norm_z_sq) * v0 + (1 - norm_z0_sq) * v
        return num_v / (denom + self.eps)

    def __call__(self, x):
        t, y = self.to_poincare(x)
        z0 = torch.randn(2, 1, device=x.device) 
        z0 = self.shift_magnitude * z0 / (z0.norm() + self.eps)
        u0, v0 = z0[0], z0[1]
        y_new = self.mobius_add(t, y, u0, v0)  
        return y_new


# ── Vision-style augmentations (1D adaptations) ──────────────────────────────
class gaussian_blur:
    """Vision GaussianBlur analog for 1D series.

    Smooths each channel along the time axis with a Gaussian kernel of random
    sigma (drawn per sample). Encourages scale/high-frequency invariance, like
    the blur DINO applies to one of its views.

    x: [seq_len, n_vars] → same shape.
    """
    def __init__(self, sigma_range=(0.1, 2.0), truncate=3.0):
        self.sigma_range = sigma_range
        self.truncate    = truncate

    def __call__(self, x):
        seq_len, n_vars = x.shape
        sigma  = random.uniform(*self.sigma_range)
        if sigma <= 0:
            return x
        radius = max(1, int(self.truncate * sigma + 0.5))
        k = 2 * radius + 1
        if k >= seq_len:                       # kernel wider than the series → skip
            return x
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - radius
        kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel = (kernel / kernel.sum()).view(1, 1, k)
        inp = x.t().unsqueeze(1)               # [n_vars, 1, seq_len]
        inp = F.pad(inp, (radius, radius), mode='reflect')
        out = F.conv1d(inp, kernel)            # [n_vars, 1, seq_len]
        return out.squeeze(1).t()              # [seq_len, n_vars]


class gaussian_noise:
    """Pure additive Gaussian noise (SimCLR/DINO-style, no contrast/brightness).

    Adds i.i.d. N(0, std^2) noise to every timestep/channel, with std drawn per
    sample from std_range. Unlike jitter_contrast, this does NOT rescale or shift
    the signal — it only perturbs it, so the global structure is preserved while
    fine detail is corrupted. Data is expected pre-standardised, so std is absolute.

    x: [seq_len, n_vars] → same shape.
    """
    def __init__(self, std_range=(0.05, 0.2)):
        self.std_range = std_range

    def __call__(self, x):
        std = random.uniform(*self.std_range)
        if std <= 0:
            return x
        return x + torch.randn_like(x) * std


# DINO-vision "gaussiancrop" view = random crop (via spec crop_ratio) + this
# additive Gaussian noise. Alias so the config type name reads as one recipe.
gaussiancrop = gaussian_noise


class jitter_contrast:
    """SimCLR color-jitter analog for 1D series.

    Combines, per sample:
      • contrast   – scale about the per-channel temporal mean   (x-mean)*c + mean
      • brightness – additive offset                             + b
      • jitter     – additive Gaussian noise                     + N(0, std^2)

    x: [seq_len, n_vars] → same shape.
    """
    def __init__(self, jitter_range=(0.0, 0.1),
                 contrast_range=(0.7, 1.3),
                 brightness_range=(-0.2, 0.2)):
        self.jitter_range     = jitter_range
        self.contrast_range   = contrast_range
        self.brightness_range = brightness_range

    def __call__(self, x):
        c         = random.uniform(*self.contrast_range)
        b         = random.uniform(*self.brightness_range)
        noise_std = random.uniform(*self.jitter_range)
        mean = x.mean(dim=0, keepdim=True)     # per-channel mean over time
        out  = (x - mean) * c + mean + b
        if noise_std > 0:
            out = out + torch.randn_like(x) * noise_std
        return out
