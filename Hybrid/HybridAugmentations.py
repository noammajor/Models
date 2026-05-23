"""
Hybrid augmentation pipeline — union of LE-JEPA and TSDiNO augmentations.

All transforms operate on batched [B, T, C] float32 tensors.
Each is applied independently with p=0.5 per call.

LE-JEPA set:
  GaussianJitter, AmplitudeScaling, ChannelDropout, FrequencyMasking, DWTAugmentation

TSDiNO-exclusive set (adapted to batched tensors):
  PolarTransformation, RotationTransformation, BoostTransformation,
  LorentzTransformation, HyperbolicAmplitudeWarp, HyperbolicGeometry
"""

import random
import numpy as np
import torch
import torch.nn as nn

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# ── LE-JEPA augmentations ─────────────────────────────────────────────────────

class GaussianJitter:
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = random.uniform(0.0, self.noise_std)
        return x + torch.randn_like(x) * std


class AmplitudeScaling:
    """Per-sample amplitude scaling (same as galilien_transformation in DINO)."""
    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        scales = torch.empty(B, 1, 1, device=x.device, dtype=x.dtype).uniform_(*self.scale_range)
        return x * scales


class ChannelDropout:
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 1:
            return x
        B, T, C = x.shape
        mask = (torch.rand(B, 1, C, device=x.device) >= self.p).to(x.dtype)
        return x * mask


class FrequencyMasking:
    def __init__(self, mask_ratio: float = 0.3):
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.fft.rfft(x, dim=1)
        n_freqs = x_f.size(1)
        n_mask = max(1, int(n_freqs * random.uniform(0.0, self.mask_ratio)))
        start = random.randint(0, n_freqs - n_mask)
        x_f[:, start:start + n_mask, :] = 0.0
        return torch.fft.irfft(x_f, n=x.size(1), dim=1)


class DWTAugmentation:
    def __init__(self, wavelet='db4', level=3, mode='zero_out_detail',
                 soft_threshold_sigma=0.3, zero_out_ratio=0.3, finest_levels=1,
                 high_perturb_noise_range=(0.03, 0.08),
                 band_scale_approx_range=(0.9, 1.1),
                 band_scale_detail_range=(0.6, 1.4)):
        if not _HAS_PYWT:
            raise ImportError("DWTAugmentation requires pywt: pip install PyWavelets")
        self.wavelet                  = wavelet
        self.level                    = level
        self.mode                     = mode
        self.soft_threshold_sigma     = soft_threshold_sigma
        self.zero_out_ratio           = zero_out_ratio
        self.finest_levels            = finest_levels
        self.high_perturb_noise_range = high_perturb_noise_range
        self.band_scale_approx_range  = band_scale_approx_range
        self.band_scale_detail_range  = band_scale_detail_range

    def _soft_thresh(self, c, sigma):
        threshold = sigma * np.abs(c).max() if c.size > 0 else 0.0
        return np.sign(c) * np.maximum(np.abs(c) - threshold, 0.0)

    def _apply_single(self, x_np):
        T, C = x_np.shape
        result = np.zeros_like(x_np)
        for v in range(C):
            coeffs = pywt.wavedec(x_np[:, v], self.wavelet, level=self.level)
            if self.mode == 'low_pass':
                new_c = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            elif self.mode == 'soft_threshold':
                new_c = [coeffs[0]] + [self._soft_thresh(c, self.soft_threshold_sigma) for c in coeffs[1:]]
            elif self.mode == 'zero_out_detail':
                new_c = list(coeffs)
                for ci in range(len(coeffs) - self.finest_levels, len(coeffs)):
                    c = new_c[ci].copy()
                    c[np.random.rand(*c.shape) < self.zero_out_ratio] = 0.0
                    new_c[ci] = c
            elif self.mode == 'high_perturb':
                scale = random.uniform(*self.high_perturb_noise_range)
                new_c = [coeffs[0]] + [c + np.random.randn(*c.shape) * scale for c in coeffs[1:]]
            elif self.mode == 'band_scale':
                new_c = ([coeffs[0] * random.uniform(*self.band_scale_approx_range)] +
                         [c * random.uniform(*self.band_scale_detail_range) for c in coeffs[1:]])
            else:
                raise ValueError(f"Unknown DWT mode: {self.mode}")
            rec = pywt.waverec(new_c, self.wavelet)
            result[:, v] = rec[:T]
        return result

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        x_np = x.cpu().numpy()
        out  = np.stack([self._apply_single(x_np[b]) for b in range(x_np.shape[0])], axis=0)
        return torch.tensor(out, dtype=dtype, device=device)


# ── TSDiNO-exclusive augmentations (batched) ──────────────────────────────────

class PolarTransformation:
    """Map (t, x) → polar coords, warp angle, map back. x: [B, T, C]"""
    def __init__(self, warp_range=(0.7, 1.3)):
        self.warp_range = warp_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        warp_factor = random.uniform(*self.warp_range)
        t = torch.linspace(0, 1, steps=T, device=x.device).view(1, T, 1)
        r     = torch.sqrt(t ** 2 + x ** 2)
        theta = torch.atan2(x, t) * warp_factor
        return r * torch.sin(theta)


class RotationTransformation:
    """Rotate (t, x) plane by a small angle. x: [B, T, C]"""
    def __init__(self, angle_range=(0, np.pi / 8)):
        self.angle_range = angle_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        angle = random.uniform(*self.angle_range)
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
        t = torch.linspace(0, 1, steps=T, device=x.device).view(1, T, 1)
        return t * sin_a + x * cos_a


class BoostTransformation:
    """Add a linear drift b·t to the series. x: [B, T, C]"""
    def __init__(self, b_range=(0.01, 0.3)):
        self.b_range = b_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        b = random.uniform(*self.b_range)
        t = torch.linspace(0, 1, steps=T, device=x.device).view(1, T, 1)
        return x + b * t


class LorentzTransformation:
    """Special-relativistic Lorentz contraction along the time axis. x: [B, T, C]"""
    def __init__(self, v_range=(0.2, 0.6)):
        self.v_range = v_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        v     = random.uniform(*self.v_range)
        gamma = 1.0 / float(np.sqrt(1 - v ** 2))
        t     = torch.linspace(0, 1, steps=T, device=x.device).view(1, T, 1)
        return gamma * (x - v * t)


class HyperbolicAmplitudeWarp:
    """Tanh-compress the amplitude with a random warp factor. x: [B, T, C]"""
    def __init__(self, warp_range=(0.5, 1.5)):
        self.warp_range = warp_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        warp_factor = random.uniform(*self.warp_range)
        return torch.tanh(x * warp_factor)


class HyperbolicGeometry:
    """
    Möbius transformation on the Poincaré disk.
    Projects (t, x) onto the disk, applies a random Möbius shift, returns y-component.
    x: [B, T, C]
    """
    def __init__(self, shift_magnitude=0.3, eps=1e-8):
        self.shift_magnitude = shift_magnitude
        self.eps = eps

    def _to_poincare(self, x):
        B, T, C = x.shape
        t     = torch.linspace(-0.9, 0.9, steps=T, device=x.device).view(1, T, 1)
        y_min = x.min(dim=1, keepdim=True)[0]
        y_max = x.max(dim=1, keepdim=True)[0]
        y     = 1.8 * (x - y_min) / (y_max - y_min + self.eps) - 0.9
        return t, y

    def _mobius_add(self, u, v, u0, v0):
        norm_z0_sq = u0 ** 2 + v0 ** 2
        norm_z_sq  = u ** 2 + v ** 2
        inner_prod = u0 * u + v0 * v
        denom  = 1 + 2 * inner_prod + norm_z0_sq * norm_z_sq
        num_v  = (1 + 2 * inner_prod + norm_z_sq) * v0 + (1 - norm_z0_sq) * v
        return num_v / (denom + self.eps)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t, y = self._to_poincare(x)
        z0 = torch.randn(2, device=x.device)
        z0 = self.shift_magnitude * z0 / (z0.norm() + self.eps)
        u0, v0 = z0[0].item(), z0[1].item()
        return self._mobius_add(t, y, u0, v0)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class HybridAugmentationPipeline:
    """
    Union of LE-JEPA and TSDiNO augmentations.

    Each augmentation is applied independently with p=0.5.
    Like LE-JEPA, two pipelines are created — one per view — with different DWT modes:
        aug_v1 = HybridAugmentationPipeline(config, dwt_mode=config["view1_dwt_mode"])
        aug_v2 = HybridAugmentationPipeline(config, dwt_mode=config["view2_dwt_mode"])
    """

    def __init__(self, config: dict, dwt_mode: str = None):
        self.augs = [
            # ── LE-JEPA ────────────────────────────────────────────────────────
            GaussianJitter(noise_std=config.get("aug_noise_std", 0.05)),
            AmplitudeScaling(scale_range=config.get("aug_amplitude_range", (0.8, 1.2))),
            ChannelDropout(p=config.get("aug_channel_drop_p", 0.2)),
            FrequencyMasking(mask_ratio=config.get("aug_freq_mask_ratio", 0.3)),
            # ── TSDiNO-exclusive ───────────────────────────────────────────────
            PolarTransformation(warp_range=config.get("aug_polar_warp_range", (0.7, 1.3))),
            RotationTransformation(angle_range=config.get("aug_rotation_angle_range", (0, np.pi / 8))),
            BoostTransformation(b_range=config.get("aug_boost_b_range", (0.01, 0.3))),
            LorentzTransformation(v_range=config.get("aug_lorentz_v_range", (0.2, 0.6))),
            HyperbolicAmplitudeWarp(warp_range=config.get("aug_hyperbolic_warp_range", (0.5, 1.5))),
            HyperbolicGeometry(shift_magnitude=config.get("aug_hyperbolic_shift_magnitude", 0.3)),
        ]
        # DWT (shared between DINO and LE-JEPA, mode controls which view)
        if dwt_mode is not None:
            if not _HAS_PYWT:
                print("[Hybrid] Warning: dwt_mode set but pywt not installed — DWT disabled.")
            else:
                self.augs.append(DWTAugmentation(
                    wavelet                  = config.get("dwt_wavelet", "db4"),
                    level                    = config.get("dwt_level", 3),
                    mode                     = dwt_mode,
                    soft_threshold_sigma     = config.get("dwt_soft_threshold_sigma", 0.3),
                    zero_out_ratio           = config.get("dwt_zero_out_ratio", 0.4),
                    finest_levels            = config.get("dwt_finest_levels", 2),
                    high_perturb_noise_range = config.get("dwt_high_perturb_noise_range", (0.1, 0.3)),
                    band_scale_approx_range  = config.get("dwt_band_scale_approx_range", (0.80, 1.20)),
                    band_scale_detail_range  = config.get("dwt_band_scale_detail_range", (0.40, 1.60)),
                ))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        for aug in self.augs:
            if random.random() < 0.5:
                x = aug(x)
        return x
