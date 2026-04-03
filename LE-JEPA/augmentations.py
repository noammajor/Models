"""
LE-JEPA augmentation pipeline.

All augmentations operate on [B, T, C] float32 tensors.
Each is stochastic — random parameters are sampled every call.
The pipeline applies each augmentation independently with p=0.5,
so both views receive different random transformations.

DWT augmentation is re-implemented from TSDiNO (same logic as DINO's
DWTAugmentation). The mode is controlled by config["dwt_mode"].
Set config["dwt_mode"] = None to disable DWT entirely.

DWT requires pywt: pip install PyWavelets
"""

import random
import numpy as np
import torch

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# ── Simple augmentations ──────────────────────────────────────────────────────

class GaussianJitter:
    """Additive Gaussian noise. std sampled uniformly in [0, noise_std]."""
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = random.uniform(0.0, self.noise_std)
        return x + torch.randn_like(x) * std


class AmplitudeScaling:
    """
    Per-sample amplitude scaling (from TSDiNO galilien_transformation).
    Each sample in the batch gets an independent scale factor drawn from [a, b].
    """
    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        scales = torch.empty(B, 1, 1, device=x.device, dtype=x.dtype).uniform_(*self.scale_range)
        return x * scales


class ChannelDropout:
    """
    Zero out entire channels with probability p (multivariate only).
    Each channel is dropped independently per sample.
    """
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 1:
            return x
        B, T, C = x.shape
        mask = (torch.rand(B, 1, C, device=x.device) >= self.p).to(x.dtype)
        return x * mask


class FrequencyMasking:
    """
    Mask a random contiguous band of FFT frequency bins.
    Applied uniformly across samples and channels in the batch.
    """
    def __init__(self, mask_ratio: float = 0.3):
        self.mask_ratio = mask_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_f = torch.fft.rfft(x, dim=1)          # [B, T//2+1, C]
        n_freqs = x_f.size(1)
        n_mask = max(1, int(n_freqs * random.uniform(0.0, self.mask_ratio)))
        start = random.randint(0, n_freqs - n_mask)
        x_f[:, start:start + n_mask, :] = 0.0
        return torch.fft.irfft(x_f, n=x.size(1), dim=1)


# ── DWT augmentation (re-implemented from TSDiNO) ────────────────────────────

class DWTAugmentation:
    """
    Discrete Wavelet Transform augmentation, re-implemented from
    TSDiNO/data_agumentation.py.  Requires pywt.

    Operates per sample in the batch (each [T, C] processed independently).

    mode options (same as DINO):
      'low_pass'        – zero all detail coefficients (smooth global view).
      'soft_threshold'  – soft-threshold detail coeffs; removes small-magnitude
                          high-frequency components while keeping dominant structure.
      'zero_out_detail' – randomly zero out zero_out_ratio fraction of the finest
                          detail levels stochastically.
      'high_perturb'    – add Gaussian noise to all detail coefficients.
      'band_scale'      – randomly scale each frequency band independently.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 3,
        mode: str = 'zero_out_detail',
        soft_threshold_sigma: float = 0.3,
        zero_out_ratio: float = 0.3,
        finest_levels: int = 1,
        high_perturb_noise_range: tuple = (0.03, 0.08),
        band_scale_approx_range: tuple = (0.9, 1.1),
        band_scale_detail_range: tuple = (0.6, 1.4),
    ):
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

    def _apply_single(self, x_np: np.ndarray) -> np.ndarray:
        """x_np: [T, C] → [T, C]"""
        T, C = x_np.shape
        result = np.zeros_like(x_np)
        for v in range(C):
            coeffs = pywt.wavedec(x_np[:, v], self.wavelet, level=self.level)
            if self.mode == 'low_pass':
                new_c = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            elif self.mode == 'soft_threshold':
                new_c = [coeffs[0]] + [
                    self._soft_thresh(c, self.soft_threshold_sigma) for c in coeffs[1:]
                ]
            elif self.mode == 'zero_out_detail':
                new_c = list(coeffs)
                for ci in range(len(coeffs) - self.finest_levels, len(coeffs)):
                    c = new_c[ci].copy()
                    c[np.random.rand(*c.shape) < self.zero_out_ratio] = 0.0
                    new_c[ci] = c
            elif self.mode == 'high_perturb':
                scale = random.uniform(*self.high_perturb_noise_range)
                new_c = [coeffs[0]] + [
                    c + np.random.randn(*c.shape) * scale for c in coeffs[1:]
                ]
            elif self.mode == 'band_scale':
                new_c = (
                    [coeffs[0] * random.uniform(*self.band_scale_approx_range)] +
                    [c * random.uniform(*self.band_scale_detail_range) for c in coeffs[1:]]
                )
            else:
                raise ValueError(f"Unknown DWT mode: {self.mode}")
            rec = pywt.waverec(new_c, self.wavelet)
            result[:, v] = rec[:T]
        return result

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        device, dtype = x.device, x.dtype
        x_np = x.cpu().numpy()
        out  = np.stack([self._apply_single(x_np[b]) for b in range(x_np.shape[0])], axis=0)
        return torch.tensor(out, dtype=dtype, device=device)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AugmentationPipeline:
    """
    Randomly applies each augmentation with p=0.5.

    Like DINO, the two views use different DWT modes:
      - view 1 (smooth): dwt_mode = config["view1_dwt_mode"]  (default: soft_threshold)
      - view 2 (perturbed): dwt_mode = config["view2_dwt_mode"] (default: high_perturb)

    Instantiate two pipelines — one per view — passing the appropriate mode:
        aug_v1 = AugmentationPipeline(config, dwt_mode=config["view1_dwt_mode"])
        aug_v2 = AugmentationPipeline(config, dwt_mode=config["view2_dwt_mode"])
    """

    def __init__(self, config: dict, dwt_mode: str = None):
        self.augs = [
            GaussianJitter(noise_std=config.get("aug_noise_std", 0.05)),
            AmplitudeScaling(scale_range=config.get("aug_amplitude_range", (0.8, 1.2))),
            ChannelDropout(p=config.get("aug_channel_drop_p", 0.2)),
            FrequencyMasking(mask_ratio=config.get("aug_freq_mask_ratio", 0.3)),
        ]
        if dwt_mode is not None:
            if not _HAS_PYWT:
                print("[LE-JEPA] Warning: dwt_mode set but pywt not installed — DWT disabled.")
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
