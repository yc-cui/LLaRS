"""
Deterministic degradations for evaluation.

Unlike training ops, these use fixed parameters (no random sampling);
each function maps to one strength for controlled quantitative tests.

Naming:
  - gaussian_noise_sigma_XXX: Gaussian noise, sigma = 0.XXX
  - gaussian_blur_kN_sXX:     Gaussian blur, kernel=N, sigma=X.X
  - bicubic_downsample_xN:    Bicubic downsample factor N
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor
RNG = np.random.Generator


# ============================================================================
# Gaussian noise (fixed sigma)
# ============================================================================

def gaussian_noise_sigma_001(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian noise, sigma=0.01."""
    sigma = 0.01
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)
    ).to(clean.device)
    return clean + noise, clean


def gaussian_noise_sigma_005(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian noise, sigma=0.05."""
    sigma = 0.05
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)
    ).to(clean.device)
    return clean + noise, clean


def gaussian_noise_sigma_010(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian noise, sigma=0.10."""
    sigma = 0.10
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)
    ).to(clean.device)
    return clean + noise, clean


def gaussian_noise_sigma_025(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian noise, sigma=0.25."""
    sigma = 0.25
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)
    ).to(clean.device)
    return clean + noise, clean


def gaussian_noise_sigma_050(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian noise, sigma=0.50."""
    sigma = 0.50
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)
    ).to(clean.device)
    return clean + noise, clean


# ============================================================================
# Gaussian blur (fixed kernel + sigma)
# ============================================================================

def gaussian_blur_k3_s05(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian blur, kernel=3, sigma=0.5."""
    import kornia
    inp = kornia.filters.gaussian_blur2d(
        clean.unsqueeze(0), (3, 3), (0.5, 0.5)
    ).squeeze(0)
    return inp, clean


def gaussian_blur_k5_s10(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian blur, kernel=5, sigma=1.0."""
    import kornia
    inp = kornia.filters.gaussian_blur2d(
        clean.unsqueeze(0), (5, 5), (1.0, 1.0)
    ).squeeze(0)
    return inp, clean


def gaussian_blur_k7_s20(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian blur, kernel=7, sigma=2.0."""
    import kornia
    inp = kornia.filters.gaussian_blur2d(
        clean.unsqueeze(0), (7, 7), (2.0, 2.0)
    ).squeeze(0)
    return inp, clean


def gaussian_blur_k9_s30(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Gaussian blur, kernel=9, sigma=3.0."""
    import kornia
    inp = kornia.filters.gaussian_blur2d(
        clean.unsqueeze(0), (9, 9), (3.0, 3.0)
    ).squeeze(0)
    return inp, clean


# ============================================================================
# Motion blur (fixed kernel size, angle=45 deg)
# ============================================================================

def _motion_blur_fixed(clean: Tensor, kernel_size: int, angle: float = 45.0) -> Tensor:
    import kornia
    return kornia.filters.motion_blur(
        clean.unsqueeze(0), kernel_size, angle, direction=0.0
    ).squeeze(0)


def motion_blur_k5(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Motion blur, kernel=5, angle=45 deg."""
    return _motion_blur_fixed(clean, 5), clean


def motion_blur_k7(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Motion blur, kernel=7, angle=45 deg."""
    return _motion_blur_fixed(clean, 7), clean


def motion_blur_k9(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Motion blur, kernel=9, angle=45 deg."""
    return _motion_blur_fixed(clean, 9), clean


def motion_blur_k11(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Motion blur, kernel=11, angle=45 deg."""
    return _motion_blur_fixed(clean, 11), clean


# ============================================================================
# Bicubic downsample (fixed scale)
# ============================================================================

def _bicubic_downsample(clean: Tensor, scale: float) -> Tensor:
    """Bicubic downsample then upsample back to original size."""
    _, h, w = clean.shape
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    down = F.interpolate(
        clean.unsqueeze(0), size=(new_h, new_w),
        mode="bicubic", align_corners=False,
    ).squeeze(0)
    up = F.interpolate(
        down.unsqueeze(0), size=(h, w),
        mode="bicubic", align_corners=False,
    ).squeeze(0)
    return torch.clamp(up, 0, 1)


def bicubic_downsample_x2(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Bicubic downsample x2 (scale=0.5)."""
    return _bicubic_downsample(clean, 0.5), clean


def bicubic_downsample_x4(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Bicubic downsample x4 (scale=0.25)."""
    return _bicubic_downsample(clean, 0.25), clean


def bicubic_downsample_x8(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Bicubic downsample x8 (scale=0.125)."""
    return _bicubic_downsample(clean, 0.125), clean


# ============================================================================
# Salt-and-pepper noise (fixed probability)
# ============================================================================

def salt_pepper_noise_p001(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Salt-and-pepper noise, prob=0.01."""
    prob = 0.01
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    inp = clean.clone()
    inp = inp.masked_fill((mask < prob / 2).expand_as(clean), 1.0)
    inp = inp.masked_fill((mask > 1 - prob / 2).expand_as(clean), 0.0)
    return inp, clean


def salt_pepper_noise_p005(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Salt-and-pepper noise, prob=0.05."""
    prob = 0.05
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    inp = clean.clone()
    inp = inp.masked_fill((mask < prob / 2).expand_as(clean), 1.0)
    inp = inp.masked_fill((mask > 1 - prob / 2).expand_as(clean), 0.0)
    return inp, clean


def salt_pepper_noise_p010(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Salt-and-pepper noise, prob=0.10."""
    prob = 0.10
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    inp = clean.clone()
    inp = inp.masked_fill((mask < prob / 2).expand_as(clean), 1.0)
    inp = inp.masked_fill((mask > 1 - prob / 2).expand_as(clean), 0.0)
    return inp, clean

