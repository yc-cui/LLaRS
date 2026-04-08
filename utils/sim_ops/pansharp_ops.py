from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import kornia

Tensor = torch.Tensor
RNG = np.random.Generator


def simulate_pan(ms_img: Tensor, rng: RNG) -> Tensor:
    """Simulate panchromatic image by weighted channel fusion"""
    num_chans = ms_img.shape[0]
    if num_chans == 1:
        return ms_img
    n_select = int(rng.integers(2, num_chans + 1))
    selected_indices = rng.choice(num_chans, size=n_select, replace=False)
    selected = ms_img[selected_indices]
    weights = torch.from_numpy(rng.random(n_select).astype(np.float32)).to(ms_img.device)
    weights = weights / weights.sum()
    pan = (selected * weights[:, None, None]).sum(dim=0, keepdim=True)
    return pan


def blur_down(img: Tensor, scale: float = 0.25, ksize: tuple = (3, 3), sigma: tuple = (1.5, 1.5)) -> Tensor:
    """Blur and downsample"""
    blur = kornia.filters.gaussian_blur2d(img.unsqueeze(0), ksize, sigma).squeeze(0)
    return F.interpolate(blur.unsqueeze(0), scale_factor=scale, mode="bicubic", align_corners=True).squeeze(0)


# ============================================================================
# Pansharp Functions
# ============================================================================

def pansharp_degradation_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak pansharpening degradation (scale=0.5)."""
    pan_img = simulate_pan(clean, rng)
    ms_downsampled = blur_down(clean, scale=0.5, ksize=(3, 3), sigma=(1.0, 1.0))
    pan_downsampled = blur_down(pan_img, scale=0.5, ksize=(3, 3), sigma=(1.0, 1.0))
    ms_upsampled = F.interpolate(
        ms_downsampled.unsqueeze(0),
        size=(pan_downsampled.shape[1], pan_downsampled.shape[2]),
        mode="bicubic",
        align_corners=True,
    ).squeeze(0)
    inp = torch.cat([ms_upsampled, pan_downsampled], dim=0)
    # Match spatial size only; channels may differ from clean, padded in SimDatasetBase.__getitem__ to MAX_CHANS
    inp = F.interpolate(inp.unsqueeze(0), size=clean.shape[1:], mode="bicubic", align_corners=True).squeeze(0)
    return inp, clean


def pansharp_degradation_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium pansharpening degradation (scale=0.33)."""
    pan_img = simulate_pan(clean, rng)
    ms_downsampled = blur_down(clean, scale=0.33, ksize=(5, 5), sigma=(1.5, 1.5))
    pan_downsampled = blur_down(pan_img, scale=0.33, ksize=(5, 5), sigma=(1.5, 1.5))
    ms_upsampled = F.interpolate(
        ms_downsampled.unsqueeze(0),
        size=(pan_downsampled.shape[1], pan_downsampled.shape[2]),
        mode="bicubic",
        align_corners=True,
    ).squeeze(0)
    inp = torch.cat([ms_upsampled, pan_downsampled], dim=0)
    inp = F.interpolate(inp.unsqueeze(0), size=clean.shape[1:], mode="bicubic", align_corners=True).squeeze(0)
    return inp, clean


def pansharp_degradation_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong pansharpening degradation (scale=0.25)."""
    pan_img = simulate_pan(clean, rng)
    ms_downsampled = blur_down(clean, scale=0.25, ksize=(7, 7), sigma=(2.0, 2.0))
    pan_downsampled = blur_down(pan_img, scale=0.25, ksize=(7, 7), sigma=(2.0, 2.0))
    ms_upsampled = F.interpolate(
        ms_downsampled.unsqueeze(0),
        size=(pan_downsampled.shape[1], pan_downsampled.shape[2]),
        mode="bicubic",
        align_corners=True,
    ).squeeze(0)
    inp = torch.cat([ms_upsampled, pan_downsampled], dim=0)
    inp = F.interpolate(inp.unsqueeze(0), size=clean.shape[1:], mode="bicubic", align_corners=True).squeeze(0)
    return inp, clean


