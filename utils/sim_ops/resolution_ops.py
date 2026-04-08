from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor
RNG = np.random.Generator


# ============================================================================
# Resolution / Downsample Functions
# ============================================================================

def bicubic_downsample_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak bicubic downsample (scale=0.7)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.7), int(w * 0.7)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False).squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def bicubic_downsample_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium bicubic downsample (scale=0.5)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.5), int(w * 0.5)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False).squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def bicubic_downsample_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong bicubic downsample (scale=0.2)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.2), int(w * 0.2)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False).squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def bilinear_downsample_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak bilinear downsample (scale=0.7)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.7), int(w * 0.7)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def bilinear_downsample_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium bilinear downsample (scale=0.5)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.5), int(w * 0.5)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def bilinear_downsample_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong bilinear downsample (scale=0.2)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.2), int(w * 0.2)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def nearest_downsample_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak nearest downsample (scale=0.7)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.7), int(w * 0.7)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return inp, clean


def nearest_downsample_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium nearest downsample (scale=0.5)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.5), int(w * 0.5)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return inp, clean


def nearest_downsample_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong nearest downsample (scale=0.2)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.2), int(w * 0.2)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    return inp, clean


def area_downsample_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak area downsample (scale=0.7)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.7), int(w * 0.7)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="area").squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def area_downsample_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium area downsample (scale=0.5)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.5), int(w * 0.5)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="area").squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def area_downsample_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong area downsample (scale=0.2)."""
    _, h, w = clean.shape
    new_h, new_w = int(h * 0.2), int(w * 0.2)
    downsampled = F.interpolate(clean.unsqueeze(0), size=(new_h, new_w), mode="area").squeeze(0)
    inp = F.interpolate(downsampled.unsqueeze(0), size=(h, w), mode="bicubic", align_corners=False).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


