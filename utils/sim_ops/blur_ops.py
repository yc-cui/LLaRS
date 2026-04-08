from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import kornia

Tensor = torch.Tensor
RNG = np.random.Generator


# ============================================================================
# Blur Functions
# ============================================================================

def gaussian_blur_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak Gaussian blur (kernel=3, sigma=0.5)."""
    inp = kornia.filters.gaussian_blur2d(clean.unsqueeze(0), (3, 3), (0.5, 0.5)).squeeze(0)
    return inp, clean


def gaussian_blur_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium Gaussian blur (kernel=5, sigma=1.0)."""
    inp = kornia.filters.gaussian_blur2d(clean.unsqueeze(0), (5, 5), (1.0, 1.0)).squeeze(0)
    return inp, clean


def gaussian_blur_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong Gaussian blur (kernel=7, sigma=2.0)."""
    inp = kornia.filters.gaussian_blur2d(clean.unsqueeze(0), (7, 7), (2.0, 2.0)).squeeze(0)
    return inp, clean


def motion_blur_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak motion blur (kernel=7)."""
    angle = float(rng.uniform(0, 360))
    inp = kornia.filters.motion_blur(clean.unsqueeze(0), 7, angle, direction=1.0).squeeze(0)
    return inp, clean


def motion_blur_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium motion blur (kernel=9)."""
    angle = float(rng.uniform(0, 360))
    inp = kornia.filters.motion_blur(clean.unsqueeze(0), 9, angle, direction=1.0).squeeze(0)
    return inp, clean


def motion_blur_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong motion blur (kernel=11)."""
    angle = float(rng.uniform(0, 360))
    inp = kornia.filters.motion_blur(clean.unsqueeze(0), 11, angle, direction=1.0).squeeze(0)
    return inp, clean


def box_blur_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak box blur (kernel=3)."""
    inp = kornia.filters.box_blur(clean.unsqueeze(0), (3, 3)).squeeze(0)
    return inp, clean


def box_blur_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium box blur (kernel=5)."""
    inp = kornia.filters.box_blur(clean.unsqueeze(0), (5, 5)).squeeze(0)
    return inp, clean


def box_blur_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong box blur (kernel=7)."""
    inp = kornia.filters.box_blur(clean.unsqueeze(0), (7, 7)).squeeze(0)
    return inp, clean


def median_blur_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak median blur (kernel=3)."""
    inp = kornia.filters.median_blur(clean.unsqueeze(0), (3, 3)).squeeze(0)
    return inp, clean


def median_blur_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium median blur (kernel=5)."""
    inp = kornia.filters.median_blur(clean.unsqueeze(0), (5, 5)).squeeze(0)
    return inp, clean


def median_blur_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong median blur (kernel=7)."""
    inp = kornia.filters.median_blur(clean.unsqueeze(0), (7, 7)).squeeze(0)
    return inp, clean


# ============================================================================
# Defocus (Disk) Blur
# ============================================================================

def _disk_kernel(radius: int) -> Tensor:
    """Generate a normalized circular disk kernel."""
    size = 2 * radius + 1
    y, x = torch.meshgrid(torch.arange(size) - radius,
                           torch.arange(size) - radius, indexing="ij")
    mask = (x.float() ** 2 + y.float() ** 2 <= radius ** 2).float()
    return mask / mask.sum()


def _apply_disk_blur(clean: Tensor, radius: int) -> Tensor:
    """Apply disk blur to each channel via 2-D convolution."""
    kernel = _disk_kernel(radius).to(clean.device)
    C = clean.shape[0]
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)  # (C,1,K,K)
    pad = radius
    return F.conv2d(clean.unsqueeze(0), kernel_4d, padding=pad, groups=C).squeeze(0)


def defocus_blur_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak defocus blur (disk radius=2)."""
    inp = _apply_disk_blur(clean, radius=2)
    return inp, clean


def defocus_blur_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium defocus blur (disk radius=4)."""
    inp = _apply_disk_blur(clean, radius=4)
    return inp, clean


def defocus_blur_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong defocus blur (disk radius=7)."""
    inp = _apply_disk_blur(clean, radius=7)
    return inp, clean


