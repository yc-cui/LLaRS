from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import kornia

Tensor = torch.Tensor
RNG = np.random.Generator


# ============================================================================
# Uniform stripe generator (original)
# ============================================================================

def generate_stripes(height, width, angle=0, density=0.5, stripe_width=5):
    """Generate stripe pattern with controllable angle, density and width"""
    x = torch.arange(width).float() - width / 2
    y = torch.arange(height).float() - height / 2
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    angle_rad = torch.deg2rad(torch.tensor(angle))
    stripe_dir = torch.cos(angle_rad) * xx + torch.sin(angle_rad) * yy
    stripe_pattern = (torch.sin(stripe_dir * (2 * torch.pi / (stripe_width / density))) > 0).float()
    return stripe_pattern


# ============================================================================
# Non-uniform stripe generator (variable width + per-band brightness)
# ============================================================================

def _generate_nonuniform_stripe_mask(
    length: int,
    rng: RNG,
    num_stripes_range: Tuple[int, int],
    width_range: Tuple[int, int],
    scale_range: Tuple[float, float],
) -> Tensor:
    """Build a 1-D per-pixel scale array with non-uniform stripe bands.

    Returns a 1-D tensor of shape (length,) where stripe regions have random
    scales drawn from *scale_range* and non-stripe regions are 1.0 (no change).
    """
    mask = torch.ones(length)
    num_stripes = int(rng.integers(*num_stripes_range))
    positions = rng.integers(0, length, size=num_stripes)
    for pos in positions:
        w = int(rng.integers(*width_range))
        lo = max(0, pos - w // 2)
        hi = min(length, lo + w)
        scale = float(rng.uniform(*scale_range))
        mask[lo:hi] = scale
    return mask


def _apply_nonuniform_stripes(
    clean: Tensor,
    rng: RNG,
    axis: str,
    num_stripes_range: Tuple[int, int],
    width_range: Tuple[int, int],
    scale_range: Tuple[float, float],
) -> Tensor:
    """Apply non-uniform stripes along the given axis ('h' or 'w')."""
    _, h, w = clean.shape
    if axis == "w":
        mask_1d = _generate_nonuniform_stripe_mask(w, rng, num_stripes_range, width_range, scale_range)
        mask = mask_1d.unsqueeze(0).unsqueeze(0).to(clean.device)         # (1, 1, W)
    else:
        mask_1d = _generate_nonuniform_stripe_mask(h, rng, num_stripes_range, width_range, scale_range)
        mask = mask_1d.unsqueeze(0).unsqueeze(-1).to(clean.device)        # (1, H, 1)
    return torch.clamp(clean * mask, 0, 1)


# ============================================================================
# Stripe Functions
# ============================================================================

def vertical_stripe_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak vertical stripes (density=0.05, width=1)."""
    _, h, w = clean.shape
    stripe = generate_stripes(h + 200, w + 200, angle=0, density=0.05, stripe_width=1.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def vertical_stripe_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium vertical stripes (density=0.1, width=3)."""
    _, h, w = clean.shape
    stripe = generate_stripes(h + 200, w + 200, angle=0, density=0.1, stripe_width=3.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def vertical_stripe_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong vertical stripes (density=0.2, width=5)."""
    _, h, w = clean.shape
    stripe = generate_stripes(h + 200, w + 200, angle=0, density=0.2, stripe_width=5.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def horizontal_stripe_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak horizontal stripes (density=0.05, width=1)."""
    _, h, w = clean.shape
    stripe = generate_stripes(h + 200, w + 200, angle=90, density=0.05, stripe_width=1.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def horizontal_stripe_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium horizontal stripes (density=0.1, width=3)."""
    _, h, w = clean.shape
    stripe = generate_stripes(h + 200, w + 200, angle=90, density=0.1, stripe_width=3.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def horizontal_stripe_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong horizontal stripes (density=0.2, width=5)."""
    _, h, w = clean.shape
    stripe = generate_stripes(h + 200, w + 200, angle=90, density=0.2, stripe_width=5.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def rotated_stripe_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak rotated stripes (density=0.05, width=1, random angle)."""
    _, h, w = clean.shape
    angle = float(rng.uniform(0, 360))
    stripe = generate_stripes(h + 200, w + 200, angle=angle, density=0.05, stripe_width=1.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def rotated_stripe_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium rotated stripes (density=0.1, width=3, random angle)."""
    _, h, w = clean.shape
    angle = float(rng.uniform(0, 360))
    stripe = generate_stripes(h + 200, w + 200, angle=angle, density=0.1, stripe_width=3.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


def rotated_stripe_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong rotated stripes (density=0.2, width=5, random angle)."""
    _, h, w = clean.shape
    angle = float(rng.uniform(0, 360))
    stripe = generate_stripes(h + 200, w + 200, angle=angle, density=0.2, stripe_width=5.0).to(clean.device)
    stripe = kornia.geometry.center_crop(stripe.unsqueeze(0).unsqueeze(0), (h, w)).squeeze(0).squeeze(0)
    scale = float(rng.uniform(0, 0.9))
    inp = clean * (1 - stripe) + stripe * scale * clean
    return inp, clean


# ============================================================================
# Non-uniform Stripe Functions (variable width + per-band brightness)
# ============================================================================

def nonuniform_vertical_stripe_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak nonuniform vertical stripes (8–20, width 1–3 px, scale 0.80–1.20)."""
    inp = _apply_nonuniform_stripes(clean, rng, axis="w",
                                    num_stripes_range=(8, 21),
                                    width_range=(1, 4),
                                    scale_range=(0.80, 1.20))
    return inp, clean


def nonuniform_vertical_stripe_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium nonuniform vertical stripes (15–40, width 1–5 px, scale 0.55–1.35)."""
    inp = _apply_nonuniform_stripes(clean, rng, axis="w",
                                    num_stripes_range=(15, 41),
                                    width_range=(1, 6),
                                    scale_range=(0.55, 1.35))
    return inp, clean


def nonuniform_vertical_stripe_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong nonuniform vertical stripes (30–70, width 1–8 px, scale 0.30–1.50)."""
    inp = _apply_nonuniform_stripes(clean, rng, axis="w",
                                    num_stripes_range=(30, 71),
                                    width_range=(1, 9),
                                    scale_range=(0.30, 1.50))
    return inp, clean


def nonuniform_horizontal_stripe_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak nonuniform horizontal stripes (8–20, width 1–3 px, scale 0.80–1.20)."""
    inp = _apply_nonuniform_stripes(clean, rng, axis="h",
                                    num_stripes_range=(8, 21),
                                    width_range=(1, 4),
                                    scale_range=(0.80, 1.20))
    return inp, clean


def nonuniform_horizontal_stripe_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium nonuniform horizontal stripes (15–40, width 1–5 px, scale 0.55–1.35)."""
    inp = _apply_nonuniform_stripes(clean, rng, axis="h",
                                    num_stripes_range=(15, 41),
                                    width_range=(1, 6),
                                    scale_range=(0.55, 1.35))
    return inp, clean


def nonuniform_horizontal_stripe_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong nonuniform horizontal stripes (30–70, width 1–8 px, scale 0.30–1.50)."""
    inp = _apply_nonuniform_stripes(clean, rng, axis="h",
                                    num_stripes_range=(30, 71),
                                    width_range=(1, 9),
                                    scale_range=(0.30, 1.50))
    return inp, clean


def _apply_nonuniform_rotated_stripes(
    clean: Tensor,
    rng: RNG,
    num_stripes_range: Tuple[int, int],
    width_range: Tuple[int, int],
    scale_range: Tuple[float, float],
) -> Tensor:
    """Generate non-uniform vertical stripes on a larger canvas, rotate, then center crop."""
    _, h, w = clean.shape
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    pad = diag + 20
    mask_1d = _generate_nonuniform_stripe_mask(pad, rng, num_stripes_range, width_range, scale_range)
    mask_2d = mask_1d.unsqueeze(0).expand(pad, -1)                         # (pad, pad)
    angle = float(rng.uniform(0, 360))
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).to(clean.device)           # (1,1,pad,pad)
    rotated = kornia.geometry.rotate(mask_4d, torch.tensor(angle, device=clean.device).unsqueeze(0))
    cropped = kornia.geometry.center_crop(rotated, (h, w)).squeeze(0)      # (1, H, W)
    return torch.clamp(clean * cropped, 0, 1)


def nonuniform_rotated_stripe_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak nonuniform rotated stripes (8–20, width 1–3 px, scale 0.80–1.20, random angle)."""
    inp = _apply_nonuniform_rotated_stripes(clean, rng,
                                            num_stripes_range=(8, 21),
                                            width_range=(1, 4),
                                            scale_range=(0.80, 1.20))
    return inp, clean


def nonuniform_rotated_stripe_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium nonuniform rotated stripes (15–40, width 1–5 px, scale 0.55–1.35, random angle)."""
    inp = _apply_nonuniform_rotated_stripes(clean, rng,
                                            num_stripes_range=(15, 41),
                                            width_range=(1, 6),
                                            scale_range=(0.55, 1.35))
    return inp, clean


def nonuniform_rotated_stripe_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong nonuniform rotated stripes (30–70, width 1–8 px, scale 0.30–1.50, random angle)."""
    inp = _apply_nonuniform_rotated_stripes(clean, rng,
                                            num_stripes_range=(30, 71),
                                            width_range=(1, 9),
                                            scale_range=(0.30, 1.50))
    return inp, clean


# ============================================================================
# Fixed-parameter non-uniform rotated stripe functions
# ============================================================================

def _generate_fixed_stripe_mask(
    length: int,
    num_stripes: int,
    width: int,
    scale: float,
) -> Tensor:
    """Build a 1-D stripe mask with evenly-spaced stripes of fixed width and scale."""
    mask = torch.ones(length)
    if num_stripes <= 0:
        return mask
    spacing = length / num_stripes
    for i in range(num_stripes):
        center = int(spacing * (i + 0.5))
        lo = max(0, center - width // 2)
        hi = min(length, lo + width)
        mask[lo:hi] = scale
    return mask


def _apply_nonuniform_rotated_stripes_fixed(
    clean: Tensor,
    num_stripes: int,
    width: int,
    scale: float,
    angle: float,
) -> Tensor:
    """Fixed-parameter version: evenly-spaced stripes, fixed angle, no rng."""
    _, h, w = clean.shape
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    pad = diag + 20
    mask_1d = _generate_fixed_stripe_mask(pad, num_stripes, width, scale)
    mask_2d = mask_1d.unsqueeze(0).expand(pad, -1)
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).to(clean.device)
    rotated = kornia.geometry.rotate(
        mask_4d, torch.tensor(angle, device=clean.device).unsqueeze(0)
    )
    cropped = kornia.geometry.center_crop(rotated, (h, w)).squeeze(0)
    return torch.clamp(clean * cropped, 0, 1)


def _apply_nonuniform_rotated_stripes_seeded(
    clean: Tensor,
    seed: int,
    num_stripes_range: Tuple[int, int],
    width_range: Tuple[int, int],
    scale_range: Tuple[float, float],
    angle: float,
) -> Tensor:
    """Fixed-seed version of _apply_nonuniform_rotated_stripes: reproducible
    non-uniform stripes with random widths, positions, and scales (both bright
    and dark), but deterministic across calls with the same seed."""
    fixed_rng = np.random.default_rng(seed)
    _, h, w = clean.shape
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))
    pad = diag + 20
    mask_1d = _generate_nonuniform_stripe_mask(pad, fixed_rng, num_stripes_range, width_range, scale_range)
    mask_2d = mask_1d.unsqueeze(0).expand(pad, -1)
    mask_4d = mask_2d.unsqueeze(0).unsqueeze(0).to(clean.device)
    rotated = kornia.geometry.rotate(
        mask_4d, torch.tensor(angle, device=clean.device).unsqueeze(0)
    )
    cropped = kornia.geometry.center_crop(rotated, (h, w)).squeeze(0)
    return torch.clamp(clean * cropped, 0, 1)


def nonuniform_rotated_stripe_fixed_alt_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Seeded weak nonuniform rotated stripes (8–20, width 1–3 px, scale 0.80–1.20, angle=10 deg)."""
    inp = _apply_nonuniform_rotated_stripes_seeded(
        clean, seed=20250306, num_stripes_range=(8, 21),
        width_range=(1, 4), scale_range=(0.80, 1.20), angle=10.0)
    return inp, clean


def nonuniform_rotated_stripe_fixed_alt_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Seeded medium nonuniform rotated stripes (15–40, width 1–5 px, scale 0.55–1.35, angle=10 deg)."""
    inp = _apply_nonuniform_rotated_stripes_seeded(
        clean, seed=20250306, num_stripes_range=(15, 41),
        width_range=(1, 6), scale_range=(0.55, 1.35), angle=10.0)
    return inp, clean


def nonuniform_rotated_stripe_fixed_alt_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Seeded strong nonuniform rotated stripes (30–70, width 1–8 px, scale 0.30–1.50, angle=10 deg)."""
    inp = _apply_nonuniform_rotated_stripes_seeded(
        clean, seed=20250306, num_stripes_range=(30, 71),
        width_range=(1, 9), scale_range=(0.30, 1.50), angle=10.0)
    return inp, clean


def nonuniform_rotated_stripe_fixed_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Fixed weak nonuniform rotated stripes (12, width 2 px, scale=0.75, angle=45 deg)."""
    inp = _apply_nonuniform_rotated_stripes_fixed(clean, num_stripes=12, width=2, scale=0.75, angle=45.0)
    return inp, clean


def nonuniform_rotated_stripe_fixed_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Fixed medium nonuniform rotated stripes (25, width 3 px, scale=0.55, angle=45 deg)."""
    inp = _apply_nonuniform_rotated_stripes_fixed(clean, num_stripes=25, width=3, scale=0.55, angle=45.0)
    return inp, clean


def nonuniform_rotated_stripe_fixed_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Fixed strong nonuniform rotated stripes (50, width 5 px, scale=0.30, angle=45 deg)."""
    inp = _apply_nonuniform_rotated_stripes_fixed(clean, num_stripes=50, width=5, scale=0.30, angle=45.0)
    return inp, clean


