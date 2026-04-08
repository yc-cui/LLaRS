from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import kornia

Tensor = torch.Tensor
RNG = np.random.Generator


# ============================================================================
# Sharpen Functions
# ============================================================================

def sharpen_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak sharpen (factor=1.5)."""
    inp = kornia.enhance.sharpness(clean.unsqueeze(0), 1.5).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def sharpen_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium sharpen (factor=2.5)."""
    inp = kornia.enhance.sharpness(clean.unsqueeze(0), 2.5).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def sharpen_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong sharpen (factor=3.5)."""
    inp = kornia.enhance.sharpness(clean.unsqueeze(0), 3.5).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def over_sharpen_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak oversharpen (factor=4.0)."""
    inp = kornia.enhance.sharpness(clean.unsqueeze(0), 4.0).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def over_sharpen_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium oversharpen (factor=5.0)."""
    inp = kornia.enhance.sharpness(clean.unsqueeze(0), 5.0).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


def over_sharpen_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong oversharpen (factor=6.0)."""
    inp = kornia.enhance.sharpness(clean.unsqueeze(0), 6.0).squeeze(0)
    return torch.clamp(inp, 0, 1), clean


