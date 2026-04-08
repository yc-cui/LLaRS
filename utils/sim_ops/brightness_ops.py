from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

Tensor = torch.Tensor
RNG = np.random.Generator

# Multiplicative brightness: 1=no change, <1=darker, >1=brighter.
# Task matches function name; random forward/reverse pair so (inp, gt) matches task:
# - increase: 50% (clean, bright), 50% (dark, clean)
# - decrease: 50% (clean, dark), 50% (bright, clean)


def _brightness(clean: Tensor, factor: float) -> Tensor:
    """Scale pixel values by factor; works for any C."""
    return torch.clamp(clean * factor, 0.0, 1.0)


# ============================================================================
# Brightness / Low-light Functions
# ============================================================================

def brightness_decrease_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak decrease: 50% (clean, dark), 50% (bright, clean)."""
    if rng.random() < 0.5:
        return clean, _brightness(clean, 0.7)
    return _brightness(clean, 1.2), clean


def brightness_decrease_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium decrease: 50% (clean, dark), 50% (bright, clean)."""
    if rng.random() < 0.5:
        return clean, _brightness(clean, 0.5)
    return _brightness(clean, 1.4), clean


def brightness_decrease_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong decrease: 50% (clean, dark), 50% (bright, clean)."""
    if rng.random() < 0.5:
        return clean, _brightness(clean, 0.3)
    return _brightness(clean, 1.6), clean


def brightness_increase_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak increase: 50% (clean, bright), 50% (dark, clean)."""
    if rng.random() < 0.5:
        return clean, _brightness(clean, 1.2)
    return _brightness(clean, 0.7), clean


def brightness_increase_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium increase: 50% (clean, bright), 50% (dark, clean)."""
    if rng.random() < 0.5:
        return clean, _brightness(clean, 1.4)
    return _brightness(clean, 0.5), clean


def brightness_increase_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong increase: 50% (clean, bright), 50% (dark, clean)."""
    if rng.random() < 0.5:
        return clean, _brightness(clean, 1.6)
    return _brightness(clean, 0.3), clean


# def low_light_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
#     """Weak low-light (brightness=0.7, noise=0.01)"""
#     inp = _brightness(clean, 0.7)
#     noise = torch.from_numpy(rng.normal(0.0, 0.01, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
#     return torch.clamp(inp + noise, 0, 1), clean


# def low_light_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
#     """Medium low-light (brightness=0.5, noise=0.03)"""
#     inp = _brightness(clean, 0.5)
#     noise = torch.from_numpy(rng.normal(0.0, 0.03, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
#     return torch.clamp(inp + noise, 0, 1), clean


# def low_light_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
#     """Strong low-light (brightness=0.3, noise=0.05)"""
#     inp = _brightness(clean, 0.3)
#     noise = torch.from_numpy(rng.normal(0.0, 0.05, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
#     return torch.clamp(inp + noise, 0, 1), clean


