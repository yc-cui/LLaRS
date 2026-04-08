from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

Tensor = torch.Tensor
RNG = np.random.Generator


def _percentile_stretch(img: Tensor, low_pct: float, high_pct: float) -> Tensor:
    """Per-channel percentile linear stretch to [0, 1]."""
    C = img.shape[0]
    out = torch.empty_like(img)
    for c in range(C):
        ch = img[c]
        flat = ch.flatten()
        lo = torch.quantile(flat, low_pct / 100.0)
        hi = torch.quantile(flat, high_pct / 100.0)
        if hi - lo < 1e-6:
            out[c] = ch
        else:
            out[c] = (ch - lo) / (hi - lo)
    return torch.clamp(out, 0, 1)


def linear_stretch_2_98(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """2%-98% percentile linear stretch (enhancement op).

    inp = original image, gt = linearly stretched image.
    """
    gt = _percentile_stretch(clean, low_pct=2, high_pct=98)
    return clean, gt
