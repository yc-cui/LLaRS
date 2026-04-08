from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import kornia

Tensor = torch.Tensor
RNG = np.random.Generator


# ============================================================================
# Equalize Functions
# ============================================================================

def histogram_equalization(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Global histogram equalization: clean input, equalized output as gt."""
    clean_clamped = torch.clamp(clean, 0, 1)
    gt = kornia.enhance.equalize(clean_clamped.unsqueeze(0)).squeeze(0)
    return clean, gt


def clahe_equalization(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """CLAHE: clean input, contrast-enhanced output as gt."""
    clean_clamped = torch.clamp(clean, 0, 1)
    gt = kornia.enhance.equalize_clahe(clean_clamped.unsqueeze(0)).squeeze(0)
    return clean, gt


