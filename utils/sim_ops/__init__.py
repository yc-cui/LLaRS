from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from .noise_ops import (
    gaussian_noise_weak, uniform_noise_weak, poisson_noise_weak, rayleigh_noise_weak,
    gamma_noise_weak, impulse_white_noise_weak, salt_pepper_noise_weak, speckle_noise_weak,
    gaussian_noise_medium, uniform_noise_medium, poisson_noise_medium, rayleigh_noise_medium,
    gamma_noise_medium, impulse_white_noise_medium, salt_pepper_noise_medium, speckle_noise_medium,
    gaussian_noise_strong, uniform_noise_strong, poisson_noise_strong, rayleigh_noise_strong,
    gamma_noise_strong, impulse_white_noise_strong, salt_pepper_noise_strong, speckle_noise_strong,
)
from .blur_ops import (
    gaussian_blur_weak, motion_blur_weak, box_blur_weak, median_blur_weak, defocus_blur_weak,
    gaussian_blur_medium, motion_blur_medium, box_blur_medium, median_blur_medium, defocus_blur_medium,
    gaussian_blur_strong, motion_blur_strong, box_blur_strong, median_blur_strong, defocus_blur_strong,
)
from .resolution_ops import (
    bicubic_downsample_weak, bilinear_downsample_weak, nearest_downsample_weak, area_downsample_weak,
    bicubic_downsample_medium, bilinear_downsample_medium, nearest_downsample_medium, area_downsample_medium,
    bicubic_downsample_strong, bilinear_downsample_strong, nearest_downsample_strong, area_downsample_strong,
)
from .stripe_ops import (
    vertical_stripe_weak, horizontal_stripe_weak, rotated_stripe_weak,
    vertical_stripe_medium, horizontal_stripe_medium, rotated_stripe_medium,
    vertical_stripe_strong, horizontal_stripe_strong, rotated_stripe_strong,
    nonuniform_vertical_stripe_weak, nonuniform_horizontal_stripe_weak, nonuniform_rotated_stripe_weak,
    nonuniform_vertical_stripe_medium, nonuniform_horizontal_stripe_medium, nonuniform_rotated_stripe_medium,
    nonuniform_vertical_stripe_strong, nonuniform_horizontal_stripe_strong, nonuniform_rotated_stripe_strong,
    nonuniform_rotated_stripe_fixed_weak, nonuniform_rotated_stripe_fixed_medium, nonuniform_rotated_stripe_fixed_strong,
    nonuniform_rotated_stripe_fixed_alt_weak, nonuniform_rotated_stripe_fixed_alt_medium, nonuniform_rotated_stripe_fixed_alt_strong,
)
from .brightness_ops import (
    brightness_decrease_weak, brightness_increase_weak,
    brightness_decrease_medium, brightness_increase_medium,
    brightness_decrease_strong, brightness_increase_strong,
)
from .sharpen_ops import (
    sharpen_weak, over_sharpen_weak,
    sharpen_medium, over_sharpen_medium,
    sharpen_strong, over_sharpen_strong,
)
from .equalize_ops import (
    histogram_equalization, clahe_equalization,
)
from .linear_ops import (
    linear_stretch_2_98,
)
from .pansharp_ops import (
    pansharp_degradation_weak, pansharp_degradation_medium, pansharp_degradation_strong,
)
from .test_ops import (
    gaussian_noise_sigma_001, gaussian_noise_sigma_005,
    gaussian_noise_sigma_010, gaussian_noise_sigma_025, gaussian_noise_sigma_050,
    gaussian_blur_k3_s05, gaussian_blur_k5_s10, gaussian_blur_k7_s20, gaussian_blur_k9_s30,
    motion_blur_k5, motion_blur_k7, motion_blur_k9, motion_blur_k11,
    bicubic_downsample_x2, bicubic_downsample_x4, bicubic_downsample_x8,
    salt_pepper_noise_p001, salt_pepper_noise_p005, salt_pepper_noise_p010,
)

SimOp = Callable[[torch.Tensor, np.random.Generator], Tuple[torch.Tensor, torch.Tensor]]

# ============================================================================
# Per-degradation atomic op lists by strength (internal, not exported)
# ============================================================================

_denoise_w = [gaussian_noise_weak, uniform_noise_weak, poisson_noise_weak, rayleigh_noise_weak,
              gamma_noise_weak, impulse_white_noise_weak, salt_pepper_noise_weak, speckle_noise_weak]
_denoise_m = [gaussian_noise_medium, uniform_noise_medium, poisson_noise_medium, rayleigh_noise_medium,
              gamma_noise_medium, impulse_white_noise_medium, salt_pepper_noise_medium, speckle_noise_medium]
_denoise_s = [gaussian_noise_strong, uniform_noise_strong, poisson_noise_strong, rayleigh_noise_strong,
              gamma_noise_strong, impulse_white_noise_strong, salt_pepper_noise_strong, speckle_noise_strong]

_deblur_w = [gaussian_blur_weak, motion_blur_weak, box_blur_weak, median_blur_weak, defocus_blur_weak]
_deblur_m = [gaussian_blur_medium, motion_blur_medium, box_blur_medium, median_blur_medium, defocus_blur_medium]
_deblur_s = [gaussian_blur_strong, motion_blur_strong, box_blur_strong, median_blur_strong, defocus_blur_strong]

_sr_w = [bicubic_downsample_weak, bilinear_downsample_weak, nearest_downsample_weak, area_downsample_weak]
_sr_m = [bicubic_downsample_medium, bilinear_downsample_medium, nearest_downsample_medium, area_downsample_medium]
_sr_s = [bicubic_downsample_strong, bilinear_downsample_strong, nearest_downsample_strong, area_downsample_strong]

_destripe_w = [vertical_stripe_weak, horizontal_stripe_weak, rotated_stripe_weak,
               nonuniform_vertical_stripe_weak, nonuniform_horizontal_stripe_weak, nonuniform_rotated_stripe_weak]
_destripe_m = [vertical_stripe_medium, horizontal_stripe_medium, rotated_stripe_medium,
               nonuniform_vertical_stripe_medium, nonuniform_horizontal_stripe_medium, nonuniform_rotated_stripe_medium]
_destripe_s = [vertical_stripe_strong, horizontal_stripe_strong, rotated_stripe_strong,
               nonuniform_vertical_stripe_strong, nonuniform_horizontal_stripe_strong, nonuniform_rotated_stripe_strong]

_brightness_w = [brightness_decrease_weak, brightness_increase_weak]
_brightness_m = [brightness_decrease_medium, brightness_increase_medium]
_brightness_s = [brightness_decrease_strong, brightness_increase_strong]

_sharpen_w = [sharpen_weak, over_sharpen_weak]
_sharpen_m = [sharpen_medium, over_sharpen_medium]
_sharpen_s = [sharpen_strong, over_sharpen_strong]

_equalize = [histogram_equalization, clahe_equalization]

_pansharp_w = [pansharp_degradation_weak]
_pansharp_m = [pansharp_degradation_medium]
_pansharp_s = [pansharp_degradation_strong]

# ============================================================================
# SIM_OP_GROUPS — public registry; key = sim_type (matches DATASET_CONFIGS)
# ============================================================================

SIM_OP_GROUPS: Dict[str, List[SimOp]] = {
    # ===== training: random over all strengths =====
    "denoise":    _denoise_w + _denoise_m + _denoise_s,
    "deblur":     _deblur_w + _deblur_m + _deblur_s,
    "sr":         _sr_w + _sr_m + _sr_s,
    "destripe":   _destripe_w + _destripe_m + _destripe_s,
    "brightness": _brightness_w + _brightness_m + _brightness_s,
    "sharpen":    _sharpen_w + _sharpen_m + _sharpen_s,
    "equalize":   _equalize,
    "pansharp":   _pansharp_w + _pansharp_m + _pansharp_s,
    # ===== brightness tiers =====
    "brightness_increase_weak":   [brightness_increase_weak],
    "brightness_increase_medium": [brightness_increase_medium],
    "brightness_increase_strong": [brightness_increase_strong],
    "brightness_decrease_weak":   [brightness_decrease_weak],
    "brightness_decrease_medium": [brightness_decrease_medium],
    "brightness_decrease_strong": [brightness_decrease_strong],
    # ===== linear stretch =====
    "linear_stretch": [linear_stretch_2_98],
    # weak/medium/strong only (training)
    "denoise_weak": _denoise_w,
    "denoise_medium": _denoise_m,
    "denoise_strong": _denoise_s,
    "deblur_weak": _deblur_w,
    "deblur_medium": _deblur_m,
    "deblur_strong": _deblur_s,
    "sr_weak": _sr_w,
    "sr_medium": _sr_m,
    "sr_strong": _sr_s,
    "destripe_weak": _destripe_w,
    "destripe_medium": _destripe_m,
    "destripe_strong": _destripe_s,
    # ===== fixed nonuniform rotated stripes (dark only) =====
    "destripe_nr_fixed_weak": [nonuniform_rotated_stripe_fixed_weak],
    "destripe_nr_fixed_medium": [nonuniform_rotated_stripe_fixed_medium],
    "destripe_nr_fixed_strong": [nonuniform_rotated_stripe_fixed_strong],
    # ===== fixed nonuniform rotated stripes (alt bright/dark) =====
    "destripe_nr_fixed_alt_weak": [nonuniform_rotated_stripe_fixed_alt_weak],
    "destripe_nr_fixed_alt_medium": [nonuniform_rotated_stripe_fixed_alt_medium],
    "destripe_nr_fixed_alt_strong": [nonuniform_rotated_stripe_fixed_alt_strong],
    # ===== eval: Gaussian noise =====
    "denoise_gauss_001": [gaussian_noise_sigma_001],
    "denoise_gauss_005": [gaussian_noise_sigma_005],
    "denoise_gauss_010": [gaussian_noise_sigma_010],
    "denoise_gauss_025": [gaussian_noise_sigma_025],
    "denoise_gauss_050": [gaussian_noise_sigma_050],
    # ===== eval: salt-pepper =====
    "denoise_sp_001": [salt_pepper_noise_p001],
    "denoise_sp_005": [salt_pepper_noise_p005],
    "denoise_sp_010": [salt_pepper_noise_p010],
    # ===== eval: Gaussian blur =====
    "deblur_gauss_k3": [gaussian_blur_k3_s05],
    "deblur_gauss_k5": [gaussian_blur_k5_s10],
    "deblur_gauss_k7": [gaussian_blur_k7_s20],
    "deblur_gauss_k9": [gaussian_blur_k9_s30],
    # ===== eval: motion blur =====
    "deblur_motion_k5": [motion_blur_k5],
    "deblur_motion_k7": [motion_blur_k7],
    "deblur_motion_k9": [motion_blur_k9],
    "deblur_motion_k11": [motion_blur_k11],
    # ===== eval: bicubic downsample =====
    "sr_bicubic_x2": [bicubic_downsample_x2],
    "sr_bicubic_x4": [bicubic_downsample_x4],
    "sr_bicubic_x8": [bicubic_downsample_x8],
}

# sim_type -> prompts text_deg_type
SIM_TYPE_TO_TEXT_DEG_TYPE: Dict[str, str] = {
    # values must match TEXT_DEG_TYPE_CLASS / prompt JSON keys
    "denoise": "denoise",
    "denoise_weak": "denoise",
    "denoise_medium": "denoise",
    "denoise_strong": "denoise",

    "deblur": "blur",
    "deblur_weak": "blur",
    "deblur_medium": "blur",
    "deblur_strong": "blur",

    "sr": "sr",
    "sr_weak": "sr",
    "sr_medium": "sr",
    "sr_strong": "sr",

    "destripe": "destripe",
    "destripe_weak": "destripe",
    "destripe_medium": "destripe",
    "destripe_strong": "destripe",
    "brightness": "brightness",
    "brightness_increase_weak": "brightness_increase_weak",
    "brightness_increase_medium": "brightness_increase_medium",
    "brightness_increase_strong": "brightness_increase_strong",
    "brightness_decrease_weak": "brightness_decrease_weak",
    "brightness_decrease_medium": "brightness_decrease_medium",
    "brightness_decrease_strong": "brightness_decrease_strong",
    "linear_stretch": "linear",
    "sharpen": "sharpen",
    "equalize": "equal",
    "pansharp": "pansharp",
    "destripe_nr_fixed_weak": "destripe",
    "destripe_nr_fixed_medium": "destripe",
    "destripe_nr_fixed_strong": "destripe",
    "destripe_nr_fixed_alt_weak": "destripe",
    "destripe_nr_fixed_alt_medium": "destripe",
    "destripe_nr_fixed_alt_strong": "destripe",
    "deblur_motion_k5": "blur", "deblur_motion_k7": "blur",
    "deblur_motion_k9": "blur", "deblur_motion_k11": "blur",
    "denoise_gauss_001": "noise", "denoise_gauss_005": "noise",
    "denoise_gauss_010": "noise", "denoise_gauss_025": "noise",
    "denoise_gauss_050": "noise",
    "denoise_sp_001": "noise", "denoise_sp_005": "noise", "denoise_sp_010": "noise",
    "deblur_gauss_k3": "blur", "deblur_gauss_k5": "blur",
    "deblur_gauss_k7": "blur", "deblur_gauss_k9": "blur",
    "sr_bicubic_x2": "sr", "sr_bicubic_x4": "sr", "sr_bicubic_x8": "sr",
}
