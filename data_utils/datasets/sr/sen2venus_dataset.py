"""Sen2Venus super-resolution: .pt tensors (N_patches, C, H, W); dataset.json rows have lr, hr, patch_idx."""
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_sen2venus_reader
from utils.dataset_utils import get_item_stem
from utils.vis_utils import save_tensor_image

import logging
logger = logging.getLogger(__name__)


class Sen2VenusDataset(DatasetBase):
    """Sen2Venus SR; one class for all sites, distinguished by meta_file."""

    input_readers = {
        "lr": create_sen2venus_reader("lr"),
        "hr": create_sen2venus_reader("hr"),
    }
    vis_savers = {
        "lr": save_tensor_image,
        "gt": save_tensor_image,
        "pred": save_tensor_image,
    }

    def __init__(self, meta_file: str, prompt_file: str = None, mode: str = "train", seed: int = 0):
        super().__init__(meta_file, prompt_file, mode)
        self.rng = random.Random(seed)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]
        patch_idx = item["patch_idx"]

        normalized = self._load_and_normalize_inputs(item)
        lr = normalized["lr"]
        hr = normalized["hr"]

        _, hr_h, hr_w = hr.shape
        lr_upsampled = F.interpolate(
            lr.unsqueeze(0), size=(hr_h, hr_w), mode="bicubic", align_corners=False
        ).squeeze(0)

        inputs = self.dataset_meta["inputs"]
        vc = inputs["hr"]["visual_channels"]
        nc = inputs["hr"]["num_channels"]
        stem = get_item_stem(item, fallback_key="hr") or f"{Path(item['hr']).stem}_p{patch_idx}"
        lr_cfg = {"path": stem, "visual_channels": vc, "num_channels": nc, "patch_idx": patch_idx, "stretch_min": lr_upsampled.amin(dim=(1,2)).tolist(), "stretch_max": lr_upsampled.amax(dim=(1,2)).tolist(), **{k: inputs["lr"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": stem, "visual_channels": vc, "num_channels": nc, "patch_idx": patch_idx, "stretch_min": hr.amin(dim=(1,2)).tolist(), "stretch_max": hr.amax(dim=(1,2)).tolist(), **{k: inputs["hr"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": vc, "num_channels": nc, "stretch_min": hr.amin(dim=(1,2)).tolist(), "stretch_max": hr.amax(dim=(1,2)).tolist(), **{k: inputs["hr"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"lr": lr_cfg, "gt": gt_cfg, "pred": pred_cfg},
        }
        return lr_upsampled, hr, lr_upsampled, image_meta
