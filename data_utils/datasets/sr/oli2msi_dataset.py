"""OLI2MSI super-resolution dataset."""
import random
from typing import Tuple, Dict, Any

import torch

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_tif_reader
from utils.vis_utils import save_tensor_image


class Oli2MsiDataset(DatasetBase):
    """OLI2MSI super-resolution."""

    input_readers = {
        "lr": create_tif_reader("lr"),
        "hr": create_tif_reader("hr"),
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
        normalized_inputs = self._load_and_normalize_inputs(item)
        lr = normalized_inputs["lr"]
        hr = normalized_inputs["hr"]

        _, hr_h, hr_w = hr.shape
        lr_upsampled = torch.nn.functional.interpolate(
            lr.unsqueeze(0), size=(hr_h, hr_w), mode="bicubic", align_corners=False
        ).squeeze(0)

        inputs = self.dataset_meta["inputs"]
        vc = inputs["hr"]["visual_channels"]
        nc = inputs["hr"]["num_channels"]
        lr_cfg = {"path": str(item["lr"]), "visual_channels": vc, "num_channels": nc, "stretch_min": lr_upsampled.amin(dim=(1,2)).tolist(), "stretch_max": lr_upsampled.amax(dim=(1,2)).tolist(), **{k: inputs["lr"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": str(item["hr"]), "visual_channels": vc, "num_channels": nc, "stretch_min": hr.amin(dim=(1,2)).tolist(), "stretch_max": hr.amax(dim=(1,2)).tolist(), **{k: inputs["hr"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": vc, "num_channels": nc, "stretch_min": hr.amin(dim=(1,2)).tolist(), "stretch_max": hr.amax(dim=(1,2)).tolist(), **{k: inputs["hr"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"lr": lr_cfg, "gt": gt_cfg, "pred": pred_cfg},
        }
        return lr_upsampled, hr, lr_upsampled, image_meta
