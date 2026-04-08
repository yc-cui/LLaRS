"""SAR despeckling filters denoising dataset."""
import random
from typing import Tuple, Dict, Any

import torch

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_tif_reader
from utils.vis_utils import save_tensor_image


class SarFilterDataset(DatasetBase):
    """SAR despeckling filters denoising."""

    input_readers = {
        "noisy": create_tif_reader("noisy"),
        "gt": create_tif_reader("gt"),
    }
    vis_savers = {
        "noisy": save_tensor_image,
        "gt": save_tensor_image,
        "pred": save_tensor_image,
    }

    def __init__(self, meta_file: str, prompt_file: str = None, mode: str = "train", seed: int = 0):
        super().__init__(meta_file, prompt_file, mode)
        self.rng = random.Random(seed)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]
        normalized_inputs = self._load_and_normalize_inputs(item)
        noisy = normalized_inputs["noisy"]
        gt = normalized_inputs["gt"]

        inputs = self.dataset_meta["inputs"]
        vc = inputs["gt"]["visual_channels"]
        nc = inputs["gt"]["num_channels"]
        noisy_cfg = {"path": str(item["noisy"]), "visual_channels": vc, "num_channels": nc, "stretch_min": noisy.amin(dim=(1,2)).tolist(), "stretch_max": noisy.amax(dim=(1,2)).tolist(), **{k: inputs["noisy"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": str(item["gt"]), "visual_channels": vc, "num_channels": nc, "stretch_min": gt.amin(dim=(1,2)).tolist(), "stretch_max": gt.amax(dim=(1,2)).tolist(), **{k: inputs["gt"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": vc, "num_channels": nc, "stretch_min": gt.amin(dim=(1,2)).tolist(), "stretch_max": gt.amax(dim=(1,2)).tolist(), **{k: inputs["gt"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"noisy": noisy_cfg, "gt": gt_cfg, "pred": pred_cfg},
        }
        return noisy, gt, noisy, image_meta
