"""SAR despeckle: each sample is a .mat with intensity (noisy) and shp_intensity (clean)."""
import random
from typing import Tuple, Dict, Any

import torch

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_sar_despeckle_reader
from utils.vis_utils import save_tensor_image_log


class SarDespeckleDataset(DatasetBase):
    """SAR despeckle dataset."""

    input_readers = {
        "intensity": create_sar_despeckle_reader("intensity"),
        "shp_intensity": create_sar_despeckle_reader("shp_intensity"),
    }
    vis_savers = {
        "intensity": save_tensor_image_log,
        "gt": save_tensor_image_log,
        "pred": save_tensor_image_log,
    }

    def __init__(self, meta_file: str, prompt_file: str = None, mode: str = "train", seed: int = 0):
        super().__init__(meta_file, prompt_file, mode)
        self.rng = random.Random(seed)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]
        mat_path = item["mat_path"]

        normalized = self._load_and_normalize_inputs(item)
        noisy = normalized["intensity"]
        clean = normalized["shp_intensity"]

        inputs = self.dataset_meta["inputs"]
        vc = inputs["shp_intensity"]["visual_channels"]
        nc = inputs["shp_intensity"]["num_channels"]
        intensity_cfg = {"path": str(mat_path), "visual_channels": vc, "num_channels": nc, "stretch_min": noisy.amin(dim=(1,2)).tolist(), "stretch_max": noisy.amax(dim=(1,2)).tolist(), **{k: inputs["intensity"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": str(mat_path), "visual_channels": vc, "num_channels": nc, "stretch_min": clean.amin(dim=(1,2)).tolist(), "stretch_max": clean.amax(dim=(1,2)).tolist(), **{k: inputs["shp_intensity"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": vc, "num_channels": nc, "stretch_min": clean.amin(dim=(1,2)).tolist(), "stretch_max": clean.amax(dim=(1,2)).tolist(), **{k: inputs["shp_intensity"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"intensity": intensity_cfg, "gt": gt_cfg, "pred": pred_cfg},
        }
        return noisy, clean, noisy, image_meta
