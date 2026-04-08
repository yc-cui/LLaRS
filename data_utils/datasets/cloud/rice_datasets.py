"""RICE cloud removal (RICE1, RICE2)."""
import random
from typing import Tuple, Dict, Any

import torch

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_png_reader
from utils.vis_utils import save_tensor_image


class RiceDatasetBase(DatasetBase):
    """RICE cloud removal base (RICE1 and RICE2)."""

    input_readers = {
        "cloud": create_png_reader("cloud"),
        "label": create_png_reader("label"),
    }
    vis_savers = {
        "cloud": save_tensor_image,
        "gt": save_tensor_image,
        "pred": save_tensor_image,
    }

    def __init__(self, meta_file: str, prompt_file: str = None, mode: str = "train", seed: int = 0):
        super().__init__(meta_file, prompt_file, mode)
        self.rng = random.Random(seed)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]
        normalized_inputs = self._load_and_normalize_inputs(item)
        cloud = normalized_inputs["cloud"]
        label = normalized_inputs["label"]

        inputs = self.dataset_meta["inputs"]
        vc = inputs["label"]["visual_channels"]
        nc = inputs["label"]["num_channels"]
        cloud_cfg = {"path": str(item["cloud"]), "visual_channels": vc, "num_channels": nc, "stretch_min": cloud.amin(dim=(1,2)).tolist(), "stretch_max": cloud.amax(dim=(1,2)).tolist(), **{k: inputs["cloud"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        label_cfg = {"path": str(item["label"]), "visual_channels": vc, "num_channels": nc, "stretch_min": label.amin(dim=(1,2)).tolist(), "stretch_max": label.amax(dim=(1,2)).tolist(), **{k: inputs["label"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": vc, "num_channels": nc, "stretch_min": label.amin(dim=(1,2)).tolist(), "stretch_max": label.amax(dim=(1,2)).tolist(), **{k: inputs["label"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"cloud": cloud_cfg, "gt": label_cfg, "pred": pred_cfg},
        }
        return cloud, label, cloud, image_meta


class Rice1Dataset(RiceDatasetBase):
    pass


class Rice2Dataset(RiceDatasetBase):
    pass
