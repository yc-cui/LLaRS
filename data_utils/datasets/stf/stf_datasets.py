"""STF (Spatio-Temporal Fusion) dataset base and concrete implementations."""
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_tif_reader
from utils.vis_utils import save_tensor_image


class STFDatasetBase(DatasetBase):
    """
    STF base: inp = concat(lr_prev_up, hr_prev, lr_curr_up) 18 ch; gt = hr_gt 6 ch; add = lr_curr_up 6 ch.
    """

    input_readers = {
        "hr_prev": create_tif_reader("hr_prev"),
        "lr_prev": create_tif_reader("lr_prev"),
        "hr_gt": create_tif_reader("hr_gt"),
        "lr_curr": create_tif_reader("lr_curr"),
    }
    vis_savers = {
        "lr_prev": save_tensor_image,
        "hr_prev": save_tensor_image,
        "lr_curr": save_tensor_image,
        "gt": save_tensor_image,
        "pred": save_tensor_image,
    }

    def __init__(self, meta_file: str, prompt_file: str = None, mode: str = "train", seed: int = 0):
        super().__init__(meta_file, prompt_file, mode)
        self.rng = random.Random(seed)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]
        normalized_inputs = self._load_and_normalize_inputs(item)
        hr_prev = normalized_inputs["hr_prev"]
        lr_prev = normalized_inputs["lr_prev"]
        hr_gt = normalized_inputs["hr_gt"]
        lr_curr = normalized_inputs["lr_curr"]

        _, hr_h, hr_w = hr_gt.shape
        lr_prev_up = F.interpolate(lr_prev.unsqueeze(0), size=(hr_h, hr_w), mode='bicubic', align_corners=False).squeeze(0)
        lr_curr_up = F.interpolate(lr_curr.unsqueeze(0), size=(hr_h, hr_w), mode='bicubic', align_corners=False).squeeze(0)
        hr_prev = F.interpolate(hr_prev.unsqueeze(0), size=(hr_h, hr_w), mode='bicubic', align_corners=False).squeeze(0)

        inp = torch.cat([lr_prev_up, hr_prev, lr_curr_up], dim=0)
        gt = hr_gt
        add = lr_curr_up

        inputs = self.dataset_meta["inputs"]
        lr_prev_cfg = {"path": str(item["lr_prev"]), "visual_channels": inputs["lr_prev"]["visual_channels"], "num_channels": inputs["lr_prev"]["num_channels"], "stretch_min": lr_prev_up.amin(dim=(1,2)).tolist(), "stretch_max": lr_prev_up.amax(dim=(1,2)).tolist(), **{k: inputs["lr_prev"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        hr_prev_cfg = {"path": str(item["hr_prev"]), "visual_channels": inputs["hr_prev"]["visual_channels"], "num_channels": inputs["hr_prev"]["num_channels"], "stretch_min": hr_prev.amin(dim=(1,2)).tolist(), "stretch_max": hr_prev.amax(dim=(1,2)).tolist(), **{k: inputs["hr_prev"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        lr_curr_cfg = {"path": str(item["lr_curr"]), "visual_channels": inputs["lr_curr"]["visual_channels"], "num_channels": inputs["lr_curr"]["num_channels"], "stretch_min": lr_curr_up.amin(dim=(1,2)).tolist(), "stretch_max": lr_curr_up.amax(dim=(1,2)).tolist(), **{k: inputs["lr_curr"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": str(item["hr_gt"]), "visual_channels": inputs["hr_gt"]["visual_channels"], "num_channels": inputs["hr_gt"]["num_channels"], "stretch_min": hr_gt.amin(dim=(1,2)).tolist(), "stretch_max": hr_gt.amax(dim=(1,2)).tolist(), **{k: inputs["hr_gt"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": inputs["hr_gt"]["visual_channels"], "num_channels": inputs["hr_gt"]["num_channels"], "stretch_min": hr_gt.amin(dim=(1,2)).tolist(), "stretch_max": hr_gt.amax(dim=(1,2)).tolist(), **{k: inputs["hr_gt"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"lr_prev": lr_prev_cfg, "hr_prev": hr_prev_cfg, "lr_curr": lr_curr_cfg, "gt": gt_cfg, "pred": pred_cfg},
        }
        return inp, gt, add, image_meta

    @classmethod
    def visualize_sample(cls, sample: dict, log_dir: str, model_name: str, epoch: int, mode: str, rank: int = 0) -> None:
        if rank != 0:
            return
        meta = sample["image_meta"]
        extra = meta["extra"]
        nc = extra["gt"]["num_channels"]
        save_dir = Path(log_dir) / meta["dataset_name"] / mode
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(extra["gt"]["path"]).stem

        inp = sample["inp"]
        for key, save_fn in cls.vis_savers.items():
            if key == "gt":
                tensor = sample["gt"][:nc]
            elif key == "pred":
                tensor = sample["pred"][:nc]
            elif key == "lr_prev":
                tensor = inp[:nc]
            elif key == "hr_prev":
                tensor = inp[nc:2 * nc]
            elif key == "lr_curr":
                tensor = inp[2 * nc:3 * nc]
            else:
                tensor = inp[:nc]

            if key == "pred":
                path = save_dir / f"{stem}-{key}-{model_name}-{epoch:02d}.png"
            else:
                path = save_dir / f"{stem}-{key}.png"
                if path.exists():
                    continue
            extra[key]["save_path"] = path
            save_fn(tensor, extra[key])

        from utils.vis_utils import save_allband_npz_and_error_vis
        save_allband_npz_and_error_vis(sample, save_dir, stem, model_name, epoch)


class CiaDataset(STFDatasetBase):
    pass


class AhbDataset(STFDatasetBase):
    pass


class DaxingDataset(STFDatasetBase):
    pass


class LgcDataset(STFDatasetBase):
    pass


class TianjinDataset(STFDatasetBase):
    pass
