"""Pansharpening dataset base and concrete classes."""
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn.functional as F

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_nbu_ms_reader, create_nbu_pan_reader
from utils.vis_utils import save_tensor_image


class PansharpeningDatasetBase(DatasetBase):
    """
    Pansharpening base (Wald). Raw: ms (C,ms_h,ms_w), pan (1,pan_h,pan_w), scale = pan_h/ms_h.
    Training: gt=ms; lr_ms=downsample(ms); lr_pan=downsample(pan); lms=bicubic(lr_ms to lr_pan size); inp=cat(lms,lr_pan); add=lms.
    """

    vis_savers = {
        "lms": save_tensor_image,
        "pan": save_tensor_image,
        "gt": save_tensor_image,
        "pred": save_tensor_image,
    }

    def __init__(self, meta_file: str, prompt_file: str = None, mode: str = "train", seed: int = 0):
        super().__init__(meta_file, prompt_file, mode)
        self.rng = random.Random(seed)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]
        normalized_inputs = self._load_and_normalize_inputs(item)
        ms = normalized_inputs["ms"]    # (C, ms_h, ms_w)
        pan = normalized_inputs["pan"]  # (1, pan_h, pan_w)

        _, ms_h, ms_w = ms.shape
        _, pan_h, pan_w = pan.shape
        scale = pan_h // ms_h

        lr_ms_h, lr_ms_w = ms_h // scale, ms_w // scale
        lr_ms = F.interpolate(
            ms.unsqueeze(0), size=(lr_ms_h, lr_ms_w),
            mode='bicubic', align_corners=False,
        ).squeeze(0)

        lr_pan = F.interpolate(
            pan.unsqueeze(0), size=(ms_h, ms_w),
            mode='bicubic', align_corners=False,
        ).squeeze(0)

        lms = F.interpolate(
            lr_ms.unsqueeze(0), size=(ms_h, ms_w),
            mode='bicubic', align_corners=False,
        ).squeeze(0)

        gt = ms
        inp = torch.cat([lms, lr_pan], dim=0)
        add = lms

        inputs = self.dataset_meta["inputs"]
        vc = inputs["ms"]["visual_channels"]
        nc = inputs["ms"]["num_channels"]
        lms_cfg = {"path": str(item["ms"]), "visual_channels": vc, "num_channels": nc, "stretch_min": lms.amin(dim=(1,2)).tolist(), "stretch_max": lms.amax(dim=(1,2)).tolist(), **{k: inputs["ms"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pan_cfg = {"path": str(item["pan"]), "visual_channels": inputs["pan"]["visual_channels"], "num_channels": inputs["pan"]["num_channels"], "stretch_min": lr_pan.amin(dim=(1,2)).tolist(), "stretch_max": lr_pan.amax(dim=(1,2)).tolist(), **{k: inputs["pan"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": str(item["ms"]), "visual_channels": vc, "num_channels": nc, "stretch_min": gt.amin(dim=(1,2)).tolist(), "stretch_max": gt.amax(dim=(1,2)).tolist(), **{k: inputs["ms"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": vc, "num_channels": nc, "stretch_min": gt.amin(dim=(1,2)).tolist(), "stretch_max": gt.amax(dim=(1,2)).tolist(), **{k: inputs["ms"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"lms": lms_cfg, "pan": pan_cfg, "gt": gt_cfg, "pred": pred_cfg},
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
            elif key == "lms":
                tensor = inp[:nc]
            elif key == "pan":
                tensor = inp[nc:nc + 1]
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


class Gaofen1Dataset(PansharpeningDatasetBase):
    input_readers = {
        "ms": create_nbu_ms_reader("I_MS"),
        "pan": create_nbu_pan_reader("I_PAN"),
    }


class IkonosDataset(PansharpeningDatasetBase):
    input_readers = {
        "ms": create_nbu_ms_reader("imgMS"),
        "pan": create_nbu_pan_reader("imgPAN"),
    }


class QuickbirdDataset(PansharpeningDatasetBase):
    input_readers = {
        "ms": create_nbu_ms_reader("imgMS"),
        "pan": create_nbu_pan_reader("imgPAN"),
    }


class Worldview2Dataset(PansharpeningDatasetBase):
    input_readers = {
        "ms": create_nbu_ms_reader("imgMS"),
        "pan": create_nbu_pan_reader("imgPAN"),
    }


class Worldview3Dataset(PansharpeningDatasetBase):
    input_readers = {
        "ms": create_nbu_ms_reader("imgMS"),
        "pan": create_nbu_pan_reader("imgPAN"),
    }


class Worldview4Dataset(PansharpeningDatasetBase):
    input_readers = {
        "ms": create_nbu_ms_reader("imgMS"),
        "pan": create_nbu_pan_reader("imgPAN"),
    }
