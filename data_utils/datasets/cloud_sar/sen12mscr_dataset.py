"""SEN12MSCR cloud removal with SAR; split by season into separate datasets."""
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_tif_reader
from utils.vis_utils import save_tensor_image


class Sen12mscrDatasetBase(DatasetBase):
    """
    SEN12MSCR base with SAR: inp = concat(s2_cloudy, s1) 15 ch; gt = cloud-free s2, 13 ch; add = s2_cloudy 13 ch.
    """

    input_readers = {
        "s2_cloudy": create_tif_reader("s2_cloudy"),
        "s2": create_tif_reader("s2"),
        "s1": create_tif_reader("s1"),
    }
    vis_savers = {
        "s2_cloudy": save_tensor_image,
        "s1": save_tensor_image,
        "gt": save_tensor_image,
        "pred": save_tensor_image,
    }

    def __init__(self, meta_file: str, prompt_file: str = None, mode: str = "train", seed: int = 0):
        super().__init__(meta_file, prompt_file, mode)
        self.rng = random.Random(seed)

    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]
        normalized_inputs = self._load_and_normalize_inputs(item)
        s2_cloudy = normalized_inputs["s2_cloudy"]
        s2 = normalized_inputs["s2"]
        s1 = normalized_inputs["s1"]

        inp = torch.cat([s2_cloudy, s1], dim=0)
        gt = s2
        add = s2_cloudy

        inputs = self.dataset_meta["inputs"]
        s2_cloudy_cfg = {"path": str(item["s2_cloudy"]), "visual_channels": inputs["s2_cloudy"]["visual_channels"], "num_channels": inputs["s2_cloudy"]["num_channels"], "stretch_min": s2_cloudy.amin(dim=(1,2)).tolist(), "stretch_max": s2_cloudy.amax(dim=(1,2)).tolist(), **{k: inputs["s2_cloudy"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        s1_cfg = {"path": str(item["s1"]), "visual_channels": inputs["s1"]["visual_channels"], "num_channels": inputs["s1"]["num_channels"], "stretch_min": s1.amin(dim=(1,2)).tolist(), "stretch_max": s1.amax(dim=(1,2)).tolist(), **{k: inputs["s1"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": str(item["s2"]), "visual_channels": inputs["s2"]["visual_channels"], "num_channels": inputs["s2"]["num_channels"], "stretch_min": s2.amin(dim=(1,2)).tolist(), "stretch_max": s2.amax(dim=(1,2)).tolist(), **{k: inputs["s2"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": inputs["s2"]["visual_channels"], "num_channels": inputs["s2"]["num_channels"], "stretch_min": s2.amin(dim=(1,2)).tolist(), "stretch_max": s2.amax(dim=(1,2)).tolist(), **{k: inputs["s2"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        image_meta = {
            "extra": {"s2_cloudy": s2_cloudy_cfg, "s1": s1_cfg, "gt": gt_cfg, "pred": pred_cfg},
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
            elif key == "s2_cloudy":
                tensor = inp[:nc]
            elif key == "s1":
                tensor = inp[nc:nc + 2]
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


class Sen12mscrSpringDataset(Sen12mscrDatasetBase):
    pass


class Sen12mscrSummerDataset(Sen12mscrDatasetBase):
    pass


class Sen12mscrFallDataset(Sen12mscrDatasetBase):
    pass


class Sen12mscrWinterDataset(Sen12mscrDatasetBase):
    pass
