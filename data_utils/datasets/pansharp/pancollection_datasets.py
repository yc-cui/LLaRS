"""PanCollection pansharpening (GF2, QB, WV3, WV2): HDF5, multiple samples per file via h5_file + idx; Wald protocol, lms/pan/gt same spatial size."""
import random
from pathlib import Path
from typing import Tuple, Dict, Any

import torch

from data_utils.dataset_base import DatasetBase
from utils.image_readers import create_pancollection_reader
from utils.dataset_utils import get_item_stem
from utils.vis_utils import save_tensor_image

import logging
logger = logging.getLogger(__name__)


class PanCollectionDatasetBase(DatasetBase):
    """PanCollection pansharpening base (HDF5)."""

    input_readers = {
        "gt": create_pancollection_reader("gt"),
        "lms": create_pancollection_reader("lms"),
        "pan": create_pancollection_reader("pan"),
    }
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
        h5_file = item["h5_file"]
        idx = item["idx"]

        normalized = self._load_and_normalize_inputs(item)
        gt = normalized["gt"]
        lms = normalized["lms"]
        pan = normalized["pan"]

        inp = torch.cat([lms, pan], dim=0)
        add = lms

        inputs = self.dataset_meta["inputs"]
        stem = get_item_stem(item, fallback_key=None) or f"{Path(h5_file).stem}_idx{idx}"
        lms_cfg = {"path": stem, "visual_channels": inputs["lms"]["visual_channels"], "num_channels": inputs["lms"]["num_channels"], "idx": idx, "stretch_min": lms.amin(dim=(1,2)).tolist(), "stretch_max": lms.amax(dim=(1,2)).tolist(), **{k: inputs["lms"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pan_cfg = {"path": stem, "visual_channels": inputs["pan"]["visual_channels"], "num_channels": inputs["pan"]["num_channels"], "idx": idx, "stretch_min": pan.amin(dim=(1,2)).tolist(), "stretch_max": pan.amax(dim=(1,2)).tolist(), **{k: inputs["pan"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        gt_cfg = {"path": stem, "visual_channels": inputs["gt"]["visual_channels"], "num_channels": inputs["gt"]["num_channels"], "idx": idx, "stretch_min": gt.amin(dim=(1,2)).tolist(), "stretch_max": gt.amax(dim=(1,2)).tolist(), **{k: inputs["gt"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
        pred_cfg = {"visual_channels": inputs["gt"]["visual_channels"], "num_channels": inputs["gt"]["num_channels"], "stretch_min": gt.amin(dim=(1,2)).tolist(), "stretch_max": gt.amax(dim=(1,2)).tolist(), **{k: inputs["gt"][k] for k in ("mean", "std", "min", "max", "norm_type", "norm_shift", "norm_value")}}
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
        stem = extra["gt"]["path"]

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
        save_allband_npz_and_error_vis(sample, save_dir, Path(extra["gt"]["path"]).stem, model_name, epoch)


class PcGf2Dataset(PanCollectionDatasetBase):
    pass


class PcQbDataset(PanCollectionDatasetBase):
    pass


class PcWv3Dataset(PanCollectionDatasetBase):
    pass


class PcWv2Dataset(PanCollectionDatasetBase):
    pass
