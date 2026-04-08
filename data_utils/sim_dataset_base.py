import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import (
    MAX_CHANS,
    DATASET_CONFIGS,
    TEXT_DEG_TYPE_CLASS,
    IMG_DEG_TYPE_CLASS,
)
from utils.prompt_loader import load_prompts, select_prompt_for_deg_type
from utils.dataset_utils import build_sim_dataset, pad_channels_to_max, get_item_stem
from utils.image_augmentation import augment_image
from utils.image_normalization import normalize_inputs


class SimDatasetBase(Dataset):
    """
    Simulated degradation dataset; kwargs mirror real datasets where possible.

      meta_file     – source config JSON (like real dataset meta)
      sim_ops       – list of degradation callables
      text_deg_type – string key for prompt selection
      name          – set by build_dataset after construction
    """

    def __init__(
        self,
        meta_file: str,
        sim_ops: List[Callable],
        text_deg_type: str,
        prompt_file: str = None,
        mode: str = "train",
        seed: int = 0,
    ):
        self.name = None  # set by build_dataset post-construction
        self.text_deg_type = text_deg_type
        self.sim_fns = sim_ops

        merged = build_sim_dataset(meta_file)

        self.mode = mode
        assert mode in ("train", "valid", "test"), f"Invalid mode: {mode}"
        if mode == "train":
            self.dataset = merged["train"]
        elif mode == "valid":
            self.dataset = merged["valid"]
        else:
            self.dataset = merged["test"]

        self.prompts = load_prompts(prompt_file)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_source_cls(self, dataset_name: str):
        from registries.dataset_registry import DATASET_REGISTRY
        return DATASET_REGISTRY[dataset_name]

    def load_image(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        item = self.dataset[index]

        raw_item = item["raw_item"]
        source_dataset = item["dataset_name"]
        input_key = item["input_key"]
        visual_channels = item["visual_channels"]

        source_cls = self._get_source_cls(source_dataset)
        reader_fn = source_cls.input_readers[input_key]
        clean = reader_fn(raw_item)

        inputs_meta = {input_key: item}
        normalized = normalize_inputs({input_key: clean}, inputs_meta)
        clean = normalized[input_key]

        op_rng = np.random.default_rng(int(index) + 12345)
        sim_fn = self.rng.choice(self.sim_fns)
        inp, gt = sim_fn(clean, op_rng)

        add = torch.zeros_like(gt)

        if self.name is None:
            raise ValueError("SimDatasetBase.name is None; build_dataset() must set it before use.")
        cfg = DATASET_CONFIGS.get(self.name, {})
        if not cfg:
            raise KeyError(f"DATASET_CONFIGS missing entry for sim dataset name={self.name}")
        text_deg_type_str = cfg["text_deg_type"]
        img_deg_type_str = cfg["img_deg_type"]

        num_channels = int(item["num_channels"])
        stem = get_item_stem(raw_item, fallback_key=input_key)
        base_cfg = {
            "path": stem,
            "visual_channels": visual_channels,
            "num_channels": num_channels,
            "mean": item["mean"], "std": item["std"],
            "min": item["min"], "max": item["max"],
            "norm_type": item["norm_type"],
            "norm_shift": item["norm_shift"],
            "norm_value": item["norm_value"],
        }
        _smin = gt.amin(dim=(1, 2)).tolist()
        _smax = gt.amax(dim=(1, 2)).tolist()
        inp_cfg = {**base_cfg, "stretch_min": _smin, "stretch_max": _smax}
        gt_cfg = {**base_cfg, "stretch_min": _smin, "stretch_max": _smax}
        pred_cfg = {k: v for k, v in gt_cfg.items() if k != "path"}

        img_meta: Dict[str, Any] = {
            "dataset_name": self.name,
            "source_dataset": source_dataset,
            "input_key": input_key,
            "num_channels": num_channels,
            # Text/image degradation type ids + string names
            "text_deg_type": TEXT_DEG_TYPE_CLASS[text_deg_type_str],
            "img_deg_type": IMG_DEG_TYPE_CLASS[img_deg_type_str],
            "text_deg_type_name": text_deg_type_str,
            "img_deg_type_name": img_deg_type_str,
            "extra": {
                "inp": inp_cfg,
                "gt": gt_cfg,
                "pred": pred_cfg,
            },
        }
        return inp, gt, add, img_meta

    def __getitem__(self, index: int) -> dict:
        inp, gt, add, img_meta = self.load_image(index)

        inp, gt, add = augment_image(inp, gt, add, mode=self.mode)

        inp = pad_channels_to_max(inp, MAX_CHANS)
        gt = pad_channels_to_max(gt, MAX_CHANS)
        add = pad_channels_to_max(add, MAX_CHANS)

        prompt = select_prompt_for_deg_type(
            self.prompts, self.text_deg_type, self.mode != "train", self.rng
        )

        return {
            "inp": inp,
            "gt": gt,
            "add": add,
            "prompt": prompt,
            "image_meta": img_meta,
        }

    @classmethod
    def visualize_sample(
        cls,
        sample: dict,
        log_dir: str,
        model_name: str,
        epoch: int,
        mode: str,
        rank: int = 0,
    ) -> None:
        """Save via ``source_dataset`` vis_savers (min-max style viz).

        Brightness sim subclasses override for their pipeline.
        """
        if rank != 0:
            return

        from registries.dataset_registry import DATASET_REGISTRY

        meta = sample["image_meta"]
        extra = meta["extra"]
        nc = extra["gt"]["num_channels"]
        source_dataset = meta["source_dataset"]
        input_key = meta["input_key"]
        source_cls = DATASET_REGISTRY[source_dataset]
        save_fn = source_cls.vis_savers.get(
            input_key, source_cls.vis_savers.get("gt")
        )

        save_dir = Path(log_dir) / meta["dataset_name"] / source_dataset / mode
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = extra["gt"]["path"]
        pct_stretch = meta.get("text_deg_type_name") != "linear"

        for key in ("inp", "gt", "pred"):
            if key not in extra:
                continue
            if key == "gt":
                tensor = sample["gt"][:nc]
            elif key == "pred":
                tensor = sample["pred"][:nc]
            else:
                tensor = sample["inp"][:nc]

            if key == "pred":
                path = save_dir / f"{stem}-{key}-{model_name}-{epoch:02d}.png"
            else:
                path = save_dir / f"{stem}-{key}.png"

            extra[key]["save_path"] = path
            save_fn(tensor, extra[key], percentile_stretch=pct_stretch)

        from utils.vis_utils import save_allband_npz_and_error_vis
        save_allband_npz_and_error_vis(sample, save_dir, stem, model_name, epoch)
