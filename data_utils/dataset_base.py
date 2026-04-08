import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json

from constants import (
    MAX_CHANS,
    DATASET_CONFIGS,
    TEXT_DEG_TYPE_CLASS,
    IMG_DEG_TYPE_CLASS,
    resolve_project_relative_path,
)
from utils.image_augmentation import augment_image
from utils.prompt_loader import load_prompts, select_prompt_for_deg_type
from utils.dataset_utils import load_dataset_splits, pad_channels_to_max
import logging
logger = logging.getLogger(__name__)


class DatasetBase(Dataset, ABC):
    """
    Base class for multi-task remote-sensing image datasets (cloud removal, SR, etc.).
    """

    def __init__(
        self,
        meta_file: str,
        prompt_file: str = None,
        mode: str = "train",  # "train" / "valid" / "test"
    ):
        """
        Args:
            meta_file: Path to dataset meta JSON; must contain ``path`` to the real dataset JSON.
            prompt_file: Prompt JSON path; default merges all prompts under prompts/.
            mode: "train" / "valid" / "test". valid and test share the same no-augment behavior;
                  names differ for downstream eval hooks only.
        """
        self.meta_file = Path(meta_file)
        assert self.meta_file.exists(), f"meta_file not found: {self.meta_file}"

        # Full meta: inputs, visual_channels, dataset_name, ...
        with open(self.meta_file, "r", encoding="utf-8") as f:
            self.dataset_meta: Dict[str, Any] = json.load(f)

        # meta["path"] -> list JSON; relative to PROJECT_ROOT unless absolute
        assert "path" in self.dataset_meta, f"meta missing required key 'path': {self.meta_file}"
        self.dataset_file = resolve_project_relative_path(self.dataset_meta["path"])
        assert self.dataset_file.exists(), f"dataset_file (from meta.path) not found: {self.dataset_file}"

        self.mode = mode
        
        # Splits from dataset json (valid may be empty)
        self.train_dataset, self.valid_dataset, self.test_dataset = load_dataset_splits(
            self.dataset_file
        )
        
        # Pick split; valid/test fall back to each other if one is empty
        assert mode in ("train", "valid", "test"), f"Invalid mode: {mode}"
        if mode == "train":
            self.dataset = self.train_dataset
        elif mode == "valid":
            self.dataset = self.valid_dataset if (self.valid_dataset and len(self.valid_dataset) > 0) else self.test_dataset
        else:
            self.dataset = (
                self.test_dataset
                if (self.test_dataset and len(self.test_dataset) > 0)
                else (self.valid_dataset if (self.valid_dataset and len(self.valid_dataset) > 0) else self.test_dataset)
            )
        
        # Prompts
        self.prompts = load_prompts(prompt_file)
        
        # RNG for prompt sampling
        self.rng = random.Random(0)

        logger.info(f"Meta file: {meta_file}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Dataset size: {len(self.dataset)}")

    def _load_and_normalize_inputs(
        self, item: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Load tensors via ``input_readers`` and normalize using meta ``inputs``.

        Subclasses must define class-level ``input_readers``, e.g.
        ``{"ms": create_mat_reader("ms"), ...}``. Each reader is
        ``reader(item: dict) -> Tensor``.
        """
        if not hasattr(self, 'input_readers'):
            raise AttributeError("Subclass must define input_readers (class or instance attr)")

        loaded_inputs = {}
        for input_key, reader_fn in self.input_readers.items():
            loaded_inputs[input_key] = reader_fn(item)

        from utils.image_normalization import normalize_inputs
        return normalize_inputs(loaded_inputs, self.dataset_meta["inputs"])
        
    @abstractmethod
    def load_image(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Dict[str, Any]]]:
        """
        Load one sample. ``index`` indexes ``self.dataset`` (list of item dicts).

        Returns:
            inp, gt, add: (C, H, W) tensors; add is residual branch input if used
            image_meta: path, input_key, extra, ...
        """
        pass

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
        """Default viz: iterate ``cls.vis_savers`` and save per key.

        Simple single-input datasets can use as-is; multi-stream inputs may override.
        """
        if rank != 0:
            return
        from utils.vis_utils import save_allband_npz_and_error_vis

        meta = sample["image_meta"]
        extra = meta["extra"]
        nc = extra["gt"]["num_channels"]
        save_dir = Path(log_dir) / meta["dataset_name"] / mode
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(extra["gt"]["path"]).stem

        for key, save_fn in cls.vis_savers.items():
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
                if path.exists():
                    continue
            extra[key]["save_path"] = path
            save_fn(tensor, extra[key])

        save_allband_npz_and_error_vis(sample, save_dir, stem, model_name, epoch)

    def __getitem__(self, index: int) -> dict:
        inp, gt, add, img_meta = self.load_image(index)

        inp, gt, add = augment_image(inp, gt, add, mode=self.mode)

        inp = pad_channels_to_max(inp, MAX_CHANS)
        gt = pad_channels_to_max(gt, MAX_CHANS)
        add = pad_channels_to_max(add, MAX_CHANS)

        dm = self.dataset_meta
        cfg = DATASET_CONFIGS.get(dm["dataset_name"], {})
        if not cfg:
            raise KeyError(f"DATASET_CONFIGS missing entry for dataset_name={dm['dataset_name']}")

        text_deg_type_str = cfg["text_deg_type"]
        img_deg_type_str = cfg["img_deg_type"]

        prompt = select_prompt_for_deg_type(
            self.prompts, text_deg_type_str, self.mode != "train", self.rng
        )

        gt_cfg = img_meta["extra"]["gt"]
        num_channels = gt_cfg["num_channels"]

        img_meta.update(
            {
                "dataset_name": dm["dataset_name"],
                "source_dataset": dm["dataset_name"],
                "num_channels": num_channels,
                # Coarse text / fine image degradation ids (ints)
                "text_deg_type": TEXT_DEG_TYPE_CLASS[text_deg_type_str],
                "img_deg_type": IMG_DEG_TYPE_CLASS[img_deg_type_str],
                # String names for logging
                "text_deg_type_name": text_deg_type_str,
                "img_deg_type_name": img_deg_type_str,
            }
        )

        return {
            "inp": inp,
            "gt": gt,
            "add": add,
            "prompt": prompt,
            "image_meta": img_meta,
        }

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.dataset)
