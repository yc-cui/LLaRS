"""Dataset JSON loading, sim merge, channel padding."""
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from constants import DATASET_CONFIGS, resolve_project_relative_path


def get_item_stem(item: Dict[str, Any], fallback_key: str = None) -> str:
    """Unique stem for a dataset item.

    Prefer item["stem"] (index-style sets); else Path(item[fallback_key]).stem.
    """
    if "stem" in item:
        return item["stem"]
    if fallback_key and fallback_key in item:
        return Path(str(item[fallback_key])).stem
    return f"unknown_{id(item)}"


def load_dataset_splits(
    dataset_file: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]] | None, List[Dict[str, Any]]]:
    """
    Load dataset JSON: train, optional valid, test splits.

    If no "valid" key, valid_split is None. Uses full lists (no subset_ratio).
    """
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset_dict = json.load(f)

    assert isinstance(dataset_dict, dict), "dataset file must be a dict"
    assert "train" in dataset_dict, "dataset file must contain 'train'"
    assert "test" in dataset_dict, "dataset file must contain 'test'"
    train_list = dataset_dict["train"]
    test_list = dataset_dict["test"]
    valid_list = dataset_dict.get("valid", None)

    assert isinstance(train_list, list), "'train' must be a list"
    assert isinstance(test_list, list), "'test' must be a list"
    if valid_list is not None:
        assert isinstance(valid_list, list), "'valid' must be a list"

    return train_list, valid_list, test_list

def build_sim_dataset(
    sim_config_file: Path | str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Merge multiple real sources into one dict:
      {"train": [...], "test": [...], "valid": [...]}

    Each source entry uses "name" (preferred) or "meta_file":
      - "name": key in DATASET_CONFIGS; meta path resolved (cwd = project root)
      - "meta_file": explicit meta JSON path
    Also: input_key (required), max_samples, valid_max_samples, etc.

    Each merged row = raw_item from JSON plus normalized fields;
    SimDatasetBase passes raw_item to the reader.
    """
    sim_config_file = Path(sim_config_file)
    with open(sim_config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sources = cfg["sources"]
    merged = {"train": [], "test": [], "valid": []}

    for s in sources:
        if "name" in s:
            meta_file = resolve_project_relative_path(DATASET_CONFIGS[s["name"]]["meta_file"])
        else:
            meta_file = resolve_project_relative_path(s["meta_file"])
        with open(meta_file, "r", encoding="utf-8") as f:
            meta_full = json.load(f)
        dataset_file = resolve_project_relative_path(meta_full["path"])

        input_key = s["input_key"]

        train_list, valid_list, test_list = load_dataset_splits(dataset_file)

        ds_name = meta_full["dataset_name"]
        inp_meta = meta_full["inputs"][input_key]

        def _convert(split_list: List[Dict[str, Any]],
                      _ds_name=ds_name, _input_key=input_key,
                      _inp_meta=inp_meta) -> List[Dict[str, Any]]:
            base = {
                "dataset_name": _ds_name,
                "input_key": _input_key,
                "num_channels": _inp_meta["num_channels"],
                "visual_channels": _inp_meta["visual_channels"],
                "mean": _inp_meta["mean"], "std": _inp_meta["std"],
                "min": _inp_meta["min"], "max": _inp_meta["max"],
                "norm_type": _inp_meta["norm_type"],
                "norm_shift": _inp_meta["norm_shift"],
                "norm_value": _inp_meta["norm_value"],
            }
            out = []
            for item in split_list:
                out.append({"raw_item": dict(item), **base})
            return out

        max_samples = s.get("max_samples", None)
        valid_max_samples = s.get("valid_max_samples", None)

        train_subset = train_list[:max_samples] if max_samples else train_list
        merged["train"].extend(_convert(train_subset))
        if valid_list is not None:
            valid_subset = valid_list[:valid_max_samples] if valid_max_samples else valid_list
            merged["valid"].extend(_convert(valid_subset))
        merged["test"].extend(_convert(test_list))

    # Fallback: copy test to valid or valid to test if one side empty
    if not merged["valid"] and merged["test"]:
        merged["valid"] = list(merged["test"])
    if not merged["test"] and merged["valid"]:
        merged["test"] = list(merged["valid"])

    return merged



def pad_channels_to_max(tensor: torch.Tensor, max_chans: int = None) -> torch.Tensor:
    """
    Pad channel dim to max_chans (vectorized, no Python loop).

    Args:
        tensor: (C, H, W)
        max_chans: target C, default constants.MAX_CHANS

    Returns:
        (max_chans, H, W)
    """
    if max_chans is None:
        from constants import MAX_CHANS
        max_chans = MAX_CHANS
    
    C = tensor.shape[0]
    if C >= max_chans:
        return tensor[:max_chans]
    # F.pad order: (left, right, top, bottom, front, back); pad channel dim (dim 0)
    pad_size = max_chans - C
    return F.pad(tensor, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
