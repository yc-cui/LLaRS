"""Normalize / denormalize tensors."""
import torch
from typing import Dict, Any


def normalize_inputs(
    inputs: Dict[str, torch.Tensor],
    inputs_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Batch-normalize all inputs (dataset-agnostic).

    Modes from input_config["norm_type"]:
      - "minmax": (tensor + norm_shift) / norm_value  -> ~[0, 1]
      - "zscore": (tensor - mean) / std               -> ~N(0, 1)

    Args:
        inputs: {input_key: tensor}
        inputs_meta: per-key meta from JSON; each key needs norm_type and
                     norm_shift/norm_value or mean/std.
    """
    normalized = {}
    for input_key, tensor in inputs.items():
        cfg = inputs_meta[input_key]
        norm_type = cfg["norm_type"]
        if norm_type == "minmax":
            shift = cfg["norm_shift"]
            value = cfg["norm_value"]
            normalized[input_key] = (tensor + shift) / value
        else:
            mean = torch.tensor(cfg["mean"], dtype=tensor.dtype).view(-1, 1, 1)
            std = torch.tensor(cfg["std"], dtype=tensor.dtype).view(-1, 1, 1)
            normalized[input_key] = (tensor - mean) / std
    return normalized



def denormalize(
    tensor, cfg: Dict[str, Any],
) -> torch.Tensor:
    """Inverse of normalize_inputs for one tensor.

    Args:
        tensor: (C, H, W) or (N, C, H, W) in normalized space (tensor or ndarray).
        cfg: same dict as inputs_meta[key] for that input.
    """
    import numpy as np

    is_np = isinstance(tensor, np.ndarray)
    norm_type = cfg["norm_type"]

    if norm_type == "minmax":
        shift = cfg["norm_shift"]
        value = cfg["norm_value"]
        return tensor * value - shift
    else:
        mean = cfg["mean"]
        std = cfg["std"]
        if is_np:
            mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
            std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
        else:
            mean = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
            std = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)
        return tensor * std + mean


