"""Shared visualization utilities."""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def _select_channels(
    tensor: torch.Tensor,
    channels: List[int],
) -> np.ndarray:
    """Select visual channels from (C, H, W) tensor. Returns float (H, W, 3) or (H, W)."""
    t = tensor.detach().cpu().float()
    if len(channels) == 1:
        return t[channels[0]].numpy()
    return t[channels[:3]].permute(1, 2, 0).numpy()


def _save_array_as_image(
    arr: np.ndarray,
    save_path: Path,
) -> None:
    """Save float array as PNG.
    """
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr)
    img.save(save_path)


def _percentile_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile clip + linear stretch to [0, 1] for (H, W, C) or (H, W)."""
    p_lo = np.percentile(arr, lo)
    p_hi = np.percentile(arr, hi)
    if p_hi - p_lo < 1e-8:
        return np.clip(arr, 0, 1)
    return np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1)


def save_tensor_image(tensor: torch.Tensor, extra: dict, percentile_stretch: bool = True) -> None:
    """Save RGB visualization: per-band min-max, optional percentile stretch, PNG.

    stretch_min / stretch_max are per-channel lists aligned with C.
    If percentile_stretch=False, skip percentile stretch (e.g. already normalized).
    """
    channels = extra["visual_channels"]
    t = tensor.detach().cpu().float().clone()
    smin = torch.tensor(extra["stretch_min"], dtype=t.dtype).view(-1, 1, 1)
    smax = torch.tensor(extra["stretch_max"], dtype=t.dtype).view(-1, 1, 1)
    denom = (smax - smin).clamp_min(1e-6)
    t = ((t - smin) / denom).clamp(0.0, 1.0)

    arr = _select_channels(t, channels)
    if percentile_stretch:
        arr = _percentile_stretch(arr)
    _save_array_as_image(arr, extra["save_path"])


def save_tensor_image_log(tensor: torch.Tensor, extra: dict, **kwargs) -> None:
    """SAR log-domain visualization: log1p, per-image min-max, PNG.

    Reads save_path and visual_channels from extra.
    """
    channels = extra["visual_channels"]
    if hasattr(channels, "tolist"):
        channels = channels.tolist()

    arr = _select_channels(tensor, channels)
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    _save_array_as_image(arr, extra["save_path"])

def _tensor_or_array_to_numpy_chw(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or ndarray to float32 (C, H, W) numpy."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t.numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected (C, H, W) or (H, W), got shape {arr.shape}")
    return arr


def save_allband_npz(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
    extra_gt: dict,
    save_path: Union[str, Path],
    denorm: bool = True,
) -> None:
    """
    Save full-band pred/gt as compressed npz (optional denorm only; no error maps).

    Args:
        pred: (C, H, W) tensor or array, usually normalized space.
        gt:   (C, H, W) ground truth.
        extra_gt: image_meta["extra"]["gt"] with:
            - "num_channels": valid bands
            - "norm_type" + params (minmax: norm_shift/norm_value; zscore: mean/std)
        save_path: output .npz path.
        denorm: if True, denormalize before save.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred_np = _tensor_or_array_to_numpy_chw(pred)
    gt_np = _tensor_or_array_to_numpy_chw(gt)

    num_channels = int(extra_gt.get("num_channels", pred_np.shape[0]))
    num_channels = max(1, min(num_channels, pred_np.shape[0], gt_np.shape[0]))

    pred_np = pred_np[:num_channels]
    gt_np = gt_np[:num_channels]

    if denorm:
        from utils.image_normalization import denormalize
        pred_np = denormalize(pred_np, extra_gt)
        gt_np = denormalize(gt_np, extra_gt)

    np.savez_compressed(save_path, pred=pred_np, gt=gt_np)


def save_allband_npz_and_error_vis(
    sample: dict,
    save_dir: Union[str, Path],
    stem: Union[str, Path],
    model_name: str,
    epoch: int,
) -> None:
    """
    Plot MAE heatmap (no npz). Called from DatasetBase.visualize_sample after RGB PNG.
    """
    from utils.error_vis_utils import plot_mae_heatmap

    save_dir = Path(save_dir)
    stem_str = Path(stem).stem if isinstance(stem, (str, Path)) else str(stem)
    meta = sample["image_meta"]
    extra_gt = meta["extra"]["gt"]
    nc = extra_gt["num_channels"]
    pred = sample["pred"][:nc]
    gt = sample["gt"][:nc]

    plot_mae_heatmap(
        pred=pred,
        gt=gt,
        save_path=save_dir / f"{stem_str}-mae-{model_name}-{epoch:02d}.png",
        denorm=False,
    )
