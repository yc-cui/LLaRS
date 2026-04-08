"""
Error visualization helpers.

Builds figures from pred / gt; optional denormalization via `denorm` and mean/std from config.
"""
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# ============================================================================
# Matplotlib / Seaborn style
# ============================================================================
font_prop = FontProperties(size=14)
font_prop_title = FontProperties(size=14)

matplotlib.rcParams.setdefault("axes.unicode_minus", False)
matplotlib.rcParams.setdefault("figure.dpi", 150)
matplotlib.rcParams.setdefault("savefig.dpi", 300)

# Light seaborn grid style
sns.set_theme(style="whitegrid")


ArrayLike = Union[np.ndarray, "torch.Tensor"]  # type: ignore[name-defined]


def _to_numpy_chw(arr: ArrayLike) -> np.ndarray:
    """Convert input to float32 (C, H, W) numpy array."""
    # Lazy import so numpy-only use does not require torch
    try:
        import torch
    except Exception:  # pragma: no cover - numpy only if torch missing
        torch = None  # type: ignore

    if "torch" in str(type(arr)) and torch is not None:
        t = arr.detach().cpu().float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t.numpy()

    np_arr = np.asarray(arr, dtype=np.float32)
    if np_arr.ndim == 2:
        np_arr = np_arr[None, ...]
    if np_arr.ndim != 3:
        raise ValueError(f"Expected (C, H, W) or (H, W), got shape {np_arr.shape}")
    return np_arr


def _apply_denorm(
    pred: np.ndarray,
    gt: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Denormalize pred/gt using cfg norm_type."""
    from utils.image_normalization import denormalize
    return denormalize(pred, cfg), denormalize(gt, cfg)


def plot_mae_heatmap(
    pred: ArrayLike,
    gt: ArrayLike,
    save_path: Union[str, Path],
    denorm: bool = False,
    extra_gt: Optional[dict] = None,
) -> None:
    """
    Per-pixel MAE heatmap.

    Args:
        pred, gt: (C, H, W) or (H, W), normalized space.
        save_path: output path (.png).
        denorm: if True, denormalize with extra_gt.
        extra_gt: image_meta["extra"]["gt"] dict.
    """
    pred_np = _to_numpy_chw(pred)
    gt_np = _to_numpy_chw(gt)

    if denorm and extra_gt is not None:
        pred_np, gt_np = _apply_denorm(pred_np, gt_np, extra_gt)

    # (C, H, W) -> (H, W)
    mae_map = np.abs(pred_np - gt_np).mean(axis=0)

    # Normalize to [0, 1] to avoid divide-by-zero on constant maps
    vmin = float(mae_map.min())
    vmax = float(mae_map.max())
    if vmax - vmin > 1e-8:
        mae_norm = (mae_map - vmin) / (vmax - vmin)
    else:
        mae_norm = np.zeros_like(mae_map, dtype=np.float32)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(save_path, mae_norm, cmap="inferno")


def plot_r2_scatter(
    pred: ArrayLike,
    gt: ArrayLike,
    save_path: Union[str, Path],
    *,
    extra_gt: dict,
    max_points: int = 50000,
) -> None:
    """
    Pred vs GT scatter with R² in title.

    Args:
        pred, gt: (C, H, W) or (H, W), normalized space.
        save_path: output path (.png).
        extra_gt: image_meta["extra"]["gt"] dict.
        max_points: max random samples if too dense.
    """
    pred_np = _to_numpy_chw(pred)
    gt_np = _to_numpy_chw(gt)

    pred_np, gt_np = _apply_denorm(pred_np, gt_np, extra_gt)

    x = gt_np.flatten()
    y = pred_np.flatten()

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return

    if x.size > max_points:
        idx = np.random.choice(x.size, max_points, replace=False)
        x = x[idx]
        y = y[idx]

    x_mean = float(x.mean())
    ss_tot = float(np.sum((x - x_mean) ** 2))
    ss_res = float(np.sum((y - x) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mv_min = float(np.min(np.asarray(extra_gt["min"])))
    mv_max = float(np.max(np.asarray(extra_gt["max"])))
    if mv_max > mv_min:
        x = (x - mv_min) / (mv_max - mv_min)
        y = (y - mv_min) / (mv_max - mv_min)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    sns.regplot(
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"s": 6, "alpha": 0.25, "edgecolors": "none"},
        line_kws={"color": "#2ca02c", "linewidth": 1.5},
        ci=95,
        n_boot=100,
        color="#1f77b4",
    )

    xy_min, xy_max = 0.0, 1.0
    ax.plot([xy_min, xy_max], [xy_min, xy_max], "r--", linewidth=1.0, label="y = x")

    ax.set_xlim(xy_min, xy_max)
    ax.set_ylim(xy_min, xy_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Ground truth", fontproperties=font_prop)
    ax.set_ylabel("Prediction", fontproperties=font_prop)
    ax.tick_params(axis="both", labelsize=11)

    title_str = f"R² = {r2:.4f}" if np.isfinite(r2) else "R² = NaN"
    ax.set_title(title_str, fontproperties=font_prop_title)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
