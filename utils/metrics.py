"""
Image quality metrics via torchmetrics.

Metrics must not fail silently: exceptions indicate real bugs to fix.
"""
import torch
import torch.nn.functional as F
from typing import Dict

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    SpectralAngleMapper,
    ErrorRelativeGlobalDimensionlessSynthesis,
    SpatialCorrelationCoefficient,
    UniversalImageQualityIndex,
)
from torchmetrics.functional.regression import r2_score

def safe_sam(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    """Numerically stable spectral angle mapper to avoid NaNs."""
    pred_4d = _ensure_4d(pred)
    gt_4d = _ensure_4d(gt).to(pred_4d.dtype)

    # (N, C, H, W) -> (N, C, P)
    N, C, H, W = pred_4d.shape
    p = pred_4d.view(N, C, -1)
    t = gt_4d.view(N, C, -1)

    dot = (p * t).sum(dim=1)  # (N, P)
    p_norm = p.norm(dim=1)    # (N, P)
    t_norm = t.norm(dim=1)    # (N, P)

    denom = (p_norm * t_norm).clamp_min(eps)
    cos = (dot / denom).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    angle = torch.acos(cos)  # radians

    # Ignore locations where either spectrum is effectively zero
    valid = (p_norm > eps) & (t_norm > eps)
    if valid.any():
        angle = angle[valid]

    return angle.mean().item()

def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 3:
        return t.unsqueeze(0)
    return t


def cross_correlation(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Per-channel mean cross-correlation."""
    pred_4d = _ensure_4d(pred)
    gt_4d = _ensure_4d(gt)
    N_spectral = pred_4d.shape[1]
    pred_reshaped = pred_4d.view(N_spectral, -1)
    gt_reshaped = gt_4d.view(N_spectral, -1)
    mean_pred = torch.mean(pred_reshaped, 1).unsqueeze(1)
    mean_gt = torch.mean(gt_reshaped, 1).unsqueeze(1)
    numerator = torch.sum((pred_reshaped - mean_pred) * (gt_reshaped - mean_gt), 1)
    denominator = torch.sqrt(
        torch.sum((pred_reshaped - mean_pred) ** 2, 1)
        * torch.sum((gt_reshaped - mean_gt) ** 2, 1)
    )
    cc = numerator / denominator.clamp(min=1e-10)
    return torch.mean(cc).item()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> Dict[str, float]:
    """
    Compute the full set of image quality metrics.

    Args:
        pred: (C, H, W) or (1, C, H, W)
        gt:   (C, H, W) or (1, C, H, W)
        max_val: data range

    Returns:
        dict of metric_name -> float
    """
    pred_4d = _ensure_4d(pred).clamp(0.0, max_val)
    gt_4d = _ensure_4d(gt).clamp(0.0, max_val).to(pred_4d.dtype)
    dev = pred_4d.device
    results: Dict[str, float] = {}

    results["psnr"] = PeakSignalNoiseRatio(data_range=max_val).to(dev)(pred_4d, gt_4d).item()
    results["ssim"] = StructuralSimilarityIndexMeasure(data_range=max_val).to(dev)(pred_4d, gt_4d).item()
    # results["sam"] = SpectralAngleMapper().to(dev)(pred_4d, gt_4d).item()
    results["sam"] = safe_sam(pred_4d, gt_4d)
    results["ergas"] = ErrorRelativeGlobalDimensionlessSynthesis().to(dev)(pred_4d, gt_4d).item()
    # results["scc"] = SpatialCorrelationCoefficient().to(dev)(pred_4d, gt_4d).item()
    # results["uqi"] = UniversalImageQualityIndex().to(dev)(pred_4d, gt_4d).item()

    # # H, W = pred_4d.shape[-2], pred_4d.shape[-1]
    # # MS_SSIM_MIN_SIZE = 160
    # # pred_ms = pred_4d
    # # gt_ms = gt_4d
    # # if H < MS_SSIM_MIN_SIZE or W < MS_SSIM_MIN_SIZE:
    # #     scale = MS_SSIM_MIN_SIZE / min(H, W)
    # #     new_h = max(MS_SSIM_MIN_SIZE, int(round(H * scale)))
    # #     new_w = max(MS_SSIM_MIN_SIZE, int(round(W * scale)))
    # #     pred_ms = F.interpolate(pred_4d, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # #     gt_ms = F.interpolate(gt_4d, size=(new_h, new_w), mode="bilinear", align_corners=False)
    
    # results["ms_ssim"] = MultiScaleStructuralSimilarityIndexMeasure(data_range=max_val).to(dev)(pred_4d, gt_4d).item()

    # results["r2"] = r2_score(pred_4d.flatten(), gt_4d.flatten()).item()
    # results["cc"] = cross_correlation(pred, gt)
    # results["mae"] = torch.mean(torch.abs(pred_4d - gt_4d)).item()
    # results["rmse"] = torch.sqrt(torch.mean((pred_4d - gt_4d) ** 2)).item()

    return results
