"""
Dynamic Weight Adjustment (DWA): per-task loss EMA and softmax weights.

Tracks an EMA of loss per degradation/task; if current loss exceeds EMA, that task
gets higher weight so harder tasks receive more optimization signal.

Unlike PCGrad/MGDA, this only rescales scalar losses in forward; no extra backward.

Enable on model wrappers via dwa_cfg.
"""

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


class DynamicWeightAdjuster:
    """
    Per-task weights from loss vs EMA.

    EMA: ema_k(t) = decay * ema_k(t-1) + (1 - decay) * loss_k(t)
    Rate: r_k = loss_k(t) / ema_k(t); r_k > 1 means loss above recent average.

    Weights: w_k = N * softmax(r / T)_k; N = number of tasks in batch, T = temperature.
    """

    def __init__(self, ema_decay: float = 0.95, temperature: float = 2.0):
        self.ema_decay = ema_decay
        self.temperature = temperature
        self._loss_ema: Dict[int, float] = {}

    @torch.no_grad()
    def compute_weights(
        self, per_task_losses: Dict[int, float], device: torch.device = None,
    ) -> Tuple[Dict[int, Tensor], Dict[str, Dict[int, float]]]:
        """
        Compute per-task weights from scalar losses.

        Args:
            per_task_losses: {task_id: float}
            device: device for weight tensors (None -> CPU)

        Returns:
            weights: {task_id: tensor}, sum equals N
            info: loss / ema / rate per task id
        """
        task_ids = list(per_task_losses.keys())
        n_tasks = len(task_ids)

        if n_tasks == 0:
            return {}, {}

        rates = []
        info: Dict[str, Dict[int, float]] = {"loss": {}, "ema": {}, "rate": {}}
        for tid in task_ids:
            cur_loss = per_task_losses[tid]
            if tid not in self._loss_ema:
                self._loss_ema[tid] = cur_loss
            else:
                self._loss_ema[tid] = (
                    self.ema_decay * self._loss_ema[tid]
                    + (1.0 - self.ema_decay) * cur_loss
                )
            ema_val = self._loss_ema[tid]
            rate = cur_loss / max(ema_val, 1e-12)
            rates.append(rate)

            info["loss"][tid] = cur_loss
            info["ema"][tid] = ema_val
            info["rate"][tid] = rate

        if n_tasks == 1:
            return {task_ids[0]: torch.tensor(1.0, device=device)}, info

        rates_t = torch.tensor(rates, dtype=torch.float32, device=device)
        weights = torch.softmax(rates_t / self.temperature, dim=0) * n_tasks

        return {tid: weights[i] for i, tid in enumerate(task_ids)}, info


def weighted_masked_l1_loss(
    pred: Tensor,
    gt: Tensor,
    num_channelss: Tensor,
    sample_weights: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Masked L1 with per-sample scalar weights.

    Args:
        pred, gt: (B, C, H, W)
        num_channelss: (B,) valid channels per sample
        sample_weights: (B,) DWA weights

    Returns:
        (weighted_loss for backward, unweighted mean for logging)
    """
    B, C, H, W = pred.shape
    device = pred.device

    channel_idx = torch.arange(C, device=device).unsqueeze(0)  # (1, C)
    mask = (channel_idx < num_channelss.unsqueeze(1)).float()   # (B, C)
    mask = mask.unsqueeze(-1).unsqueeze(-1)                     # (B, C, 1, 1)

    per_pixel_l1 = torch.abs(pred - gt) * mask  # (B, C, H, W)
    valid_per_sample = mask.sum(dim=(1, 2, 3)) * H * W  # (B,)
    per_sample_loss = per_pixel_l1.sum(dim=(1, 2, 3)) / valid_per_sample.clamp(min=1.0)  # (B,)

    unweighted_loss = per_sample_loss.mean()

    sw = sample_weights.to(device=device, dtype=pred.dtype)
    weighted_loss = (per_sample_loss * sw).mean()

    return weighted_loss, unweighted_loss


def compute_reconstruction_loss(
    pred: Tensor,
    gt: Optional[Tensor],
    num_channelss: Tensor,
    batch: Dict[str, Any],
    dwa: Optional[DynamicWeightAdjuster] = None,
    extra_losses: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    Shared reconstruction loss for model wrappers.

    - dwa is None or not training: plain masked_l1_loss
    - dwa set and training: DWA weights; loss_dict gets dwa_weight_{task_name}, etc.

    Args:
        pred, gt: (B, C, H, W) or gt None
        num_channelss: (B,) valid channels
        batch: full batch (DWA needs image_meta.text_deg_type)
        dwa: DynamicWeightAdjuster or None
        extra_losses: optional dict; ``load_loss`` (MoE auxiliary) is combined as
        ``load_loss * 0.01 * recon.detach()`` where ``recon`` is the reconstruction
        term in ``total_loss`` (DWA-weighted L1 if DWA training, else plain L1).

    Returns:
        loss_dict with l1_loss, total_loss; DWA adds l1_loss_weighted and dwa_* keys
    """
    from utils.losses import masked_l1_loss
    from constants import DEG_TYPE_ID_TO_NAME

    extra_losses = dict(extra_losses or {})
    load_loss_raw = extra_losses.pop("load_loss", None)

    def _other_extra_sum() -> Tensor:
        if not extra_losses:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return sum(extra_losses.values())

    if gt is None:
        t = _other_extra_sum()
        loss_dict: Dict[str, Tensor] = {}
        if load_loss_raw is not None:
            ll = load_loss_raw * 0.01
            loss_dict["load_loss"] = ll
            t = t + ll
        loss_dict["total_loss"] = t
        loss_dict.update(extra_losses)
        return loss_dict

    if dwa is not None and pred.requires_grad:
        deg_types = batch["image_meta"]["text_deg_type"]  # (B,) int tensor
        B = pred.shape[0]

        # Per-task mean loss over samples in batch
        task_groups: Dict[int, list] = defaultdict(list)
        with torch.no_grad():
            C, H, W = pred.shape[1], pred.shape[2], pred.shape[3]
            ch_idx = torch.arange(C, device=pred.device).unsqueeze(0)
            ch_mask = (ch_idx < num_channelss.unsqueeze(1)).float().unsqueeze(-1).unsqueeze(-1)
            per_pixel = torch.abs(pred - gt) * ch_mask
            valid = ch_mask.sum(dim=(1, 2, 3)) * H * W
            per_sample = per_pixel.sum(dim=(1, 2, 3)) / valid.clamp(min=1.0)

            for i in range(B):
                tid = int(deg_types[i])
                task_groups[tid].append(per_sample[i].item())

        per_task_losses = {
            tid: sum(vals) / len(vals) for tid, vals in task_groups.items()
        }
        weights, dwa_info = dwa.compute_weights(per_task_losses, device=pred.device)

        sample_weights = torch.ones(B, device=pred.device)
        for i in range(B):
            tid = int(deg_types[i])
            sample_weights[i] = weights[tid]

        weighted_l1, unweighted_l1 = weighted_masked_l1_loss(
            pred, gt, num_channelss, sample_weights,
        )

        load_term = (
            load_loss_raw * 0.01 * weighted_l1.detach()
            if load_loss_raw is not None
            else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        )

        loss_dict = {
            "l1_loss": unweighted_l1,
            "l1_loss_weighted": weighted_l1,
            "total_loss": weighted_l1 + load_term + _other_extra_sum(),
        }
        for tid, w in weights.items():
            name = DEG_TYPE_ID_TO_NAME.get(tid, str(tid))
            loss_dict[f"dwa_weight_{name}"] = w
            loss_dict[f"dwa_rate_{name}"] = torch.tensor(dwa_info["rate"][tid])
            loss_dict[f"dwa_ema_{name}"] = torch.tensor(dwa_info["ema"][tid])
            loss_dict[f"dwa_loss_{name}"] = torch.tensor(dwa_info["loss"][tid])
    else:
        l1 = masked_l1_loss(pred, gt, num_channelss)
        load_term = (
            load_loss_raw * 0.01 * l1.detach()
            if load_loss_raw is not None
            else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        )
        loss_dict = {
            "l1_loss": l1,
            "total_loss": l1 + load_term + _other_extra_sum(),
        }

    if load_loss_raw is not None:
        loss_dict["load_loss"] = load_term
    loss_dict.update(extra_losses)
    return loss_dict
