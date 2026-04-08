from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import RouterBase

PADDING_CHANNEL_ABS_SUM_EPS = 1e-6


def sinkhorn_transport_logspace(
    logits: Tensor,
    channel_mask: Tensor,
    tau: float = 0.1,
    n_iters: int = 8,
    eps: float = 1e-8,
    *,
    mask_logit: float = -1e9,
    clamp: float = 60.0,
) -> Tensor:
    # Log-domain Sinkhorn iterations (stable vs raw exp domain).
    B, S, C = logits.shape

    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    channel_mask = torch.nan_to_num(channel_mask, nan=0.0, posinf=0.0, neginf=0.0)

    denom = channel_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    b = channel_mask / denom
    a = torch.full((B, S), 1.0 / S, device=logits.device, dtype=logits.dtype)

    loga = torch.log(a.clamp_min(eps))
    logb = torch.log(b.clamp_min(eps))

    logK = (logits / float(tau)).clamp(min=-clamp, max=clamp)
    logK = logK - logK.max(dim=2, keepdim=True).values

    log_mask = torch.where(channel_mask > 0.5, torch.zeros_like(channel_mask), torch.full_like(channel_mask, mask_logit))
    log_mask = log_mask.unsqueeze(1)

    logu = torch.zeros((B, S), device=logits.device, dtype=logits.dtype)
    logv = torch.zeros((B, C), device=logits.device, dtype=logits.dtype)

    for _ in range(int(n_iters)):
        logu = loga - torch.logsumexp(logK + log_mask + logv.unsqueeze(1), dim=2)
        logv = logb - torch.logsumexp((logK + log_mask).transpose(1, 2) + logu.unsqueeze(1), dim=2)

    logP = logu.unsqueeze(-1) + logK + log_mask + logv.unsqueeze(1)
    P = torch.exp(logP)
    P = P * channel_mask.unsqueeze(1)
    return P


def _infer_valid_channel_mask(x: Tensor) -> Tensor:
    # Valid channels follow non-pad content in x, not image_meta["num_channels"] (GT-only count).
    l1 = x.abs().sum(dim=(2, 3))
    return l1 > PADDING_CHANNEL_ABS_SUM_EPS


class SinkhornRouterV2(RouterBase):

    def __init__(self, in_channels: int, num_slots: int, **kwargs):
        super().__init__(in_channels, num_slots, **kwargs)
        self.d_e = kwargs.get("d_e", 64)
        self.proj_dim = kwargs.get("proj_dim", 64)
        self.num_iters = kwargs.get("num_iters", 8)
        self.temperature = kwargs.get("temperature", 0.2)
        self.eps = kwargs.get("eps", 1e-8)

        self.emb_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, self.d_e),
        )

        self.slots = nn.Parameter(torch.randn(num_slots, self.d_e) * 0.02)
        self.Wq = nn.Linear(self.d_e, self.proj_dim, bias=False)
        self.Wk = nn.Linear(self.d_e, self.proj_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        S = self.num_slots

        mask = _infer_valid_channel_mask(x)
        if not bool(mask.any(dim=1).all()):
            raise RuntimeError(
                "SinkhornRouterV2: every sample must have at least one channel with "
                f"L1 mass > {PADDING_CHANNEL_ABS_SUM_EPS} over H,W (check input / padding)."
            )

        y1 = x.reshape(B * C, 1, H, W)
        e = self.emb_net(y1)
        E = e.reshape(B, C, self.d_e)

        with torch.amp.autocast("cuda", enabled=False):
            import torch.nn.functional as F

            q = F.linear(self.slots.float(), self.Wq.weight.float())
            k = F.linear(E.float(), self.Wk.weight.float())
            logits = torch.einsum("sd,bcd->bsc", q, k) / math.sqrt(q.shape[-1])

            logits = logits.masked_fill(~mask.unsqueeze(1), 0.0)

            P = sinkhorn_transport_logspace(
                logits=logits,
                channel_mask=mask.to(dtype=logits.dtype),
                tau=float(self.temperature),
                n_iters=int(self.num_iters),
                eps=float(self.eps),
            )

        P = P.to(dtype=x.dtype)
        y_aligned = torch.einsum("bsc,bchw->bshw", P, x)
        return y_aligned


__all__ = ["sinkhorn_transport_logspace", "SinkhornRouterV2", "PADDING_CHANNEL_ABS_SUM_EPS"]
