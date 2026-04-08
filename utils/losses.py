"""
Channel-masked L1 loss.

Samples are padded to MAX_CHANS; plain nn.L1Loss would penalize zero-padded channels.
masked_l1_loss sums only over the first num_channels valid bands per sample.
"""

import torch
import torch.nn.functional as F


def masked_l1_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_channelss: torch.Tensor,
) -> torch.Tensor:
    """
    L1 only on first num_channelss channels per sample; ignore padded zeros.

    Args:
        pred: (B, C, H, W)
        gt:   (B, C, H, W)
        num_channelss: (B,) int tensor, valid channels per sample

    Returns:
        scalar loss
    """
    B, C, H, W = pred.shape

    # Channel mask (B, C)
    channel_idx = torch.arange(C, device=pred.device).unsqueeze(0)        # (1, C)
    mask = (channel_idx < num_channelss.unsqueeze(1)).float()             # (B, C)
    mask = mask.unsqueeze(-1).unsqueeze(-1)                                # (B, C, 1, 1)

    # L1 on valid channels only
    diff = torch.abs(pred - gt) * mask
    num_valid = mask.sum() * H * W
    loss = diff.sum() / num_valid.clamp(min=1.0)

    return loss




