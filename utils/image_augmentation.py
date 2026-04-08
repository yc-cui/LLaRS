"""Image augmentation and resize/crop for train/valid/test."""
import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from constants import TRAIN_SIZE, VALID_SIZE, TEST_MAX_SIZE


def _resize_to_min(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Bicubic upsample (C, H, W) to at least (target_h, target_w), keep aspect ratio."""
    h, w = tensor.shape[-2:]
    if h >= target_h and w >= target_w:
        return tensor
    scale = max(target_h / h, target_w / w)
    new_h = max(target_h, int(h * scale))
    new_w = max(target_w, int(w * scale))
    return F.interpolate(
        tensor.unsqueeze(0), size=(new_h, new_w),
        mode='bicubic', align_corners=False,
    ).squeeze(0)


def augment_image(
    inp: torch.Tensor,
    gt: torch.Tensor,
    add: torch.Tensor,
    mode: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Augment or resize inp/gt/add; concat on channel dim so geometry stays aligned.

    mode:
        "train" — upsample if smaller than TRAIN_SIZE, random crop, flips/rot90
        "valid" — upsample if smaller than VALID_SIZE, center crop
        "test"  — original size; center crop only if larger than TEST_MAX_SIZE
    """
    splits = [inp.shape[0], gt.shape[0], add.shape[0]]
    x = torch.cat([inp, gt, add], dim=0)
    h, w = x.shape[-2:]

    if mode == "test":
        th, tw = min(h, TEST_MAX_SIZE[0]), min(w, TEST_MAX_SIZE[1])
        if th != h or tw != w:
            x = TF.center_crop(x, (th, tw))
        return torch.split(x, splits, dim=0)

    if mode == "valid":
        if h < VALID_SIZE[0] or w < VALID_SIZE[1]:
            x = _resize_to_min(x, VALID_SIZE[0], VALID_SIZE[1])
        x = TF.center_crop(x, VALID_SIZE)
        return torch.split(x, splits, dim=0)

    # mode == "train"
    if h < TRAIN_SIZE[0] or w < TRAIN_SIZE[1]:
        x = _resize_to_min(x, TRAIN_SIZE[0], TRAIN_SIZE[1])
        h, w = x.shape[-2:]

    top = random.randint(0, max(0, h - TRAIN_SIZE[0]))
    left = random.randint(0, max(0, w - TRAIN_SIZE[1]))
    x = TF.crop(x, top, left, TRAIN_SIZE[0], TRAIN_SIZE[1])

    if random.random() > 0.5:
        x = TF.hflip(x)
    if random.random() > 0.5:
        x = TF.vflip(x)
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k, dims=[-2, -1])

    return torch.split(x, splits, dim=0)
