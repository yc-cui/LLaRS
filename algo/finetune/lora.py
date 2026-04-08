import math
from typing import Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FinetuneBase


class LoRALinear(nn.Module):

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return out + lora_out


class LoRAConv2d(nn.Module):

    def __init__(self, original: nn.Conv2d, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_channels = original.in_channels
        out_channels = original.out_channels
        kernel_size = original.kernel_size

        self.lora_A = nn.Conv2d(
            in_channels, rank, kernel_size=kernel_size,
            stride=original.stride, padding=original.padding,
            dilation=original.dilation, groups=original.groups,
            bias=False,
        )
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return out + lora_out


def _get_parent_and_attr(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _match_target(module: nn.Module, target_modules: List[str]) -> bool:
    for target in target_modules:
        if target == "Linear" and isinstance(module, nn.Linear):
            return True
        if target == "Conv2d" and isinstance(module, nn.Conv2d):
            return True
    return False


def _match_name(name: str, target_modules: List[str]) -> bool:
    for target in target_modules:
        if target in ("Linear", "Conv2d"):
            continue
        if target in name:
            return True
    return False


class LoRAFinetune(FinetuneBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rank = kwargs.get("rank", 8)
        self.alpha = kwargs.get("alpha", 16.0)
        self.dropout = kwargs.get("dropout", 0.0)
        self.target_modules = kwargs.get("target_modules", ["Conv2d", "Linear"])

    def apply(self, model: nn.Module) -> nn.Module:
        for p in model.parameters():
            p.requires_grad = False

        replacements = []
        for name, module in model.named_modules():
            matched = _match_target(module, self.target_modules) or _match_name(name, self.target_modules)
            if not matched:
                continue

            if isinstance(module, nn.Linear):
                new_module = LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
                replacements.append((name, new_module))
            elif isinstance(module, nn.Conv2d):
                if module.groups > 1:
                    continue
                new_module = LoRAConv2d(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
                replacements.append((name, new_module))

        for name, new_module in replacements:
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, new_module)

        replaced_count = len(replacements)
        print(f"[LoRA] Injected {replaced_count} LoRA modules (rank={self.rank}, alpha={self.alpha})")

        return model

    def trainable_params(self, model: nn.Module) -> Iterator[nn.Parameter]:
        for name, param in model.named_parameters():
            if "lora_" in name:
                yield param
