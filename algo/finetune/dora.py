import math
from typing import Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import FinetuneBase
from .lora import _get_parent_and_attr, _match_target, _match_name


class DoRALinear(nn.Module):

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        with torch.no_grad():
            weight_norm = original.weight.norm(dim=1, keepdim=False)
        self.magnitude = nn.Parameter(weight_norm.clone())

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_weight = self.scaling * (self.lora_B @ self.lora_A)
        combined_weight = self.original.weight + lora_weight
        direction = combined_weight / combined_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        new_weight = self.magnitude.unsqueeze(1) * direction
        return F.linear(self.lora_dropout(x), new_weight, self.original.bias)


class DoRAConv2d(nn.Module):

    def __init__(self, original: nn.Conv2d, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank

        in_channels = original.in_channels
        out_channels = original.out_channels

        self.lora_A = nn.Conv2d(
            in_channels, rank, kernel_size=original.kernel_size,
            stride=original.stride, padding=original.padding,
            dilation=original.dilation, groups=original.groups,
            bias=False,
        )
        self.lora_B = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        with torch.no_grad():
            weight_norm = original.weight.view(out_channels, -1).norm(dim=1)
        self.magnitude = nn.Parameter(weight_norm.clone())

        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        combined = orig_out + lora_out
        with torch.no_grad():
            out_ch = self.original.out_channels
            orig_norm = self.original.weight.view(out_ch, -1).norm(dim=1).clamp(min=1e-8)
        scale = (self.magnitude / orig_norm).view(1, -1, 1, 1)
        return combined * scale


class DoRAFinetune(FinetuneBase):

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
                new_module = DoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
                replacements.append((name, new_module))
            elif isinstance(module, nn.Conv2d):
                if module.groups > 1:
                    continue
                new_module = DoRAConv2d(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
                replacements.append((name, new_module))

        for name, new_module in replacements:
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, new_module)

        replaced_count = len(replacements)
        print(f"[DoRA] Injected {replaced_count} DoRA modules (rank={self.rank}, alpha={self.alpha})")

        return model

    def trainable_params(self, model: nn.Module) -> Iterator[nn.Parameter]:
        for name, param in model.named_parameters():
            if "lora_" in name or "magnitude" in name:
                yield param
