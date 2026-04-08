from typing import Iterator, List

import torch
import torch.nn as nn

from .base import FinetuneBase
from .lora import _get_parent_and_attr, _match_target, _match_name


class AdapterMLP(nn.Module):

    def __init__(self, dim: int, bottleneck_dim: int, act: str = "relu"):
        super().__init__()
        act_fn = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
        }.get(act, nn.ReLU(inplace=True))

        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = act_fn
        self.up = nn.Linear(bottleneck_dim, dim)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class AdapterConv(nn.Module):

    def __init__(self, channels: int, bottleneck_channels: int, act: str = "relu"):
        super().__init__()
        act_fn = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
        }.get(act, nn.ReLU(inplace=True))

        self.down = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=True)
        self.act = act_fn
        self.up = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=True)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class AdapterWrappedModule(nn.Module):

    def __init__(self, original: nn.Module, adapter: nn.Module):
        super().__init__()
        self.original = original
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        return self.adapter(out)


def _get_dim(module: nn.Module) -> tuple[int | None, str]:
    if isinstance(module, nn.Linear):
        return module.out_features, "linear"
    elif isinstance(module, nn.Conv2d):
        return module.out_channels, "conv"
    return None, ""


class AdapterFinetune(FinetuneBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bottleneck_dim = kwargs.get("bottleneck_dim", 64)
        self.bottleneck_ratio = kwargs.get("bottleneck_ratio", None)
        self.act = kwargs.get("act", "relu")
        self.target_modules = kwargs.get("target_modules", ["Conv2d", "Linear"])

    def apply(self, model: nn.Module) -> nn.Module:
        for p in model.parameters():
            p.requires_grad = False

        replacements = []
        for name, module in model.named_modules():
            matched = _match_target(module, self.target_modules) or _match_name(name, self.target_modules)
            if not matched:
                continue

            dim, layer_type = _get_dim(module)
            if dim is None:
                continue

            bn_dim = int(dim * self.bottleneck_ratio) if self.bottleneck_ratio else self.bottleneck_dim
            bn_dim = max(bn_dim, 1)

            if layer_type == "linear":
                adapter = AdapterMLP(dim, bn_dim, act=self.act)
            else:
                adapter = AdapterConv(dim, bn_dim, act=self.act)

            wrapped = AdapterWrappedModule(module, adapter)
            replacements.append((name, wrapped))

        for name, new_module in replacements:
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, new_module)

        replaced_count = len(replacements)
        print(f"[Adapter] Injected {replaced_count} Adapter modules (bottleneck_dim={self.bottleneck_dim})")
        return model

    def trainable_params(self, model: nn.Module) -> Iterator[nn.Parameter]:
        for name, param in model.named_parameters():
            if ".adapter." in name:
                yield param
