from typing import Iterator, List

import torch
import torch.nn as nn

from .base import FinetuneBase
from .lora import _get_parent_and_attr, _match_target, _match_name


class SSFWrapper(nn.Module):

    def __init__(self, original: nn.Module, num_features: int):
        super().__init__()
        self.original = original
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.dim() == 4:
            return out * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        elif out.dim() == 3:
            return out * self.scale.view(1, 1, -1) + self.shift.view(1, 1, -1)
        elif out.dim() == 2:
            return out * self.scale.view(1, -1) + self.shift.view(1, -1)
        else:
            return out * self.scale + self.shift


def _get_output_features(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return module.out_features
    elif isinstance(module, nn.Conv2d):
        return module.out_channels
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        if hasattr(module, "normalized_shape"):
            return module.normalized_shape[-1] if isinstance(module.normalized_shape, (list, tuple)) else module.normalized_shape
        elif hasattr(module, "num_features"):
            return module.num_features
    return None


class SSFFinetune(FinetuneBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_modules = kwargs.get("target_modules", ["Conv2d", "Linear"])

    def apply(self, model: nn.Module) -> nn.Module:
        for p in model.parameters():
            p.requires_grad = False

        replacements = []
        for name, module in model.named_modules():
            matched = _match_target(module, self.target_modules) or _match_name(name, self.target_modules)
            if not matched:
                continue

            num_features = _get_output_features(module)
            if num_features is None:
                continue

            new_module = SSFWrapper(module, num_features)
            replacements.append((name, new_module))

        for name, new_module in replacements:
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, new_module)

        replaced_count = len(replacements)
        print(f"[SSF] Injected {replaced_count} SSF modules")
        return model

    def trainable_params(self, model: nn.Module) -> Iterator[nn.Parameter]:
        for name, param in model.named_parameters():
            if name.endswith(".scale") or name.endswith(".shift"):
                yield param
