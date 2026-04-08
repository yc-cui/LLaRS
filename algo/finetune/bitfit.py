from typing import Iterator

import torch.nn as nn

from .base import FinetuneBase


class BitFitFinetune(FinetuneBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.include_layernorm = kwargs.get("include_layernorm", False)

    def apply(self, model: nn.Module) -> nn.Module:
        trainable_count = 0

        for p in model.parameters():
            p.requires_grad = False

        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
                trainable_count += 1

        if self.include_layernorm:
            norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)
            for module in model.modules():
                if isinstance(module, norm_types):
                    for p in module.parameters():
                        if not p.requires_grad:
                            p.requires_grad = True
                            trainable_count += 1

        print(f"[BitFit] Unfroze {trainable_count} bias/norm parameters"
              f" (include_layernorm={self.include_layernorm})")
        return model

    def trainable_params(self, model: nn.Module) -> Iterator[nn.Parameter]:
        for param in model.parameters():
            if param.requires_grad:
                yield param
