from typing import Iterator

import torch.nn as nn

from .base import FinetuneBase


class FullFinetune(FinetuneBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply(self, model: nn.Module) -> nn.Module:
        trainable_count = 0
        for p in model.parameters():
            p.requires_grad = True
            trainable_count += 1

        print(f"[Full] All {trainable_count} parameter groups are trainable")
        return model

    def trainable_params(self, model: nn.Module) -> Iterator[nn.Parameter]:
        return model.parameters()
