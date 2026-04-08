from abc import ABC, abstractmethod
from typing import Iterator

import torch.nn as nn


class FinetuneBase(ABC):

    def __init__(self, **kwargs):
        self.cfg = kwargs

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        ...

    @abstractmethod
    def trainable_params(self, model: nn.Module) -> Iterator[nn.Parameter]:
        ...

    def print_trainable_info(self, model: nn.Module) -> None:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in self.trainable_params(model))
        frozen = total - trainable
        ratio = 100.0 * trainable / total if total > 0 else 0.0
        print(f"[Finetune] method={self.__class__.__name__}")
        print(f"  Total params:     {total:>12,}")
        print(f"  Trainable params: {trainable:>12,} ({ratio:.2f}%)")
        print(f"  Frozen params:    {frozen:>12,}")
