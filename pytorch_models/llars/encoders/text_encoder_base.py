from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


class TextEncoderBase(nn.Module, ABC):

    def __init__(self, target_dim: int, freeze: bool = True):
        super().__init__()
        self.target_dim = target_dim
        self.freeze = freeze
        self.proj: nn.Module = nn.Identity()

    def _init_projection(self):
        if self.native_dim != self.target_dim:
            self.proj = nn.Linear(self.native_dim, self.target_dim)
        else:
            self.proj = nn.Identity()

    @property
    @abstractmethod
    def native_dim(self) -> int:
        ...

    @abstractmethod
    def encode(self, text_batch: List[str], device: torch.device) -> torch.Tensor:
        ...

    def forward(self, text_batch: List[str], device: torch.device) -> torch.Tensor:
        raw = self.encode(text_batch, device)
        return self.proj(raw)
