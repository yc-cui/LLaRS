from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ImageEncoderBase(nn.Module, ABC):

    def __init__(self, target_dim: int, in_channels: int = 16, freeze: bool = False):
        super().__init__()
        self.target_dim = target_dim
        self.in_channels = in_channels
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
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raw_embd, logits = self.encode(x)
        projected = self.proj(raw_embd)
        return projected, logits
