from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class RouterBase(ABC, nn.Module):

    def __init__(self, in_channels: int, num_slots: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_slots = num_slots

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...
