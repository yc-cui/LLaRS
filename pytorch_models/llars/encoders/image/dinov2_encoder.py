from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..image_encoder_base import ImageEncoderBase


class DINOv2ImageEncoder(ImageEncoderBase):

    def __init__(
        self,
        target_dim: int,
        in_channels: int = 16,
        model_name_or_path: str = "facebook/dinov2-base",
        freeze: bool = True,
    ):
        super().__init__(target_dim=target_dim, in_channels=in_channels, freeze=freeze)

        self.model_name_or_path = model_name_or_path

        from transformers import Dinov2Model
        self.dino = Dinov2Model.from_pretrained(model_name_or_path)

        self._native_dim = self.dino.config.hidden_size

        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.channel_adapter = nn.Identity()

        if self.freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        self._init_projection()

    @property
    def native_dim(self) -> int:
        return self._native_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_adapted = self.channel_adapter(x)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.dino(pixel_values=x_adapted)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding, None
