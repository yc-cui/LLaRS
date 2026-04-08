from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import CLIPModel

from ..image_encoder_base import ImageEncoderBase


class ClipImageEncoder(ImageEncoderBase):

    def __init__(
        self,
        target_dim: int,
        in_channels: int = 16,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
    ):
        super().__init__(target_dim=target_dim, in_channels=in_channels, freeze=freeze)

        self.model_name_or_path = model_name_or_path

        clip_model = CLIPModel.from_pretrained(model_name_or_path)

        self.vision_model = clip_model.visual_projection
        self.vision_tower = clip_model.vision_model

        self._native_dim = clip_model.config.projection_dim

        del clip_model

        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        else:
            self.channel_adapter = nn.Identity()

        if self.freeze:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            if self.vision_model is not None:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
            self.vision_tower.eval()

        self._init_projection()

    @property
    def native_dim(self) -> int:
        return self._native_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_adapted = self.channel_adapter(x)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            vision_outputs = self.vision_tower(pixel_values=x_adapted)
            pooled_output = vision_outputs[1]

            if self.vision_model is not None:
                image_features = self.vision_model(pooled_output)
            else:
                image_features = pooled_output

        return image_features, None
