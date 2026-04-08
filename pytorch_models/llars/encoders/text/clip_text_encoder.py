from typing import List

import torch
from transformers import CLIPModel, CLIPTokenizer

from ..text_encoder_base import TextEncoderBase


class ClipTextEncoder(TextEncoderBase):

    def __init__(
        self,
        target_dim: int,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        freeze: bool = True,
    ):
        super().__init__(target_dim=target_dim, freeze=freeze)

        self.model_name_or_path = model_name_or_path

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
        clip_model = CLIPModel.from_pretrained(model_name_or_path)

        self.text_model = clip_model.text_model
        self.text_projection = clip_model.text_projection

        self._native_dim = clip_model.config.projection_dim

        del clip_model

        if self.freeze:
            for param in self.text_model.parameters():
                param.requires_grad = False
            if self.text_projection is not None:
                for param in self.text_projection.parameters():
                    param.requires_grad = False
            self.text_model.eval()

        self._init_projection()

    @property
    def native_dim(self) -> int:
        return self._native_dim

    def encode(self, text_batch: List[str], device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(
            text_batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            pooled_output = outputs[1]
            if self.text_projection is not None:
                text_features = self.text_projection(pooled_output)
            else:
                text_features = pooled_output

        return text_features
