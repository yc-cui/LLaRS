from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..text_encoder_base import TextEncoderBase


MEAN_POOLING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "TaylorAI/bge-micro-v2",
}


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (
        torch.sum(token_embeddings * input_mask_expanded, 1)
        / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    )


class HuggingFaceTextEncoder(TextEncoderBase):

    def __init__(
        self,
        target_dim: int,
        model_name_or_path: str = "distilbert-base-uncased",
        pooling: str = "auto",
        normalize: bool = False,
        freeze: bool = True,
    ):
        super().__init__(target_dim=target_dim, freeze=freeze)

        self.model_name_or_path = model_name_or_path
        self.normalize = normalize

        if pooling == "auto":
            self.pooling = "mean" if any(m in model_name_or_path for m in MEAN_POOLING_MODELS) else "cls"
        else:
            assert pooling in ("cls", "mean"), f"Unknown pooling: {pooling}"
            self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

        self._native_dim = self.model.config.hidden_size

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self._init_projection()

    @property
    def native_dim(self) -> int:
        return self._native_dim

    def encode(self, text_batch: List[str], device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(
            text_batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.pooling == "mean":
            embedding = _mean_pooling(outputs, inputs["attention_mask"])
        else:
            embedding = outputs.last_hidden_state[:, 0, :]

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
