from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..image_encoder_base import ImageEncoderBase


_RESNET_FEATURE_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}


class ResNetImageEncoder(ImageEncoderBase):

    def __init__(
        self,
        target_dim: int,
        in_channels: int = 16,
        arch: str = "resnet18",
        num_classes: int = 4,
        pretrained: bool = False,
        freeze: bool = False,
    ):
        super().__init__(target_dim=target_dim, in_channels=in_channels, freeze=freeze)

        assert arch in _RESNET_FEATURE_DIMS, (
            f"Unknown ResNet arch: {arch}. Available: {list(_RESNET_FEATURE_DIMS.keys())}"
        )
        self.arch = arch
        self._native_dim = _RESNET_FEATURE_DIMS[arch]

        import torchvision.models as tv_models
        build_fn = getattr(tv_models, arch)
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = build_fn(weights=weights)

        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if num_classes > 0:
            self.classifier = nn.Linear(self._native_dim, num_classes)
        else:
            self.classifier = None

        if self.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._init_projection()

    @property
    def native_dim(self) -> int:
        return self._native_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pooled = self.backbone(x)
        logits = self.classifier(pooled) if self.classifier is not None else None
        return pooled, logits
