"""
Text and image encoder registries.
"""
from typing import Dict, Type

from pytorch_models.llars.encoders.text_encoder_base import TextEncoderBase
from pytorch_models.llars.encoders.image_encoder_base import ImageEncoderBase

from pytorch_models.llars.encoders.text import HuggingFaceTextEncoder, ClipTextEncoder

from pytorch_models.llars.encoders.image import ResNetImageEncoder, ClipImageEncoder, DINOv2ImageEncoder


TEXT_ENCODER_REGISTRY: Dict[str, Type[TextEncoderBase]] = {
    "huggingface": HuggingFaceTextEncoder,
    "clip_text": ClipTextEncoder,
}

IMAGE_ENCODER_REGISTRY: Dict[str, Type[ImageEncoderBase]] = {
    "resnet18": ResNetImageEncoder,
    "resnet50": ResNetImageEncoder,
    "clip_image": ClipImageEncoder,
    "dinov2": DINOv2ImageEncoder,
}

_IMAGE_ENCODER_DEFAULTS = {
    "resnet18": {"arch": "resnet18"},
    "resnet50": {"arch": "resnet50"},
}


def build_text_encoder(name: str, **kwargs) -> TextEncoderBase:
    """
    Build a text encoder by registry name.

    Args:
        name: TEXT_ENCODER_REGISTRY key
        **kwargs: must include target_dim among encoder-specific args

    Returns:
        TextEncoderBase; forward -> [B, target_dim]
    """
    assert name in TEXT_ENCODER_REGISTRY, (
        f"Unknown text encoder: {name}. Available: {list(TEXT_ENCODER_REGISTRY.keys())}"
    )
    cls = TEXT_ENCODER_REGISTRY[name]
    return cls(**kwargs)


def build_image_encoder(name: str, **kwargs) -> ImageEncoderBase:
    """
    Build an image encoder by registry name.

    Args:
        name: IMAGE_ENCODER_REGISTRY key
        **kwargs: must include target_dim and in_channels

    Returns:
        ImageEncoderBase; forward -> ([B, target_dim], optional logits)
    """
    assert name in IMAGE_ENCODER_REGISTRY, (
        f"Unknown image encoder: {name}. Available: {list(IMAGE_ENCODER_REGISTRY.keys())}"
    )
    defaults = _IMAGE_ENCODER_DEFAULTS[name]
    merged = {**defaults, **kwargs}
    cls = IMAGE_ENCODER_REGISTRY[name]
    return cls(**merged)
