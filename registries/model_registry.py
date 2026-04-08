"""
Model registry: map string names to nn.Module classes.
(Open-source mirror: only LLaRS is registered.)
"""
from typing import Dict, Type

import torch.nn as nn

from pytorch_models.llars.wrapper import LLaRSNet


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "llars": LLaRSNet,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by registry name.

    Args:
        name: key in MODEL_REGISTRY
        **kwargs: forwarded to the model constructor

    Returns:
        nn.Module instance
    """
    assert name in MODEL_REGISTRY, (
        f"Unknown model name: {name}. Available: {list(MODEL_REGISTRY.keys())}"
    )
    return MODEL_REGISTRY[name](**kwargs)
