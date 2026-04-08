"""
Finetune-method registry: name -> strategy class.
"""
from typing import Dict, Type

from algo.finetune.base import FinetuneBase
from algo.finetune.lora import LoRAFinetune
from algo.finetune.dora import DoRAFinetune
from algo.finetune.bitfit import BitFitFinetune
from algo.finetune.ssf import SSFFinetune
from algo.finetune.adapter import AdapterFinetune
from algo.finetune.full import FullFinetune


FINETUNE_REGISTRY: Dict[str, Type[FinetuneBase]] = {
    "lora": LoRAFinetune,
    "dora": DoRAFinetune,
    "bitfit": BitFitFinetune,
    "ssf": SSFFinetune,
    "adapter": AdapterFinetune,
    "full": FullFinetune,
}


def build_finetune(name: str, **kwargs) -> FinetuneBase:
    """
    Instantiate a finetune strategy.

    Args:
        name: lora / dora / bitfit / ssf / adapter / full
        **kwargs: forwarded to the strategy constructor

    Returns:
        FinetuneBase instance
    """
    assert name in FINETUNE_REGISTRY, (
        f"Unknown finetune method: {name}. "
        f"Available: {list(FINETUNE_REGISTRY.keys())}"
    )
    return FINETUNE_REGISTRY[name](**kwargs)
