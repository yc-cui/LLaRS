"""
Algorithm registry (open-source mirror).
ROUTING_REGISTRY: differentiable input routing (Sinkhorn v2 only).
"""
from typing import Dict, Type

from algo.routing.base import RouterBase
from algo.routing.sinkhorn_v2 import SinkhornRouterV2


ROUTING_REGISTRY: Dict[str, Type[RouterBase]] = {
    "sinkhorn_v2": SinkhornRouterV2,
}


def build_router(name: str, **kwargs) -> RouterBase:
    """
    Build a routing module by name.

    Args:
        name: registry key (sinkhorn_v2)
        **kwargs: forwarded to the router constructor

    Returns:
        RouterBase subclass instance
    """
    assert name in ROUTING_REGISTRY, (
        f"Unknown routing algorithm: {name}. "
        f"Available: {list(ROUTING_REGISTRY.keys())}"
    )
    return ROUTING_REGISTRY[name](**kwargs)
