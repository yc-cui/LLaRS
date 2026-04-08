import torch
import torch.nn.functional as F
from contextvars import ContextVar


_load_loss_collector: ContextVar = ContextVar('load_loss_collector', default=None)
_route_cls_collector: ContextVar = ContextVar('route_cls_collector', default=None)


class LoadLossCollector:

    def __init__(self):
        self.losses = []
        self._token = None

    def __enter__(self):
        self._token = _load_loss_collector.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _load_loss_collector.reset(self._token)
        return False

    def add_loss(self, loss):
        if loss is not None and isinstance(loss, torch.Tensor):
            self.losses.append(loss)

    def get_total_loss(self, device=None):
        if not self.losses:
            if device is not None:
                return torch.tensor(0.0, device=device, requires_grad=True)
            return torch.tensor(0.0, requires_grad=True)

        total = sum(self.losses)
        return total


def get_current_collector():
    return _load_loss_collector.get()


def call_moe_with_auto_loss(moe_module, *args, **kwargs):
    output, load_loss = moe_module(*args, **kwargs)
    collector = get_current_collector()
    if collector is not None:
        collector.add_loss(load_loss)
    return output


def get_batch_metadata():
    return {}


class RouteClsCollector:

    def __init__(self):
        self.logits = []
        self._token = None

    def __enter__(self):
        self.logits.clear()
        self._token = _route_cls_collector.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _route_cls_collector.reset(self._token)
        return False

    def add_logits(self, logits: torch.Tensor) -> None:
        if logits is not None:
            self.logits.append(logits)

    def compute_route_cls_loss(
        self,
        labels: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if labels is None or not self.logits:
            return torch.zeros((), device=device, dtype=dtype)
        mean_logits = torch.stack(self.logits, dim=0).mean(dim=0)
        return F.cross_entropy(mean_logits, labels.long().to(device))


def get_route_cls_collector():
    return _route_cls_collector.get()
