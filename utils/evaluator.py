"""Metric aggregation for single- and multi-GPU training."""
import math

import torch
from typing import Dict, Any, Optional


class Evaluator:
    """
    Per-(dataset, mode) metric sums; distributed-safe.

    Single process: local sums. Distributed: all-reduce in compute(); rank 0 gets means.
    """

    def __init__(self, name: str):
        """
        Args:
            name: e.g. "cloud_sen12mscr_valid" / "sr_oli2msi_test"
        """
        self.name = name
        self.count = 0
        self.sums: Dict[str, float] = {}
        self.metric_counts: Dict[str, int] = {}

        # Distributed init check
        self.is_distributed = torch.distributed.is_initialized() if torch.distributed.is_available() else False
        if self.is_distributed:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def update(self, pred: torch.Tensor, gt: torch.Tensor, meta: Dict[str, Any]):
        """
        Accumulate one sample locally (no cross-rank sync here).

        Args:
            pred: (C, H, W)
            gt:   (C, H, W)
            meta: sample['image_meta'] (dataset_name, text_deg_type, ...)
        """
        # Stub: subclass or external code can sum metrics here, e.g. compute_metrics
        # metrics = compute_metrics(pred, gt)
        # self.count += 1
        # for k, v in metrics.items():
        #     self.sums[k] = self.sums.get(k, 0.0) + v
        pass

    def update_metrics(self, metrics: Dict[str, float]):
        """
        Add precomputed metrics; NaN values skip sum but keys are tracked.

        Args:
            metrics: e.g. {"psnr": 32.1, "ssim": 0.95}
        """
        self.count += 1
        for k, v in metrics.items():
            fv = float(v)
            self.sums.setdefault(k, 0.0)
            self.metric_counts.setdefault(k, 0)
            if math.isnan(fv):
                continue
            self.sums[k] += fv
            self.metric_counts[k] += 1

    def compute(self) -> Dict[str, float]:
        """
        Mean metrics; distributed all-reduce, non-zero rank returns {}.

        Returns:
            e.g. {"psnr": 32.1, "ssim": 0.95}
        """
        if self.count == 0:
            return {}

        if self.is_distributed:
            count_tensors = {k: torch.tensor(float(v), dtype=torch.float32)
                            for k, v in self.metric_counts.items()}
            sums_tensors = {k: torch.tensor(v, dtype=torch.float32)
                           for k, v in self.sums.items()}

            for k in sums_tensors:
                torch.distributed.all_reduce(sums_tensors[k], op=torch.distributed.ReduceOp.SUM)
            for k in count_tensors:
                torch.distributed.all_reduce(count_tensors[k], op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                result = {}
                for k in sums_tensors:
                    cnt = count_tensors[k].item()
                    result[k] = sums_tensors[k].item() / cnt if cnt > 0 else float("nan")
                return result
            else:
                return {}
        else:
            result = {}
            for k, v in self.sums.items():
                cnt = self.metric_counts.get(k, 0)
                result[k] = v / cnt if cnt > 0 else float("nan")
            return result

    def reset(self):
        """Clear running sums."""
        self.count = 0
        self.sums.clear()
        self.metric_counts.clear()


def create_evaluator_dict(dataset_names: list, modes: list = ["train", "valid", "test"]) -> Dict[tuple, Evaluator]:
    """
    Build {(dataset_name, mode): Evaluator}.

    Args:
        dataset_names: e.g. ["cloud_sen12mscr", "sr_oli2msi"]
        modes: e.g. ["train", "valid", "test"]
    """
    evaluators = {}
    for ds_name in dataset_names:
        for mode in modes:
            key = (ds_name, mode)
            evaluators[key] = Evaluator(f"{ds_name}_{mode}")
    return evaluators

