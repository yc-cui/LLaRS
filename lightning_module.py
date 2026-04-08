"""
Generic LightningModule:
- Builds the underlying nn.Module from the model registry by name
- forward(batch) returns (pred, loss_dict)
- Shared train/val/test steps, logging, and visualization hooks
- Multi-dataset training with per-dataset metrics
- Optional finetune_cfg and optional input routing (algo_cfg.routing; registry supports sinkhorn_v2)
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.metrics import compute_metrics
from utils.evaluator import Evaluator
from utils.csv_logger import CSVLogger
from registries.model_registry import build_model


class GenericLitModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        lr: float = 1e-4,
        log_dir: str = "logs",
        dataset_names: List[str] = None,
        finetune_cfg: Optional[Dict[str, Any]] = None,
        algo_cfg: Optional[Dict[str, Any]] = None,
        vis_limits: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["log_dir", "vis_limits"])

        self.automatic_optimization = False

        self.model = build_model(model_name, **(model_kwargs or {}))
        self.lr = lr
        self.log_dir = log_dir
        self.dataset_names = dataset_names if dataset_names is not None else ["default"]

        self.finetuner = None
        if finetune_cfg is not None:
            from registries.finetune_registry import build_finetune
            method = finetune_cfg["method"]
            kwargs = finetune_cfg.get("kwargs", {})
            self.finetuner = build_finetune(method, **kwargs)
            self.model = self.finetuner.apply(self.model)
            self.finetuner.print_trainable_info(self.model)

        algo_cfg = algo_cfg or {}

        self.router = None
        routing_cfg = algo_cfg.get("routing", None)
        if routing_cfg is not None:
            from registries.algo_registry import build_router
            self.router = build_router(
                routing_cfg["name"],
                in_channels=routing_cfg.get("kwargs", {}).get("in_channels", 20),
                num_slots=routing_cfg.get("kwargs", {}).get("num_slots", 20),
                **{k: v for k, v in routing_cfg.get("kwargs", {}).items()
                   if k not in ("in_channels", "num_slots")},
            )

        self.evaluators_valid: Dict[str, Evaluator] = {}
        self.evaluators_test: Dict[str, Evaluator] = {}
        for ds_name in self.dataset_names:
            self.evaluators_valid[ds_name] = Evaluator(f"{ds_name}_valid")
            self.evaluators_test[ds_name] = Evaluator(f"{ds_name}_test")

        self.csv_loggers_train: Dict[str, CSVLogger] = {}
        self.csv_loggers_valid: Dict[str, CSVLogger] = {}
        self.csv_loggers_test: Dict[str, CSVLogger] = {}
        self.log_dir_path = Path(log_dir)

        self.train_losses = []

        self.vis_limits = vis_limits or {}
        self._vis_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _ensure_csv_logger(self, ds_name: str, stage: str, metric_names: List[str]):
        """Lazily create CSV logger for a dataset and stage."""
        logger_dict = getattr(self, f"csv_loggers_{stage}")
        if ds_name not in logger_dict:
            csv_path = self.log_dir_path / f"{stage}_{ds_name}.csv"
            logger_dict[ds_name] = CSVLogger(csv_path, metric_names)

    def _get_actual_epoch(self) -> int:
        """
        Epoch index for logging after fit vs during fit.

        After post-fit test, current_epoch may reset; use Lightning progress when available.
        """
        if self.trainer.fit_loop.epoch_progress.current.completed > 0:
            return self.trainer.fit_loop.epoch_progress.current.completed
        return self.current_epoch

    def forward(self, batch: Dict[str, Any]) -> tuple:
        """
        Run model.forward(batch) -> (pred, loss_dict).
        Optionally route padded multi-channel input through self.router first.
        """
        if self.router is not None:
            batch = dict(batch)
            batch["inp"] = self.router(batch["inp"])
        return self.model(batch)

    def _visualize_batch(self, batch: Dict[str, Any], pred: torch.Tensor, mode: str) -> None:
        """Dispatch per-sample visualization via DATASET_REGISTRY."""
        if self.global_rank != 0:
            return

        from registries.dataset_registry import DATASET_REGISTRY

        meta_batch = batch["image_meta"]
        inp = batch["inp"]
        batch_size = inp.shape[0]
        limits = self.vis_limits.get(mode, self.vis_limits.get("train", {}))
        counts = self._vis_counts[mode]

        for i in range(batch_size):
            ds_name = meta_batch["dataset_name"][i]
            if ds_name in limits and counts[ds_name] >= limits[ds_name]:
                continue
            counts[ds_name] += 1

            cls = DATASET_REGISTRY[ds_name]

            image_meta_i = {
                "dataset_name": ds_name,
                "source_dataset": meta_batch["source_dataset"][i] if "source_dataset" in meta_batch else ds_name,
                "input_key": meta_batch["input_key"][i] if "input_key" in meta_batch else "",
                "extra": meta_batch["extra"][i],
            }

            sample = {
                "inp": inp[i].detach().cpu(),
                "gt": batch["gt"][i].detach().cpu(),
                "add": batch["add"][i].detach().cpu(),
                "prompt": batch["prompt"][i],
                "image_meta": image_meta_i,
                "pred": pred[i].detach().cpu(),
            }
            cls.visualize_sample(
                sample=sample,
                log_dir=self.log_dir,
                model_name=self.hparams.model_name,
                epoch=self._get_actual_epoch(),
                mode=mode,
                rank=self.global_rank,
            )

    def on_train_epoch_start(self) -> None:
        self._vis_counts.pop("train", None)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad()

        pred, loss_dict = self._training_step_default(batch, opt)

        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                self.log(f"train/{loss_name}", loss_value, prog_bar=(loss_name == "total_loss"),
                        on_step=True, on_epoch=True, sync_dist=True)

        self.train_losses.append(loss_dict)

        return loss_dict["total_loss"]

    def _training_step_default(self, batch: Dict[str, Any], opt) -> tuple:
        pred, loss_dict = self(batch)
        self.manual_backward(loss_dict["total_loss"])
        opt.step()
        return pred, loss_dict

    def on_train_epoch_end(self) -> None:
        """Write average training losses to CSV on rank 0."""
        if len(self.train_losses) > 0 and self.global_rank == 0:
            all_keys: set = set()
            for d in self.train_losses:
                all_keys.update(d.keys())
            loss_names = sorted(all_keys)

            avg_losses = {}
            for loss_name in loss_names:
                vals = [d[loss_name].item() for d in self.train_losses if loss_name in d]
                avg_losses[loss_name] = sum(vals) / len(vals) if vals else 0.0

            if len(self.dataset_names) > 0:
                ds_name = self.dataset_names[0]
                self._ensure_csv_logger(ds_name, "train", loss_names)
                self.csv_loggers_train[ds_name].log(self.current_epoch, avg_losses)

            self.train_losses = []

    def _shared_eval_step(
        self, batch: Dict[str, Any], batch_idx: int, evaluators: Dict[str, Evaluator], stage: str
    ) -> None:
        if self.trainer.sanity_checking:
            return

        pred, loss_dict = self(batch)

        total_loss = loss_dict["total_loss"]
        self.log(f"{stage}/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        gt = batch["gt"]
        b = pred.size(0)
        meta_batch = batch["image_meta"]
        dataset_names_batch = meta_batch["dataset_name"]
        valid_channels = meta_batch["num_channels"]

        for i in range(b):
            vc = int(valid_channels[i])
            metrics = compute_metrics(pred[i][:vc], gt[i][:vc], max_val=1.0)
            ds_name = dataset_names_batch[i]
            evaluators[ds_name].update_metrics(metrics)

        if self.global_rank == 0:
            self._visualize_batch(batch, pred, stage)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._shared_eval_step(batch, batch_idx, self.evaluators_valid, "valid")

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._shared_eval_step(batch, batch_idx, self.evaluators_test, "test")

    def on_validation_epoch_start(self) -> None:
        self._vis_counts.pop("valid", None)
        for evaluator in self.evaluators_valid.values():
            evaluator.reset()

    def on_test_epoch_start(self) -> None:
        self._vis_counts.pop("test", None)
        for evaluator in self.evaluators_test.values():
            evaluator.reset()

    def _shared_eval_epoch_end(self, evaluators: Dict[str, Evaluator], stage: str):
        if self.trainer.sanity_checking:
            return

        csv_loggers = getattr(self, f"csv_loggers_{stage}")
        actual_epoch = self._get_actual_epoch()

        for ds_name, evaluator in evaluators.items():
            metrics = evaluator.compute()

            if not metrics:
                continue

            metric_names = list(metrics.keys())

            if self.global_rank == 0:
                self._ensure_csv_logger(ds_name, stage, metric_names)
                csv_loggers[ds_name].log(actual_epoch, metrics)

    def on_validation_epoch_end(self) -> None:
        self._shared_eval_epoch_end(self.evaluators_valid, "valid")

    def on_test_epoch_end(self) -> None:
        self._shared_eval_epoch_end(self.evaluators_test, "test")

    def configure_optimizers(self):
        if self.finetuner is not None:
            params = list(self.finetuner.trainable_params(self.model))
        else:
            params = list(self.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-2)
        return opt
