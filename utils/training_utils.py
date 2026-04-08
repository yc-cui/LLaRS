"""Helpers to construct the Lightning model and Trainer."""
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class MinEpochModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that only saves after ``min_save_epoch``."""

    def __init__(self, min_save_epoch: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.min_save_epoch = min_save_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.min_save_epoch:
            return
        super().on_train_epoch_end(trainer, pl_module)


def create_model(
    model_cfg: Dict[str, Any],
    trainer_cfg: Dict[str, Any],
    log_dir: Path,
    dataset_names: list,
    resume_ckpt_path: str | None,
    project_root: Path,
    finetune_cfg: Dict[str, Any] | None = None,
    algo_cfg: Dict[str, Any] | None = None,
    vis_limits: Dict[str, Dict[str, int]] | None = None,
):
    """Instantiate or load checkpoint into GenericLitModel."""
    if resume_ckpt_path:
        ckpt_path = project_root / resume_ckpt_path
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        from lightning_module import GenericLitModel
        lit_model = GenericLitModel.load_from_checkpoint(str(ckpt_path))
        if finetune_cfg is not None and lit_model.finetuner is None:
            from registries.finetune_registry import build_finetune
            method = finetune_cfg["method"]
            kwargs = finetune_cfg.get("kwargs", {})
            lit_model.finetuner = build_finetune(method, **kwargs)
            lit_model.model = lit_model.finetuner.apply(lit_model.model)
            lit_model.finetuner.print_trainable_info(lit_model.model)

        lit_model.lr = trainer_cfg["lr"]
        lit_model.log_dir = str(log_dir)
        lit_model.dataset_names = dataset_names
        lit_model.log_dir_path = log_dir
        lit_model.vis_limits = vis_limits or {}

        from utils.evaluator import Evaluator
        lit_model.evaluators_valid = {}
        lit_model.evaluators_test = {}
        for ds_name in dataset_names:
            lit_model.evaluators_valid[ds_name] = Evaluator(f"{ds_name}_valid")
            lit_model.evaluators_test[ds_name] = Evaluator(f"{ds_name}_test")

        lit_model.csv_loggers_train = {}
        lit_model.csv_loggers_valid = {}
        lit_model.csv_loggers_test = {}

        return lit_model
    else:
        from lightning_module import GenericLitModel
        return GenericLitModel(
            model_name=model_cfg["name"],
            model_kwargs=model_cfg.get("kwargs", {}),
            lr=trainer_cfg["lr"],
            log_dir=str(log_dir),
            dataset_names=dataset_names,
            finetune_cfg=finetune_cfg,
            algo_cfg=algo_cfg,
            vis_limits=vis_limits,
        )


def create_trainer(trainer_cfg: Dict[str, Any], log_dir: Path, dataset_names: list) -> pl.Trainer:
    """Build ``pl.Trainer`` with checkpointing hooks."""
    devices = trainer_cfg.get("devices", None)
    strategy = trainer_cfg.get("strategy", None)
    check_val_every_n_epoch = trainer_cfg.get("check_val_every_n_epoch", 1)
    save_every_n_epochs = trainer_cfg.get("save_every_n_epochs", 10)
    min_save_epoch = trainer_cfg.get("min_save_epoch", 100)
    
    callbacks = []

    # Single-dataset runs: isolate ckpt dir so parallel split jobs do not clobber each other
    if len(dataset_names) == 1:
        ckpt_dir = log_dir / "ckpt" / dataset_names[0]
    else:
        ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save every N epochs; MinEpochModelCheckpoint skips before min_save_epoch
    checkpoint_callback = MinEpochModelCheckpoint(
        min_save_epoch=min_save_epoch,
        dirpath=str(ckpt_dir),
        filename="epoch={epoch:04d}",
        every_n_epochs=save_every_n_epochs,
        save_top_k=-1,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    callbacks.append(checkpoint_callback)
    
    trainer_kwargs = {
        "max_epochs": trainer_cfg["max_epochs"],
        "default_root_dir": str(log_dir),
        "precision": trainer_cfg["precision"],
        "accelerator": "gpu",
        "log_every_n_steps": 10,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "callbacks": callbacks,
        "enable_progress_bar": True,
    }
    
    if devices is not None:
        trainer_kwargs["devices"] = devices
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    
    return pl.Trainer(**trainer_kwargs)
