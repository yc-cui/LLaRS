"""
PyTorch Lightning training entrypoint:
- Loads model, dataset, and trainer settings from a JSON config
- Supports concatenated multi-dataset training (first N samples per dataset via max_samples)
- Separate train and test dataset sections
- Uses GenericLitModel for fit/validate/test
- Instantiates dataset classes by registered names
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Redirect Python warnings to temp/warnings.log (not the console)
_warn_log_path = PROJECT_ROOT / "temp" / "warnings.log"
_warn_log_path.parent.mkdir(parents=True, exist_ok=True)
_warn_fh = open(_warn_log_path, "w", encoding="utf-8")


def _warn_to_file(message, category, filename, lineno, file=None, line=None):
    _warn_fh.write(warnings.formatwarning(message, category, filename, lineno, line))
    _warn_fh.flush()


warnings.showwarning = _warn_to_file

logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

from data_utils.multi_dataset_datamodule import MultiDatasetDataModule  # noqa: E402
from utils.config_utils import get_log_dir_from_config_path, load_config  # noqa: E402
from utils.dataset_builder import extract_dataset_names  # noqa: E402
from utils.logging_utils import setup_logging  # noqa: E402
from utils.training_utils import create_model, create_trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="config/test.json",
        help="Path to JSON config (relative to project root or absolute); default config/smoke_llars.json",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    cfg = load_config(cfg_path)

    log_dir = get_log_dir_from_config_path(cfg_path, PROJECT_ROOT)
    trainer_cfg = cfg["trainer"]
    if "log_dir" in trainer_cfg:
        log_dir = PROJECT_ROOT / trainer_cfg["log_dir"]

    only_valid = trainer_cfg.get("only_valid", False)
    only_test = trainer_cfg.get("only_test", False)

    setup_logging(log_dir, cfg, cfg_path)

    dm = MultiDatasetDataModule(cfg, PROJECT_ROOT)
    model_cfg = cfg["model"]
    finetune_cfg = cfg.get("finetune", None)
    algo_cfg = cfg.get("algo", None)
    dataset_names = extract_dataset_names(cfg, PROJECT_ROOT)

    resume_ckpt_path = trainer_cfg.get("resume_ckpt_path")

    vis_limits = {"train": {}, "valid": {}}
    for ds_cfg in cfg.get("train_datasets", []):
        n = ds_cfg["name"]
        vis_limits["train"][n] = ds_cfg.get("max_vis_samples", 100)
        vis_limits["valid"][n] = ds_cfg.get("valid_max_vis_samples", 30)

    lit_model = create_model(
        model_cfg, trainer_cfg, log_dir, dataset_names,
        resume_ckpt_path, PROJECT_ROOT, finetune_cfg=finetune_cfg,
        algo_cfg=algo_cfg, vis_limits=vis_limits,
    )

    finetune_modifies_structure = (
        finetune_cfg is not None
        and finetune_cfg.get("method") not in (None, "zero_shot", "full")
    )
    load_weights_only = trainer_cfg.get("load_weights_only", False)
    if resume_ckpt_path and not finetune_modifies_structure and not load_weights_only:
        import torch
        ckpt_meta = torch.load(
            str(PROJECT_ROOT / resume_ckpt_path), map_location="cpu", weights_only=False
        )
        ckpt_epoch = ckpt_meta.get("epoch", 0)
        max_epochs = trainer_cfg.get("max_epochs", float("inf"))
        if ckpt_epoch >= max_epochs:
            print(f"[INFO] Checkpoint epoch ({ckpt_epoch}) >= max_epochs ({max_epochs}), "
                  f"loading weights only (skip training state).")
            ckpt_path_str = None
        else:
            ckpt_path_str = str(PROJECT_ROOT / resume_ckpt_path)
        del ckpt_meta
    else:
        ckpt_path_str = None

    if only_valid or only_test:
        dm.setup(stage=None)
        if only_valid:
            val_dl = dm.val_dataloader()
            if val_dl is None or len(val_dl.dataset) == 0:
                raise ValueError("only_valid=True but no validation data (train_datasets have no valid/test split).")
        if only_test:
            test_dl = dm.test_dataloader()
            if test_dl is None or len(test_dl.dataset) == 0:
                raise ValueError("only_test=True but no test data (test_datasets have no test/valid split).")
        trainer = create_trainer(trainer_cfg, log_dir, dataset_names)
        if only_valid:
            trainer.validate(lit_model, datamodule=dm, ckpt_path=ckpt_path_str)
        if only_test:
            trainer.test(lit_model, datamodule=dm, ckpt_path=ckpt_path_str)
    else:
        trainer = create_trainer(trainer_cfg, log_dir, dataset_names)
        trainer.fit(lit_model, datamodule=dm, ckpt_path=ckpt_path_str)

        devices = trainer_cfg.get("devices", None)
        if devices and isinstance(devices, list) and len(devices) > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

            if rank == 0 and cfg.get("test_datasets"):
                print("\n" + "=" * 80)
                print("Training completed. Starting testing with single device to avoid sample duplication...")
                print("=" * 80)

                trainer_cfg_single = trainer_cfg.copy()
                trainer_cfg_single["devices"] = [devices[0]]
                trainer_cfg_single["strategy"] = None

                test_trainer = create_trainer(trainer_cfg_single, log_dir, dataset_names)
                test_trainer.test(lit_model, datamodule=dm)
        else:
            if cfg.get("test_datasets"):
                trainer.test(lit_model, datamodule=dm)


if __name__ == "__main__":
    main()
