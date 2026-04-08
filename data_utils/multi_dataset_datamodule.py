"""PyTorch Lightning DataModule that concatenates multiple datasets."""
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from torch.utils.data._utils.collate import default_collate

from utils.dataset_builder import build_datasets_for_mode


def multi_dataset_collate(batch):
    """Pop non-stackable ``image_meta.extra`` before collate; restore after."""
    extras = [sample["image_meta"].pop("extra", {}) for sample in batch]
    for sample in batch:
        if "input_key" not in sample["image_meta"]:
            sample["image_meta"]["input_key"] = ""
    result = default_collate(batch)
    result["image_meta"]["extra"] = extras
    return result


class MultiDatasetDataModule(pl.LightningDataModule):
    """Lightning DataModule over multiple datasets."""
    
    def __init__(self, cfg: Dict[str, Any], project_root: Path):
        super().__init__()
        self.cfg = cfg
        self.project_root = project_root

    def _build_datasets_list(
        self,
        datasets_dict: Dict[str, Dataset],
        datasets_cfg: list,
        max_samples_key: str = "max_samples",
        oversample: bool = False,
    ) -> list:
        """Build list of (possibly Subset) datasets with optional caps and oversampling.

        Args:
            datasets_dict: name -> full Dataset
            datasets_cfg: list of per-dataset configs
            max_samples_key: key for cap (e.g. max_samples)
            oversample: if True and len < cap, repeat indices to reach cap (train only)
        """
        datasets_list = []
        for ds_cfg in datasets_cfg:
            dataset_name = ds_cfg["name"]
            ds_full = datasets_dict[dataset_name]
            
            actual = len(ds_full)
            # Skip empty
            if actual == 0:
                print(f"Warning: Dataset {dataset_name} is empty (size=0), skipping...")
                continue
            
            max_samples = ds_cfg.get(max_samples_key)
            if max_samples is not None:
                max_samples = int(max_samples)
                if max_samples == 0:
                    # max_samples==0 -> skip
                    print(f"Warning: max_samples=0 for dataset {dataset_name}, skipping...")
                    continue
                if actual >= max_samples:
                    # Truncate to first max_samples
                    indices = list(range(max_samples))
                elif oversample:
                    # Oversample by repeating indices
                    repeats = max_samples // actual
                    remainder = max_samples % actual
                    indices = list(range(actual)) * repeats + list(range(remainder))
                else:
                    # Use all samples
                    indices = list(range(actual))
                ds = Subset(ds_full, indices)
                # print(f"  [{dataset_name}] {max_samples_key}={max_samples}, actual={actual}, used={len(indices)}")
            else:
                ds = ds_full
                # print(f"  [{dataset_name}] no limit, used={actual}")
            
            datasets_list.append(ds)
        return datasets_list

    def setup(self, stage: str | None = None) -> None:
        train_datasets_cfg = self.cfg["train_datasets"]
        test_datasets_cfg = self.cfg.get("test_datasets", [])
        
        # Full datasets before Subset
        self.train_datasets = build_datasets_for_mode(train_datasets_cfg, "train", self.project_root)
        self.valid_datasets = build_datasets_for_mode(train_datasets_cfg, "valid", self.project_root)
        self.test_datasets = build_datasets_for_mode(test_datasets_cfg, "test", self.project_root)

        # Train merge with oversampling enabled
        train_datasets_list = self._build_datasets_list(self.train_datasets, train_datasets_cfg, "max_samples", oversample=True)
        self.train_dataset = train_datasets_list[0] if len(train_datasets_list) == 1 else ConcatDataset(train_datasets_list)

    def train_dataloader(self):
        trainer_cfg = self.cfg["trainer"]
        return DataLoader(
            self.train_dataset,
            batch_size=trainer_cfg["batch_size"],
            num_workers=trainer_cfg["num_workers"],
            shuffle=True,
            pin_memory=True,
            collate_fn=multi_dataset_collate,
        )

    def _create_eval_dataloader(
        self, datasets_dict: Dict[str, Dataset], datasets_cfg: list,
        is_valid: bool = False, batch_size_override: int | None = None,
    ) -> DataLoader | None:
        """Val/test DataLoader over concatenated datasets."""
        if len(datasets_dict) == 0:
            return None
        
        max_samples_key = "valid_max_samples" if is_valid else "max_samples"
        eval_datasets_list = self._build_datasets_list(datasets_dict, datasets_cfg, max_samples_key)
        
        trainer_cfg = self.cfg["trainer"]
        dataset = eval_datasets_list[0] if len(eval_datasets_list) == 1 else ConcatDataset(eval_datasets_list)
        batch_size = batch_size_override if batch_size_override is not None else trainer_cfg["batch_size"]
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=trainer_cfg["num_workers"],
            shuffle=False,
            pin_memory=True,
            collate_fn=multi_dataset_collate,
        )

    def val_dataloader(self):
        return self._create_eval_dataloader(
            self.valid_datasets, self.cfg["train_datasets"],
            is_valid=True, batch_size_override=self.cfg["trainer"]["valid_batch_size"],
        )

    def test_dataloader(self):
        return self._create_eval_dataloader(
            self.test_datasets, self.cfg.get("test_datasets", []),
            is_valid=False, batch_size_override=self.cfg["trainer"]["test_batch_size"],
        )
