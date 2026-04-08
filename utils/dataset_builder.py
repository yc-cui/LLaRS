"""Build torch Datasets from training config."""
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset

from constants import DEFAULT_PROMPT_FILE, DATASET_CONFIGS
from registries.dataset_registry import build_dataset


def build_datasets_for_mode(
    datasets_cfg: List[Dict[str, Any]], mode: str, project_root: Path
) -> Dict[str, Dataset]:
    """Build {dataset_name: Dataset} for train/valid/test mode."""
    from utils.sim_ops import SIM_OP_GROUPS

    datasets_dict = {}
    prompt_file = str(project_root / DEFAULT_PROMPT_FILE)

    for ds_cfg in datasets_cfg:
        name = ds_cfg["name"]
        seed = ds_cfg.get("seed", 0) + (0 if mode == "train" else 1 if mode == "valid" else 2)
        cfg = DATASET_CONFIGS[name]

        kwargs: Dict[str, Any] = {
            "meta_file": str(project_root / cfg["meta_file"]),
            "prompt_file": prompt_file,
            "mode": mode,
        }
        # Simulated data: sim_ops + text_deg_type for SimDatasetBase
        if "sim_ops" in cfg:
            sim_key = cfg["sim_ops"]
            kwargs.update(
                sim_ops=SIM_OP_GROUPS[sim_key],
                text_deg_type=cfg["text_deg_type"],
                seed=seed,
            )

        datasets_dict[name] = build_dataset(name, **kwargs)

    return datasets_dict


def extract_dataset_names(cfg: Dict[str, Any], project_root: Path = None) -> List[str]:
    """Collect unique dataset names from train_datasets and test_datasets."""
    dataset_names = []
    for ds_cfg in cfg["train_datasets"] + cfg.get("test_datasets", []):
        ds_name = ds_cfg["name"]
        if ds_name not in dataset_names:
            dataset_names.append(ds_name)
    return dataset_names
