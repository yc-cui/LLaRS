"""Config and meta JSON helpers."""
import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path | str) -> Dict[str, Any]:
    """Load JSON config file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(meta_file: Path | str) -> Dict[str, Any]:
    """Load dataset meta JSON and return full dict."""
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def get_log_dir_from_config_path(config_path: Path, project_root: Path) -> Path:
    """
    Derive log directory from config path.

    Examples:
    - config/multi_dataset_example.json -> logs/multi_dataset_example
    - config/sr/oli2msi.json -> logs/sr/oli2msi

    Args:
        config_path: path to config (absolute or relative to project_root)
        project_root: repository root

    Returns:
        Absolute path under project_root for logs
    """
    config_path = Path(config_path)
    project_root = Path(project_root)
    
    # Resolve path relative to project_root if absolute
    if config_path.is_absolute():
        rel_path = config_path.relative_to(project_root)
    else:
        rel_path = config_path
    
    # Strip .json
    if rel_path.suffix == ".json":
        rel_path = rel_path.with_suffix("")
    
    # config/... -> logs/...
    if str(rel_path).startswith("config/"):
        log_rel_path = Path("logs") / rel_path.relative_to("config")
    else:
        # Otherwise logs/<basename>
        log_rel_path = Path("logs") / rel_path.name
    
    return project_root / log_rel_path

