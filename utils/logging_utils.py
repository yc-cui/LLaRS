"""Logging setup helpers."""
import json
from pathlib import Path
from typing import Any, Dict


def setup_logging(log_dir: Path, cfg: Dict[str, Any], cfg_path: Path) -> None:
    """
    Create log directory and copy config JSON into it.

    Args:
        log_dir: log directory
        cfg: config dict
        cfg_path: path to config file (unused here; kept for API compatibility)
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Copy config snapshot into log dir
    config_copy_path = log_dir / "config.json"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    
    print(f"Log directory: {log_dir}")
    print(f"Config copied to: {config_copy_path}")
