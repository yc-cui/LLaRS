"""CSV logger for per-epoch aggregated val/test metrics."""
import csv
from pathlib import Path
from typing import Dict, List


class CSVLogger:
    """Append one row per epoch with aggregated metrics."""
    
    def __init__(self, csv_path: Path, metric_names: List[str]):
        """
        Args:
            csv_path: output CSV path
            metric_names: column names after epoch, e.g. ["psnr", "ssim"]
        """
        self.csv_path = Path(csv_path)
        self.metric_names = metric_names
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write header if new file
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch"] + metric_names)
    
    def log(self, epoch: int, metrics: Dict[str, float]):
        """
        Log one epoch row.

        Args:
            epoch: epoch index
            metrics: values for metric_names, e.g. {"psnr": 32.1, "ssim": 0.95}
        """
        row = [epoch]
        for metric_name in self.metric_names:
            row.append(metrics[metric_name])
        
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
