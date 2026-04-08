from pathlib import Path
from typing import Dict, Any

from data_utils.sim_dataset_base import SimDatasetBase
from utils.vis_utils import save_tensor_image


class BrightnessSimDataset(SimDatasetBase):
    """Brightness sim dataset; viz denormalizes then re-normalizes."""

    @classmethod
    def visualize_sample(
        cls,
        sample: dict,
        log_dir: str,
        model_name: str,
        epoch: int,
        mode: str,
        rank: int = 0,
    ) -> None:
        if rank != 0:
            return

        meta: Dict[str, Any] = sample["image_meta"]
        extra = meta["extra"]
        nc = extra["gt"]["num_channels"]

        save_dir = Path(log_dir) / meta["dataset_name"] / meta["source_dataset"] / mode
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(extra["gt"]["path"]).stem

        for key in ("inp", "gt", "pred"):
            if key not in extra:
                continue
            if key == "gt":
                tensor = sample["gt"][:nc]
            elif key == "pred":
                tensor = sample["pred"][:nc]
            else:
                tensor = sample["inp"][:nc]

            if key == "pred":
                path = save_dir / f"{stem}-{key}-{model_name}-{epoch:02d}.png"
            else:
                path = save_dir / f"{stem}-{key}.png"

            extra[key]["save_path"] = path
            save_tensor_image(tensor, extra[key], percentile_stretch=False)

        from utils.vis_utils import save_allband_npz_and_error_vis
        save_allband_npz_and_error_vis(sample, save_dir, stem, model_name, epoch)

