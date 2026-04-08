"""Factories that return ``reader(item: dict) -> torch.Tensor`` (C, H, W) float32.

Shared readers (GeoTIFF, PNG) are reusable; exotic formats get dedicated factories.
"""
import numpy as np
import torch
import rasterio
import scipy.io
from typing import Callable, Dict, Any

from constants import resolve_data_file_path

ItemReader = Callable[[Dict[str, Any]], torch.Tensor]


# ---------------------- Generic ----------------------

def create_tif_reader(item_key: str) -> ItemReader:
    """GeoTIFF from ``item[item_key]``."""
    def read(item: dict) -> torch.Tensor:
        with rasterio.open(str(resolve_data_file_path(item[item_key]))) as src:
            arr = src.read()  # (C, H, W)
        return torch.from_numpy(arr.astype(np.float32))
    return read


def create_png_reader(item_key: str, mode: str = "RGB") -> ItemReader:
    """PNG/JPEG from ``item[item_key]``."""
    from PIL import Image

    def read(item: dict) -> torch.Tensor:
        img = Image.open(str(resolve_data_file_path(item[item_key]))).convert(mode)
        arr = np.array(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        return torch.from_numpy(arr)
    return read


# ====================== NBU Pansharp (MAT) ======================

_NBU_MS_KEYS = ("I_MS", "imgMS")
_NBU_PAN_KEYS = ("I_PAN", "imgPAN", "block")


def _find_mat_key(data: dict, candidates: tuple) -> str:
    """First matching key in MAT struct; some sensors use inconsistent names."""
    for k in candidates:
        if k in data:
            return k
    actual = [k for k in data if not k.startswith("_")]
    raise KeyError(f"MAT file has no expected key. candidates={candidates}, actual={actual}")


def create_nbu_ms_reader(mat_key: str) -> ItemReader:
    """NBU MS branch: ``item["ms"]``; uses ``mat_key`` or falls back to known keys."""
    def read(item: dict) -> torch.Tensor:
        data = scipy.io.loadmat(str(resolve_data_file_path(item["ms"])))
        key = mat_key if mat_key in data else _find_mat_key(data, _NBU_MS_KEYS)
        arr = data[key]  # (H, W, C)
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        return torch.from_numpy(arr.astype(np.float32))
    return read


def create_nbu_pan_reader(mat_key: str) -> ItemReader:
    """NBU PAN branch: ``item["pan"]``; same key fallback as MS."""
    def read(item: dict) -> torch.Tensor:
        data = scipy.io.loadmat(str(resolve_data_file_path(item["pan"])))
        key = mat_key if mat_key in data else _find_mat_key(data, _NBU_PAN_KEYS)
        arr = data[key]  # (H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # (1, H, W)
        return torch.from_numpy(arr.astype(np.float32))
    return read


# ====================== PanCollection (HDF5) ======================

def create_pancollection_reader(h5_key: str) -> ItemReader:
    """PanCollection HDF5: ``f[h5_key][item["idx"]]`` from ``item["h5_file"]``."""
    import h5py

    def read(item: dict) -> torch.Tensor:
        with h5py.File(str(resolve_data_file_path(item["h5_file"])), "r") as f:
            arr = f[h5_key][item["idx"]]  # (C, H, W)
        return torch.from_numpy(arr.astype(np.float32))
    return read


# ====================== Sen2Venus (PyTorch .pt) ======================

def create_sen2venus_reader(item_key: str) -> ItemReader:
    """Sen2Venus stacked patches: slice ``item["patch_idx"]`` from ``item[item_key]`` .pt."""
    def read(item: dict) -> torch.Tensor:
        t = torch.load(
            str(resolve_data_file_path(item[item_key])),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        return t[item["patch_idx"]].float()
    return read


# ====================== SAR Despeckle (MAT) ======================

def create_sar_despeckle_reader(mat_key: str) -> ItemReader:
    """SAR despeckle MAT: ``data[mat_key]`` from ``item["mat_path"]``."""
    def read(item: dict) -> torch.Tensor:
        data = scipy.io.loadmat(str(resolve_data_file_path(item["mat_path"])))
        arr = data[mat_key]  # (H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # (1, H, W)
        return torch.from_numpy(arr.astype(np.float32))
    return read
