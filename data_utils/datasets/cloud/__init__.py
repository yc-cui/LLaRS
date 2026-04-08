"""
Cloud removal / Haze removal datasets (non-SAR)
"""
from .cuhk_datasets import CuhkCr1Dataset, CuhkCr2Dataset
from .rice_datasets import Rice1Dataset, Rice2Dataset
from .haze1k_datasets import Haze1kThinDataset, Haze1kModerateDataset, Haze1kThickDataset
from .rrshid_datasets import RrshidThinDataset, RrshidModerateDataset, RrshidThickDataset
from .rsid_dataset import RsidDataset
from .hazy_rs_datasets import DhidDataset, LhidDataset

__all__ = [
    "CuhkCr1Dataset",
    "CuhkCr2Dataset",
    "Rice1Dataset",
    "Rice2Dataset",
    "Haze1kThinDataset",
    "Haze1kModerateDataset",
    "Haze1kThickDataset",
    "RrshidThinDataset",
    "RrshidModerateDataset",
    "RrshidThickDataset",
    "RsidDataset",
    "DhidDataset",
    "LhidDataset",
]
