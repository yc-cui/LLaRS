"""
Pansharpening datasets
"""
from .nbu_datasets import (
    PansharpeningDatasetBase,
    Gaofen1Dataset,
    IkonosDataset,
    QuickbirdDataset,
    Worldview2Dataset,
    Worldview3Dataset,
    Worldview4Dataset,
)
from .pancollection_datasets import (
    PanCollectionDatasetBase,
    PcGf2Dataset,
    PcQbDataset,
    PcWv3Dataset,
    PcWv2Dataset,
)

__all__ = [
    "PansharpeningDatasetBase",
    "Gaofen1Dataset",
    "IkonosDataset",
    "QuickbirdDataset",
    "Worldview2Dataset",
    "Worldview3Dataset",
    "Worldview4Dataset",
    "PanCollectionDatasetBase",
    "PcGf2Dataset",
    "PcQbDataset",
    "PcWv3Dataset",
    "PcWv2Dataset",
]
