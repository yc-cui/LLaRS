"""
Dataset registry.
- Paths and options come from constants.DATASET_CONFIGS
- Keys use snake_case (e.g. sr_oli2msi, sim_denoise_gauss_005)
"""
from typing import Dict, Type, Union

from data_utils.dataset_base import DatasetBase
from data_utils.sim_dataset_base import SimDatasetBase
from data_utils.datasets.sim.brightness_datasets import BrightnessSimDataset

# ========================================================
# Real-data dataset classes
# ========================================================

from data_utils.datasets.sr import Oli2MsiDataset, Sen2VenusDataset

from data_utils.datasets.cloud import (
    CuhkCr1Dataset, CuhkCr2Dataset,
    Rice1Dataset, Rice2Dataset,
    Haze1kThinDataset, Haze1kModerateDataset, Haze1kThickDataset,
    RrshidThinDataset, RrshidModerateDataset, RrshidThickDataset,
    RsidDataset, DhidDataset, LhidDataset,
)

from data_utils.datasets.cloud_sar import (
    Sen12mscrSpringDataset, Sen12mscrSummerDataset,
    Sen12mscrFallDataset, Sen12mscrWinterDataset,
)

from data_utils.datasets.pansharp import (
    Gaofen1Dataset, IkonosDataset, QuickbirdDataset,
    Worldview2Dataset, Worldview3Dataset, Worldview4Dataset,
    PcGf2Dataset, PcQbDataset, PcWv3Dataset, PcWv2Dataset,
)

from data_utils.datasets.stf import (
    CiaDataset, AhbDataset, DaxingDataset, LgcDataset, TianjinDataset,
)

from data_utils.datasets.noise import SarFilterDataset, SarDespeckleDataset

# ========================================================
# Registry map
# ========================================================

DATASET_REGISTRY: Dict[str, Type[Union[DatasetBase, SimDatasetBase]]] = {
    # SR
    "sr_oli2msi": Oli2MsiDataset,
    "sr_sen2venus_alsace": Sen2VenusDataset,
    "sr_sen2venus_anji": Sen2VenusDataset,
    "sr_sen2venus_arm": Sen2VenusDataset,
    "sr_sen2venus_atto": Sen2VenusDataset,
    "sr_sen2venus_bambenw2": Sen2VenusDataset,
    "sr_sen2venus_benga": Sen2VenusDataset,
    "sr_sen2venus_es_ic3xg": Sen2VenusDataset,
    "sr_sen2venus_es_ltera": Sen2VenusDataset,
    "sr_sen2venus_esgisb_1": Sen2VenusDataset,
    "sr_sen2venus_esgisb_2": Sen2VenusDataset,
    "sr_sen2venus_esgisb_3": Sen2VenusDataset,
    "sr_sen2venus_estuamar": Sen2VenusDataset,
    "sr_sen2venus_fgmanaus": Sen2VenusDataset,
    "sr_sen2venus_fr_bil": Sen2VenusDataset,
    "sr_sen2venus_fr_lam": Sen2VenusDataset,
    "sr_sen2venus_fr_lq1": Sen2VenusDataset,
    "sr_sen2venus_jam2018": Sen2VenusDataset,
    "sr_sen2venus_k34_amaz": Sen2VenusDataset,
    "sr_sen2venus_kudaliar": Sen2VenusDataset,
    "sr_sen2venus_lerida_1": Sen2VenusDataset,
    "sr_sen2venus_mad_ambo": Sen2VenusDataset,
    "sr_sen2venus_naryn": Sen2VenusDataset,
    "sr_sen2venus_so1": Sen2VenusDataset,
    "sr_sen2venus_so2": Sen2VenusDataset,
    "sr_sen2venus_sudoue_2": Sen2VenusDataset,
    "sr_sen2venus_sudoue_3": Sen2VenusDataset,
    "sr_sen2venus_sudoue_4": Sen2VenusDataset,
    "sr_sen2venus_sudoue_5": Sen2VenusDataset,
    "sr_sen2venus_sudoue_6": Sen2VenusDataset,
    # Cloud removal
    "cloud_cuhk_cr1": CuhkCr1Dataset,
    "cloud_cuhk_cr2": CuhkCr2Dataset,
    "cloud_rice1": Rice1Dataset,
    "cloud_rice2": Rice2Dataset,
    # Cloud removal (SAR)
    "cloud_sar_sen12mscr_spring": Sen12mscrSpringDataset,
    "cloud_sar_sen12mscr_summer": Sen12mscrSummerDataset,
    "cloud_sar_sen12mscr_fall":   Sen12mscrFallDataset,
    "cloud_sar_sen12mscr_winter": Sen12mscrWinterDataset,
    # Haze removal
    "haze_haze1k_thin": Haze1kThinDataset,
    "haze_haze1k_moderate": Haze1kModerateDataset,
    "haze_haze1k_thick": Haze1kThickDataset,
    "haze_rrshid_thin": RrshidThinDataset,
    "haze_rrshid_moderate": RrshidModerateDataset,
    "haze_rrshid_thick": RrshidThickDataset,
    "haze_rsid": RsidDataset,
    "haze_dhid": DhidDataset,
    "haze_lhid": LhidDataset,
    # Simulated (mild degradation)
    "sim_denoise_weak": SimDatasetBase,
    "sim_deblur_weak": SimDatasetBase,
    # Simulated (generic + weak/medium/strong tiers)
    "sim_denoise": SimDatasetBase,
    "sim_denoise_medium": SimDatasetBase,
    "sim_denoise_strong": SimDatasetBase,

    "sim_deblur": SimDatasetBase,
    "sim_deblur_medium": SimDatasetBase,
    "sim_deblur_strong": SimDatasetBase,

    "sim_sr": SimDatasetBase,
    "sim_sr_weak": SimDatasetBase,
    "sim_sr_medium": SimDatasetBase,
    "sim_sr_strong": SimDatasetBase,

    "sim_destripe": SimDatasetBase,
    "sim_destripe_weak": SimDatasetBase,
    "sim_destripe_medium": SimDatasetBase,
    "sim_destripe_strong": SimDatasetBase,
    "sim_destripe_nr_fixed_weak": SimDatasetBase,
    "sim_destripe_nr_fixed_medium": SimDatasetBase,
    "sim_destripe_nr_fixed_strong": SimDatasetBase,
    "sim_destripe_nr_fixed_alt_weak": SimDatasetBase,
    "sim_destripe_nr_fixed_alt_medium": SimDatasetBase,
    "sim_destripe_nr_fixed_alt_strong": SimDatasetBase,

    # Simulated brightness (tiered)
    "sim_brightness_increase_weak": BrightnessSimDataset,
    "sim_brightness_increase_medium": BrightnessSimDataset,
    "sim_brightness_increase_strong": BrightnessSimDataset,
    "sim_brightness_decrease_weak": BrightnessSimDataset,
    "sim_brightness_decrease_medium": BrightnessSimDataset,
    "sim_brightness_decrease_strong": BrightnessSimDataset,
    "sim_linear_stretch": SimDatasetBase,
    "sim_equalize": SimDatasetBase,
    "sim_denoise_gauss_001": SimDatasetBase,
    "sim_denoise_gauss_005": SimDatasetBase,
    "sim_denoise_gauss_010": SimDatasetBase,
    "sim_denoise_gauss_025": SimDatasetBase,
    "sim_denoise_gauss_050": SimDatasetBase,
    "sim_denoise_sp_001": SimDatasetBase,
    "sim_denoise_sp_005": SimDatasetBase,
    "sim_denoise_sp_010": SimDatasetBase,
    "sim_deblur_gauss_k3": SimDatasetBase,
    "sim_deblur_gauss_k5": SimDatasetBase,
    "sim_deblur_gauss_k7": SimDatasetBase,
    "sim_deblur_gauss_k9": SimDatasetBase,
    "sim_deblur_motion_k5": SimDatasetBase,
    "sim_deblur_motion_k7": SimDatasetBase,
    "sim_deblur_motion_k9": SimDatasetBase,
    "sim_deblur_motion_k11": SimDatasetBase,
    "sim_sr_bicubic_x2": SimDatasetBase,
    "sim_sr_bicubic_x4": SimDatasetBase,
    "sim_sr_bicubic_x8": SimDatasetBase,
    # Visualization test split
    "sim_test_viz": SimDatasetBase,
    # Isolated source lists (internal experiment configs)
    "sim_denoise_gauss_005_iso": SimDatasetBase,
    "sim_deblur_motion_k5_iso": SimDatasetBase,
    "sim_destripe_nr_fixed_weak_iso": SimDatasetBase,
    "sim_destripe_nr_fixed_alt_weak_iso": SimDatasetBase,
    "sim_destripe_nr_fixed_alt_medium_iso": SimDatasetBase,
    "sim_destripe_nr_fixed_alt_strong_iso": SimDatasetBase,
    "sim_linear_stretch_iso": SimDatasetBase,
    "sim_equalize_iso": SimDatasetBase,
    "sim_brightness_increase_weak_iso": BrightnessSimDataset,
    # Pansharpening (NBU)
    "pansharp_gaofen1": Gaofen1Dataset,
    "pansharp_ikonos": IkonosDataset,
    "pansharp_quickbird": QuickbirdDataset,
    "pansharp_worldview2": Worldview2Dataset,
    "pansharp_worldview3": Worldview3Dataset,
    "pansharp_worldview4": Worldview4Dataset,
    # Pansharpening (PanCollection)
    "pansharp_pc_gf2": PcGf2Dataset,
    "pansharp_pc_qb": PcQbDataset,
    "pansharp_pc_wv3": PcWv3Dataset,
    "pansharp_pc_wv2": PcWv2Dataset,
    # STF
    "stf_cia": CiaDataset,
    "stf_ahb": AhbDataset,
    "stf_daxing": DaxingDataset,
    "stf_lgc": LgcDataset,
    "stf_tianjin": TianjinDataset,
    # Noise / SAR
    "noise_sar_filter": SarFilterDataset,
    "noise_sar_despeckle": SarDespeckleDataset,
}


def build_dataset(name: str, **kwargs) -> Union[DatasetBase, SimDatasetBase]:
    """Construct a dataset; SimDatasetBase subclasses get .name assigned."""
    assert name in DATASET_REGISTRY, (
        f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
    )
    cls = DATASET_REGISTRY[name]
    ds = cls(**kwargs)
    if isinstance(ds, SimDatasetBase):
        ds.name = name
    return ds
