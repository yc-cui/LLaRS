"""
Project-wide constants.
"""
from pathlib import Path

# LLaRS_github repository root (directory containing this file).
PROJECT_ROOT = Path(__file__).resolve().parent

# On-disk root for image/binary paths stored as relative strings in *_dataset.json.
DATA_ROOT = Path("/data/LLaRS1M")


def resolve_project_relative_path(path: str | Path) -> Path:
    """Resolve paths relative to PROJECT_ROOT (e.g. meta ``path`` to list JSON)."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def resolve_data_file_path(path: str | Path) -> Path:
    """Resolve sample file paths relative to DATA_ROOT."""
    p = Path(path)
    if p.is_absolute():
        return p
    return DATA_ROOT / p


# TRAIN_SIZE = (64, 64)
TRAIN_SIZE = (128, 128)
VALID_SIZE = (256, 256)
TEST_MAX_SIZE = (2048, 2048)
MAX_CHANS = 20

DEFAULT_PROMPT_FILE = "data_utils/prompts/all_prompts.json"

TEXT_DEG_TYPE_CLASS = {
    "cloud": 0,
    "cloud_sar": 1,
    "haze": 2,
    # Keys follow task-style names aligned with prompt files
    "denoise": 3,
    # SR: real datasets use sr; simulated tiers use sr_weak / sr_medium / sr_strong
    "sr": 4,
    "pansharp": 5,
    "stf": 6,
    "blur": 7,
    "destripe": 8,
    "equal": 10,
    "linear": 11,
    # SR tiers (simulated super-resolution only)
    "sr_weak": 15,
    "sr_medium": 16,
    "sr_strong": 17,
    # Brightness subtypes share one id (no coarse brightness bucket)
    "brightness_increase_weak": 20,
    "brightness_increase_medium": 20,
    "brightness_increase_strong": 20,
    "brightness_decrease_weak": 20,
    "brightness_decrease_medium": 20,
    "brightness_decrease_strong": 20,
}

# CrossEntropy over TEXT_DEG_TYPE_CLASS ids (max id 20 -> 21 logits).
ROUTE_CLS_NUM_CLASSES = max(TEXT_DEG_TYPE_CLASS.values()) + 1

DEG_TYPE_ID_TO_NAME = {
    0: "cloud", 1: "cloud_sar", 2: "haze", 3: "denoise",
    4: "sr", 5: "pansharp", 6: "stf", 7: "blur",
    8: "destripe", 10: "histeq", 11: "linstretch",
    15: "sr_weak", 16: "sr_medium", 17: "sr_strong",
    20: "brightness",
}

IMG_DEG_TYPE_CLASS = {
    # Image degradation label space (img_deg_type at training time).
    # Use coarse classes the model should distinguish (e.g. denoise weak/medium/strong).
    # Do not add raw dataset keys; map dataset-specific configs onto these coarse ids.
    #
    # Real-data coarse types (aligned with TEXT_DEG_TYPE_CLASS where needed)
    "cloud": 0,
    "cloud_sar": 1,
    "haze": 2,
    "noise": 3,
    "pansharp": 5,
    "stf": 6,
    "blur": 7,
    "stripe": 8,

    # Inputs that are visually clean / near-original
    "raw": 900,

    # Brightness simulation subclasses (legacy ids; training maps to raw)
    "brightness_increase_weak": 100,
    "brightness_increase_medium": 101,
    "brightness_increase_strong": 102,
    "brightness_decrease_weak": 103,
    "brightness_decrease_medium": 104,
    "brightness_decrease_strong": 105,

    # Destriping simulation subclasses
    "destripe": 8,         # shares coarse bucket with stripe
    "destripe_weak": 810,
    "destripe_medium": 811,
    "destripe_strong": 812,

    # Noise simulation subclasses (weak / medium / strong)
    # Fixed ids:
    #   - denoise_weak:   200
    #   - denoise_medium: 201
    #   - denoise_strong: 202
    "denoise_weak": 200,
    "denoise_medium": 201,
    "denoise_strong": 202,
    # Optional mixed-noise bucket (sim_denoise -> treat as medium)
    "denoise": 201,

    # Blur simulation subclasses (weak / medium / strong)
    # Fixed ids:
    #   - deblur_weak:   300
    #   - deblur_medium: 301
    #   - deblur_strong: 302
    "deblur_weak": 300,
    "deblur_medium": 301,
    "deblur_strong": 302,
    # Optional mixed-blur bucket (sim_deblur -> treat as medium)
    "deblur": 301,


    # Linear stretch (training maps img_deg_type to raw)
    "linear_stretch": 500,
}

# ========================================================
# Dataset entries (paths relative to project root)
#   - Real: meta_file + text_deg_type + img_deg_type
#   - Simulated: meta_file + sim_ops + text_deg_type + img_deg_type
# ========================================================

DATASET_CONFIGS = {
    # ===== SR =====
    # Text task sr; image label raw (input is native low-res)
    "sr_oli2msi": {"meta_file": "data_utils/dataset_files/sr/oli2msi_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_alsace": {"meta_file": "data_utils/dataset_files/sr/sen2venus_alsace_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_anji": {"meta_file": "data_utils/dataset_files/sr/sen2venus_anji_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_arm": {"meta_file": "data_utils/dataset_files/sr/sen2venus_arm_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_atto": {"meta_file": "data_utils/dataset_files/sr/sen2venus_atto_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_bambenw2": {"meta_file": "data_utils/dataset_files/sr/sen2venus_bambenw2_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_benga": {"meta_file": "data_utils/dataset_files/sr/sen2venus_benga_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_es_ic3xg": {"meta_file": "data_utils/dataset_files/sr/sen2venus_es_ic3xg_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_es_ltera": {"meta_file": "data_utils/dataset_files/sr/sen2venus_es_ltera_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_esgisb_1": {"meta_file": "data_utils/dataset_files/sr/sen2venus_esgisb_1_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_esgisb_2": {"meta_file": "data_utils/dataset_files/sr/sen2venus_esgisb_2_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_esgisb_3": {"meta_file": "data_utils/dataset_files/sr/sen2venus_esgisb_3_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_estuamar": {"meta_file": "data_utils/dataset_files/sr/sen2venus_estuamar_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_fgmanaus": {"meta_file": "data_utils/dataset_files/sr/sen2venus_fgmanaus_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_fr_bil": {"meta_file": "data_utils/dataset_files/sr/sen2venus_fr_bil_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_fr_lam": {"meta_file": "data_utils/dataset_files/sr/sen2venus_fr_lam_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_fr_lq1": {"meta_file": "data_utils/dataset_files/sr/sen2venus_fr_lq1_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_jam2018": {"meta_file": "data_utils/dataset_files/sr/sen2venus_jam2018_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_k34_amaz": {"meta_file": "data_utils/dataset_files/sr/sen2venus_k34_amaz_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_kudaliar": {"meta_file": "data_utils/dataset_files/sr/sen2venus_kudaliar_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_lerida_1": {"meta_file": "data_utils/dataset_files/sr/sen2venus_lerida_1_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_mad_ambo": {"meta_file": "data_utils/dataset_files/sr/sen2venus_mad_ambo_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_naryn": {"meta_file": "data_utils/dataset_files/sr/sen2venus_naryn_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_so1": {"meta_file": "data_utils/dataset_files/sr/sen2venus_so1_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_so2": {"meta_file": "data_utils/dataset_files/sr/sen2venus_so2_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_sudoue_2": {"meta_file": "data_utils/dataset_files/sr/sen2venus_sudoue_2_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_sudoue_3": {"meta_file": "data_utils/dataset_files/sr/sen2venus_sudoue_3_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_sudoue_4": {"meta_file": "data_utils/dataset_files/sr/sen2venus_sudoue_4_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_sudoue_5": {"meta_file": "data_utils/dataset_files/sr/sen2venus_sudoue_5_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    "sr_sen2venus_sudoue_6": {"meta_file": "data_utils/dataset_files/sr/sen2venus_sudoue_6_dataset_meta.json", "text_deg_type": "sr", "img_deg_type": "raw"},
    # ===== Cloud removal =====
    "cloud_cuhk_cr1": {"meta_file": "data_utils/dataset_files/cloud/cuhk_cr1_dataset_meta.json", "text_deg_type": "cloud", "img_deg_type": "cloud"},
    "cloud_cuhk_cr2": {"meta_file": "data_utils/dataset_files/cloud/cuhk_cr2_dataset_meta.json", "text_deg_type": "cloud", "img_deg_type": "cloud"},
    "cloud_rice1": {"meta_file": "data_utils/dataset_files/cloud/rice1_dataset_meta.json", "text_deg_type": "cloud", "img_deg_type": "cloud"},
    "cloud_rice2": {"meta_file": "data_utils/dataset_files/cloud/rice2_dataset_meta.json", "text_deg_type": "cloud", "img_deg_type": "cloud"},
    # ===== Cloud removal (SAR) =====
    "cloud_sar_sen12mscr_spring": {"meta_file": "data_utils/dataset_files/cloud_sar/sen12mscr_spring_dataset_meta.json", "text_deg_type": "cloud_sar", "img_deg_type": "cloud_sar"},
    "cloud_sar_sen12mscr_summer": {"meta_file": "data_utils/dataset_files/cloud_sar/sen12mscr_summer_dataset_meta.json", "text_deg_type": "cloud_sar", "img_deg_type": "cloud_sar"},
    "cloud_sar_sen12mscr_fall":   {"meta_file": "data_utils/dataset_files/cloud_sar/sen12mscr_fall_dataset_meta.json", "text_deg_type": "cloud_sar", "img_deg_type": "cloud_sar"},
    "cloud_sar_sen12mscr_winter": {"meta_file": "data_utils/dataset_files/cloud_sar/sen12mscr_winter_dataset_meta.json", "text_deg_type": "cloud_sar", "img_deg_type": "cloud_sar"},
    # ===== Haze removal =====
    "haze_haze1k_thin": {"meta_file": "data_utils/dataset_files/cloud/haze1k_thin_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_haze1k_moderate": {"meta_file": "data_utils/dataset_files/cloud/haze1k_moderate_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_haze1k_thick": {"meta_file": "data_utils/dataset_files/cloud/haze1k_thick_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_rrshid_thin": {"meta_file": "data_utils/dataset_files/cloud/rrshid_thin_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_rrshid_moderate": {"meta_file": "data_utils/dataset_files/cloud/rrshid_moderate_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_rrshid_thick": {"meta_file": "data_utils/dataset_files/cloud/rrshid_thick_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_rsid": {"meta_file": "data_utils/dataset_files/cloud/rsid_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_dhid": {"meta_file": "data_utils/dataset_files/cloud/dhid_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    "haze_lhid": {"meta_file": "data_utils/dataset_files/cloud/lhid_dataset_meta.json", "text_deg_type": "haze", "img_deg_type": "haze"},
    # ===== Pansharpening (NBU) =====
    "pansharp_gaofen1": {"meta_file": "data_utils/dataset_files/pansharp/gaofen1_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_ikonos": {"meta_file": "data_utils/dataset_files/pansharp/ikonos_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_quickbird": {"meta_file": "data_utils/dataset_files/pansharp/quickbird_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_worldview2": {"meta_file": "data_utils/dataset_files/pansharp/worldview2_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_worldview3": {"meta_file": "data_utils/dataset_files/pansharp/worldview3_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_worldview4": {"meta_file": "data_utils/dataset_files/pansharp/worldview4_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    # ===== Pansharpening (PanCollection) =====
    "pansharp_pc_gf2": {"meta_file": "data_utils/dataset_files/pansharp/pc_gf2_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_pc_qb": {"meta_file": "data_utils/dataset_files/pansharp/pc_qb_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_pc_wv3": {"meta_file": "data_utils/dataset_files/pansharp/pc_wv3_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    "pansharp_pc_wv2": {"meta_file": "data_utils/dataset_files/pansharp/pc_wv2_dataset_meta.json", "text_deg_type": "pansharp", "img_deg_type": "pansharp"},
    # ===== STF =====
    "stf_cia": {"meta_file": "data_utils/dataset_files/stf/cia_dataset_meta.json", "text_deg_type": "stf", "img_deg_type": "stf"},
    "stf_ahb": {"meta_file": "data_utils/dataset_files/stf/ahb_dataset_meta.json", "text_deg_type": "stf", "img_deg_type": "stf"},
    "stf_daxing": {"meta_file": "data_utils/dataset_files/stf/daxing_dataset_meta.json", "text_deg_type": "stf", "img_deg_type": "stf"},
    "stf_lgc": {"meta_file": "data_utils/dataset_files/stf/lgc_dataset_meta.json", "text_deg_type": "stf", "img_deg_type": "stf"},
    "stf_tianjin": {"meta_file": "data_utils/dataset_files/stf/tianjin_dataset_meta.json", "text_deg_type": "stf", "img_deg_type": "stf"},
    # ===== Noise / SAR =====
    "noise_sar_filter": {"meta_file": "data_utils/dataset_files/noise/sar_filter_dataset_meta.json", "text_deg_type": "denoise", "img_deg_type": "noise"},
    "noise_sar_despeckle": {"meta_file": "data_utils/dataset_files/noise/sar_despeckle_dataset_meta.json", "text_deg_type": "denoise", "img_deg_type": "noise"},
    # ===== Simulated (training — noise / blur / SR / stripes) =====
    # Noise: random strength plus weak/medium/strong tiers
    "sim_denoise":         {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise",         "text_deg_type": "denoise", "img_deg_type": "denoise"},
    "sim_denoise_weak":    {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_weak",    "text_deg_type": "denoise", "img_deg_type": "denoise_weak"},
    "sim_denoise_medium":  {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_medium",  "text_deg_type": "denoise", "img_deg_type": "denoise_medium"},
    "sim_denoise_strong":  {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_strong",  "text_deg_type": "denoise", "img_deg_type": "denoise_strong"},

    # Blur: random strength plus weak/medium/strong tiers
    "sim_deblur":          {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur",          "text_deg_type": "blur", "img_deg_type": "deblur"},
    "sim_deblur_weak":     {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_weak",     "text_deg_type": "blur", "img_deg_type": "deblur_weak"},
    "sim_deblur_medium":   {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_medium",   "text_deg_type": "blur", "img_deg_type": "deblur_medium"},
    "sim_deblur_strong":   {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_strong",   "text_deg_type": "blur", "img_deg_type": "deblur_strong"},

    # Sim SR: text tiers sr_weak/medium/strong; image label raw
    "sim_sr":              {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "sr",              "text_deg_type": "sr",        "img_deg_type": "raw"},
    "sim_sr_weak":         {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "sr_weak",         "text_deg_type": "sr_weak",   "img_deg_type": "raw"},
    "sim_sr_medium":       {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "sr_medium",       "text_deg_type": "sr_medium", "img_deg_type": "raw"},
    "sim_sr_strong":       {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "sr_strong",       "text_deg_type": "sr_strong", "img_deg_type": "raw"},

    # Stripes: random strength plus weak/medium/strong tiers
    "sim_destripe":        {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe",        "text_deg_type": "destripe", "img_deg_type": "destripe"},
    "sim_destripe_weak":   {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_weak",   "text_deg_type": "destripe", "img_deg_type": "destripe_weak"},
    "sim_destripe_medium": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_medium", "text_deg_type": "destripe", "img_deg_type": "destripe_medium"},
    "sim_destripe_strong": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_strong", "text_deg_type": "destripe", "img_deg_type": "destripe_strong"},
    # Fixed-parameter non-uniform rotation stripes
    "sim_destripe_nr_fixed_weak":   {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_nr_fixed_weak",   "text_deg_type": "destripe", "img_deg_type": "destripe_weak"},
    "sim_destripe_nr_fixed_medium": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_nr_fixed_medium", "text_deg_type": "destripe", "img_deg_type": "destripe_medium"},
    "sim_destripe_nr_fixed_strong": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_nr_fixed_strong", "text_deg_type": "destripe", "img_deg_type": "destripe_strong"},
    # Fixed-parameter stripes with alternating intensity
    "sim_destripe_nr_fixed_alt_weak":   {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_nr_fixed_alt_weak",   "text_deg_type": "destripe", "img_deg_type": "destripe_weak"},
    "sim_destripe_nr_fixed_alt_medium": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_nr_fixed_alt_medium", "text_deg_type": "destripe", "img_deg_type": "destripe_medium"},
    "sim_destripe_nr_fixed_alt_strong": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "destripe_nr_fixed_alt_strong", "text_deg_type": "destripe", "img_deg_type": "destripe_strong"},
    # ===== Simulated brightness (input is clean; img_deg_type raw) =====
    # Text keys stay fine-grained; TEXT_DEG_TYPE_CLASS maps them to one brightness id.
    "sim_brightness_increase_weak":   {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "brightness_increase_weak",   "text_deg_type": "brightness_increase_weak",   "img_deg_type": "raw"},
    "sim_brightness_increase_medium": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "brightness_increase_medium", "text_deg_type": "brightness_increase_medium", "img_deg_type": "raw"},
    "sim_brightness_increase_strong": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "brightness_increase_strong", "text_deg_type": "brightness_increase_strong", "img_deg_type": "raw"},
    "sim_brightness_decrease_weak":   {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "brightness_decrease_weak",   "text_deg_type": "brightness_decrease_weak",   "img_deg_type": "raw"},
    "sim_brightness_decrease_medium": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "brightness_decrease_medium", "text_deg_type": "brightness_decrease_medium", "img_deg_type": "raw"},
    "sim_brightness_decrease_strong": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "brightness_decrease_strong", "text_deg_type": "brightness_decrease_strong", "img_deg_type": "raw"},

    # ===== Simulated histogram / linear stretch (clean input) =====
    "sim_equalize":      {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "equalize",       "text_deg_type": "equal",  "img_deg_type": "raw"},
    "sim_linear_stretch": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "linear_stretch", "text_deg_type": "linear", "img_deg_type": "raw"},
    # ===== Simulated (evaluation — Gaussian noise)
    # Dataset names only; img_deg_type stays weak/medium/strong coarse classes.
    "sim_denoise_gauss_001": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_gauss_001", "text_deg_type": "denoise", "img_deg_type": "denoise_weak"},
    "sim_denoise_gauss_005": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_gauss_005", "text_deg_type": "denoise", "img_deg_type": "denoise_medium"},
    "sim_denoise_gauss_010": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_gauss_010", "text_deg_type": "denoise", "img_deg_type": "denoise_medium"},
    "sim_denoise_gauss_025": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_gauss_025", "text_deg_type": "denoise", "img_deg_type": "denoise_strong"},
    "sim_denoise_gauss_050": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_gauss_050", "text_deg_type": "denoise", "img_deg_type": "denoise_strong"},
    # ===== Simulated (evaluation — salt-pepper noise) =====
    "sim_denoise_sp_001": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_sp_001", "text_deg_type": "denoise", "img_deg_type": "denoise_weak"},
    "sim_denoise_sp_005": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_sp_005", "text_deg_type": "denoise", "img_deg_type": "denoise_medium"},
    "sim_denoise_sp_010": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "denoise_sp_010", "text_deg_type": "denoise", "img_deg_type": "denoise_strong"},
    # ===== Simulated (evaluation — Gaussian blur) =====
    "sim_deblur_gauss_k3": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_gauss_k3", "text_deg_type": "blur", "img_deg_type": "deblur_weak"},
    "sim_deblur_gauss_k5": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_gauss_k5", "text_deg_type": "blur", "img_deg_type": "deblur_medium"},
    "sim_deblur_gauss_k7": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_gauss_k7", "text_deg_type": "blur", "img_deg_type": "deblur_strong"},
    "sim_deblur_gauss_k9": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_gauss_k9", "text_deg_type": "blur", "img_deg_type": "deblur_strong"},
    # ===== Simulated (evaluation — motion blur) =====
    "sim_deblur_motion_k5":  {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_motion_k5",  "text_deg_type": "blur", "img_deg_type": "deblur_weak"},
    "sim_deblur_motion_k7":  {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_motion_k7",  "text_deg_type": "blur", "img_deg_type": "deblur_medium"},
    "sim_deblur_motion_k9":  {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_motion_k9",  "text_deg_type": "blur", "img_deg_type": "deblur_medium"},
    "sim_deblur_motion_k11": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "deblur_motion_k11", "text_deg_type": "blur", "img_deg_type": "deblur_strong"},
    # ===== Simulated (evaluation — bicubic downsampling) =====
    # Scale maps to weak/medium/strong text tiers; image label raw.
    "sim_sr_bicubic_x2": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "sr_bicubic_x2", "text_deg_type": "sr_weak",   "img_deg_type": "raw"},
    "sim_sr_bicubic_x4": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "sr_bicubic_x4", "text_deg_type": "sr_medium", "img_deg_type": "raw"},
    "sim_sr_bicubic_x8": {"meta_file": "data_utils/dataset_files/sim/sim_all_clean_sources.json", "sim_ops": "sr_bicubic_x8", "text_deg_type": "sr_strong", "img_deg_type": "raw"},
    # ===== Simulated (visualization — all real sources) =====
    "sim_test_viz": {"meta_file": "data_utils/dataset_files/sim/sim_test_viz_sources.json", "sim_ops": "denoise_weak", "text_deg_type": "denoise", "img_deg_type": "denoise_weak"},
    # ===== Simulated (disjoint sources per task) =====
    "sim_denoise_gauss_005_iso":       {"meta_file": "data_utils/dataset_files/sim/sim_iso_denoise.json",    "sim_ops": "denoise_gauss_005",        "text_deg_type": "denoise",                  "img_deg_type": "denoise_medium"},
    "sim_deblur_motion_k5_iso":        {"meta_file": "data_utils/dataset_files/sim/sim_iso_deblur.json",     "sim_ops": "deblur_motion_k5",         "text_deg_type": "blur",                     "img_deg_type": "deblur_weak"},
    "sim_destripe_nr_fixed_weak_iso":      {"meta_file": "data_utils/dataset_files/sim/sim_iso_destripe.json",   "sim_ops": "destripe_nr_fixed_weak",       "text_deg_type": "destripe",                 "img_deg_type": "destripe_weak"},
    "sim_destripe_nr_fixed_alt_weak_iso":    {"meta_file": "data_utils/dataset_files/sim/sim_iso_destripe.json",   "sim_ops": "destripe_nr_fixed_alt_weak",   "text_deg_type": "destripe",                 "img_deg_type": "destripe_weak"},
    "sim_destripe_nr_fixed_alt_medium_iso":  {"meta_file": "data_utils/dataset_files/sim/sim_iso_destripe.json",   "sim_ops": "destripe_nr_fixed_alt_medium", "text_deg_type": "destripe",                 "img_deg_type": "destripe_medium"},
    "sim_destripe_nr_fixed_alt_strong_iso":  {"meta_file": "data_utils/dataset_files/sim/sim_iso_destripe.json",   "sim_ops": "destripe_nr_fixed_alt_strong", "text_deg_type": "destripe",                 "img_deg_type": "destripe_strong"},
    "sim_linear_stretch_iso":          {"meta_file": "data_utils/dataset_files/sim/sim_iso_linstretch.json", "sim_ops": "linear_stretch",           "text_deg_type": "linear",                   "img_deg_type": "raw"},
    "sim_equalize_iso":                {"meta_file": "data_utils/dataset_files/sim/sim_iso_equalize.json",   "sim_ops": "equalize",                 "text_deg_type": "equal",                    "img_deg_type": "raw"},
    "sim_brightness_increase_weak_iso":{"meta_file": "data_utils/dataset_files/sim/sim_iso_brightness.json", "sim_ops": "brightness_increase_weak", "text_deg_type": "brightness_increase_weak", "img_deg_type": "raw"},
}
