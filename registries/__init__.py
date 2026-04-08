"""
Registry package index.

Each registry lives in its own module to avoid heavy imports and cycles.
Import explicitly from submodules, for example:

    from registries.dataset_registry import DATASET_REGISTRY, build_dataset
    from registries.model_registry import MODEL_REGISTRY, build_model
    from registries.finetune_registry import FINETUNE_REGISTRY, build_finetune
    from registries.encoder_registry import TEXT_ENCODER_REGISTRY, IMAGE_ENCODER_REGISTRY
    from registries.encoder_registry import build_text_encoder, build_image_encoder
    from registries.algo_registry import ROUTING_REGISTRY, build_router
"""
