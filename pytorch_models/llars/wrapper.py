from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn

from .core.model import LLaRS
from constants import MAX_CHANS
from utils.dwa import DynamicWeightAdjuster, compute_reconstruction_loss
from registries.encoder_registry import build_text_encoder, build_image_encoder


_DEFAULT_TEXT_ENCODER = {
    "name": "huggingface",
    "kwargs": {"model_name_or_path": "distilbert-base-uncased"},
}
_DEFAULT_IMAGE_ENCODER = {
    "name": "resnet18",
    "kwargs": {},
}


class LLaRSNet(nn.Module):

    def __init__(
        self,
        inp_channels: int = MAX_CHANS,
        out_channels: int = MAX_CHANS,
        dim: int = 32,
        num_blocks: list = [2, 4, 6, 8],
        num_refinement_blocks: int = 2,
        heads: list = [1, 2, 4, 8],
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
        LayerNorm_type: str = 'WithBias',
        dual_pixel_task: bool = False,
        text_dim: int = 768,
        img_dim: int = 768,
        num_experts: list = [4, 4, 4, 4],
        refinement_experts: int = 2,
        num_experts_ICB: int = 4,
        num_experts_pre: int = 4,
        text_encoder: Optional[Dict[str, Any]] = None,
        img_encoder: Optional[Dict[str, Any]] = None,
        text_model: str = None,
        dwa_cfg: Optional[Dict[str, Any]] = None,
        disable_prompt: bool = False,
    ):
        super().__init__()

        self.text_dim = text_dim
        self.img_dim = img_dim

        self.net = LLaRS(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task,
            text_dim=text_dim,
            img_dim=img_dim,
            num_experts=num_experts,
            refinement_experts=refinement_experts,
            num_experts_ICB=num_experts_ICB,
            num_experts_pre=num_experts_pre,
        )

        text_enc_cfg = text_encoder or _DEFAULT_TEXT_ENCODER
        if text_model is not None and text_encoder is None:
            text_enc_cfg = {
                "name": "huggingface",
                "kwargs": {"model_name_or_path": text_model},
            }
        self.text_encoder = build_text_encoder(
            name=text_enc_cfg["name"],
            target_dim=text_dim,
            **text_enc_cfg.get("kwargs", {}),
        )

        img_enc_cfg = img_encoder or _DEFAULT_IMAGE_ENCODER
        self.img_encoder = build_image_encoder(
            name=img_enc_cfg["name"],
            target_dim=img_dim,
            in_channels=inp_channels,
            **img_enc_cfg.get("kwargs", {}),
        )

        self.criterion = nn.L1Loss()

        self.dwa = None
        if dwa_cfg is not None:
            self.dwa = DynamicWeightAdjuster(**dwa_cfg)

        self.disable_prompt = disable_prompt

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        inp = batch["inp"]
        gt = batch["gt"]
        prompts = batch["prompt"]

        device = inp.device

        if self.disable_prompt:
            text_embd = torch.zeros(inp.shape[0], self.text_dim, device=device)
        else:
            text_embd = self.text_encoder(prompts, device)

        img_embd, _ = self.img_encoder(inp)

        text_deg_labels = batch["image_meta"].get("text_deg_type")
        pred, load_loss, route_cls_loss = self.net(
            inp, text_embd, img_embd, text_deg_labels=text_deg_labels,
        )
        pred = pred + inp

        num_channelss = batch["image_meta"]["num_channels"]
        loss_dict = compute_reconstruction_loss(
            pred, gt, num_channelss, batch, self.dwa,
            extra_losses={
                "load_loss": load_loss,
                "route_cls_loss": route_cls_loss,
            },
        )
        return pred, loss_dict
