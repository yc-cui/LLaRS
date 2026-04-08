import numpy as np
import torch
import torch.nn as nn

from ..moe import ConvMoE, MoCEMoE
from ..loss_collector import LoadLossCollector, RouteClsCollector, call_moe_with_auto_loss

from .blocks import MoTBlock, OverlapPatchEmbed, Downsample, Upsample, MultiInputSequential


class LLaRS(nn.Module):

    def __init__(
        self,
        inp_channels=16,
        out_channels=16,
        dim=32,
        num_blocks=[4, 4, 2, 2],
        num_refinement_blocks=2,
        heads=[2, 2, 2, 2],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False,
        text_dim=768,
        img_dim=768,
        num_experts=[4, 4, 4, 4],
        refinement_experts=4,
        num_experts_ICB=4,
        num_experts_pre=4,
    ):
        super(LLaRS, self).__init__()

        self.scale = (
            np.sum([num_blocks[i] * 6 * 2 for i in range(len(num_blocks))]) +
            num_refinement_blocks * 6 + 4 + 1 + 1
        )
        print(f"scale: {self.scale}")

        self.patch_embed = OverlapPatchEmbed(
            inp_channels, dim, text_dim, img_dim, num_experts=num_experts_pre
        )

        self.encoder_level1 = MultiInputSequential(*[
            MoTBlock(
                dim=dim, text_dim=text_dim, img_dim=img_dim, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=num_experts[0],
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = MultiInputSequential(*[
            MoTBlock(
                dim=int(dim * 2**1), text_dim=text_dim, img_dim=img_dim, num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=num_experts[1],
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = MultiInputSequential(*[
            MoTBlock(
                dim=int(dim * 2**2), text_dim=text_dim, img_dim=img_dim, num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=num_experts[2],
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = MultiInputSequential(*[
            MoTBlock(
                dim=int(dim * 2**3), text_dim=text_dim, img_dim=img_dim, num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=num_experts[3],
            ) for _ in range(num_blocks[3])
        ])

        self.enc_cond1 = MoCEMoE(
            in_channels=int(dim * 2**0), text_dim=text_dim, img_dim=img_dim,
            out_channels=int(dim * 2**0), num_experts=num_experts_ICB
        )
        self.enc_cond2 = MoCEMoE(
            in_channels=int(dim * 2**1), text_dim=text_dim, img_dim=img_dim,
            out_channels=int(dim * 2**1), num_experts=num_experts_ICB
        )
        self.enc_cond3 = MoCEMoE(
            in_channels=int(dim * 2**2), text_dim=text_dim, img_dim=img_dim,
            out_channels=int(dim * 2**2), num_experts=num_experts_ICB
        )
        self.latent_cond = MoCEMoE(
            in_channels=int(dim * 2**3), text_dim=text_dim, img_dim=img_dim,
            out_channels=int(dim * 2**3), num_experts=num_experts_ICB
        )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = ConvMoE(
            int(dim * 2**3), text_dim=text_dim, img_dim=img_dim,
            out_channels=int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.decoder_level3 = MultiInputSequential(*[
            MoTBlock(
                dim=int(dim * 2**2), text_dim=text_dim, img_dim=img_dim, num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=num_experts[2],
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = ConvMoE(
            int(dim * 2**2), text_dim=text_dim, img_dim=img_dim,
            out_channels=int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.decoder_level2 = MultiInputSequential(*[
            MoTBlock(
                dim=int(dim * 2**1), text_dim=text_dim, img_dim=img_dim, num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=num_experts[1],
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2**1))
        self.decoder_level1 = MultiInputSequential(*[
            MoTBlock(
                dim=int(dim * 2**1), text_dim=text_dim, img_dim=img_dim, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=num_experts[0],
            ) for _ in range(num_blocks[0])
        ])

        self.refinement = MultiInputSequential(*[
            MoTBlock(
                dim=int(dim * 2**1), text_dim=text_dim, img_dim=img_dim, num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, num_experts=refinement_experts,
            ) for _ in range(num_refinement_blocks)
        ])

        self.output = ConvMoE(
            int(dim * 2**1), text_dim=text_dim, img_dim=img_dim,
            out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, inp_img, text_embd, img_embd, text_deg_labels=None):
        with RouteClsCollector() as route_collector:
            with LoadLossCollector() as collector:
                inp_enc_level1 = self.patch_embed(inp_img, text_embd, img_embd)

                out_enc_level1 = self.encoder_level1(inp_enc_level1, text_embd, img_embd)
                out_enc_level1 = call_moe_with_auto_loss(self.enc_cond1, out_enc_level1, text_embd, img_embd)

                inp_enc_level2 = self.down1_2(out_enc_level1)
                out_enc_level2 = self.encoder_level2(inp_enc_level2, text_embd, img_embd)
                out_enc_level2 = call_moe_with_auto_loss(self.enc_cond2, out_enc_level2, text_embd, img_embd)

                inp_enc_level3 = self.down2_3(out_enc_level2)
                out_enc_level3 = self.encoder_level3(inp_enc_level3, text_embd, img_embd)
                out_enc_level3 = call_moe_with_auto_loss(self.enc_cond3, out_enc_level3, text_embd, img_embd)

                inp_enc_level4 = self.down3_4(out_enc_level3)
                latent = self.latent(inp_enc_level4, text_embd, img_embd)
                latent = call_moe_with_auto_loss(self.latent_cond, latent, text_embd, img_embd)

                inp_dec_level3 = self.up4_3(latent)
                inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
                inp_dec_level3 = call_moe_with_auto_loss(self.reduce_chan_level3, inp_dec_level3, text_embd, img_embd)
                out_dec_level3 = self.decoder_level3(inp_dec_level3, text_embd, img_embd)

                inp_dec_level2 = self.up3_2(out_dec_level3)
                inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
                inp_dec_level2 = call_moe_with_auto_loss(self.reduce_chan_level2, inp_dec_level2, text_embd, img_embd)
                out_dec_level2 = self.decoder_level2(inp_dec_level2, text_embd, img_embd)

                inp_dec_level1 = self.up2_1(out_dec_level2)
                inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
                out_dec_level1 = self.decoder_level1(inp_dec_level1, text_embd, img_embd)

                out_dec_level1 = self.refinement(out_dec_level1, text_embd, img_embd)

                out_dec_level1 = call_moe_with_auto_loss(self.output, out_dec_level1, text_embd, img_embd)

                total_load_loss = collector.get_total_loss(device=inp_img.device)

            route_cls_loss = route_collector.compute_route_cls_loss(
                text_deg_labels, inp_img.device, inp_img.dtype,
            )

        return out_dec_level1, total_load_loss / self.scale, route_cls_loss
