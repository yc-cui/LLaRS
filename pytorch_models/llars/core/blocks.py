import torch.nn as nn

from ..moe import ConvMoE
from ..loss_collector import call_moe_with_auto_loss

from .normalization import LayerNorm
from .attention import Attention
from .feedforward import FeedForward


class MoTBlock(nn.Module):
    """Transformer-style block with mixture-of-tokens (MoT) attention path."""

    def __init__(self, dim, text_dim, img_dim, num_heads, ffn_expansion_factor,
                 bias, LayerNorm_type, num_experts=4):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, text_dim, img_dim, num_heads, bias, num_experts)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, text_dim, img_dim, ffn_expansion_factor, bias, num_experts)

    def forward(self, x, text_embd, img_embd):
        attn_out = self.attn(self.norm1(x), text_embd, img_embd)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x), text_embd, img_embd)
        x = x + ffn_out
        return x


class OverlapPatchEmbed(nn.Module):

    def __init__(self, in_channels=3, out_channels=48, text_dim=768, img_dim=768,
                 bias=False, num_experts=8):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = ConvMoE(
            in_channels, text_dim, img_dim, out_channels,
            kernel_size=3, stride=1, padding=1, bias=bias, num_experts=num_experts
        )

    def forward(self, x, text_embd, img_embd):
        x = call_moe_with_auto_loss(self.proj, x, text_embd, img_embd)
        return x


class Downsample(nn.Module):

    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):

    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class MultiInputSequential(nn.Module):

    def __init__(self, *modules):
        super(MultiInputSequential, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x, text_embd, img_embd):
        for module in self.modules_list:
            x = module(x, text_embd, img_embd)
        return x
