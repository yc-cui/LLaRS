import torch
import torch.nn as nn
import torch.nn.functional as F

from ..moe import ConvMoE
from ..loss_collector import call_moe_with_auto_loss


class FeedForward(nn.Module):

    def __init__(self, dim, text_dim, img_dim, ffn_expansion_factor, bias, num_experts=4):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = ConvMoE(
            dim, text_dim, img_dim, hidden_features * 2,
            kernel_size=1, bias=bias, num_experts=num_experts
        )
        self.dwconv = ConvMoE(
            hidden_features * 2, text_dim, img_dim, hidden_features * 2,
            kernel_size=3, stride=1, padding=1, groups=hidden_features * 2,
            bias=bias, num_experts=num_experts
        )
        self.project_out = ConvMoE(
            hidden_features, text_dim, img_dim, dim,
            kernel_size=1, bias=bias, num_experts=num_experts
        )

    def forward(self, x, text_embd, img_embd):
        x = call_moe_with_auto_loss(self.project_in, x, text_embd, img_embd)
        x = call_moe_with_auto_loss(self.dwconv, x, text_embd, img_embd)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = call_moe_with_auto_loss(self.project_out, x, text_embd, img_embd)
        return x
