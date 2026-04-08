import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertConv2d(nn.Module):
    """2D conv with per-expert kernels mixed by ``routing_weight`` (batch x num_experts)."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()
        x = x.view(1, -1, h, w)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv2d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output


class LoRAExpert1x1(nn.Module):
    """Low-rank 1x1 delta mixed per sample by expert ``gates``."""

    def __init__(self, channels: int, rank: int = 16, alpha: int = 16, num_experts: int = 4):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.num_experts = num_experts
        self.scale = alpha / float(rank)
        if self.num_experts != 1:
            self.A = nn.Parameter(torch.zeros(num_experts, rank, channels))
            self.B = nn.Parameter(torch.zeros(num_experts, channels, rank))

            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        if self.num_experts == 1:
            return x
        B, C, H, W = x.shape
        E = self.num_experts
        assert C == self.channels and gates.shape == (B, E)

        deltaW = torch.bmm(self.B, self.A) * self.scale

        deltaW_flat = deltaW.reshape(E, C * C)
        mixedW_flat = gates @ deltaW_flat
        mixedW = mixedW_flat.view(B * C, C, 1, 1)

        x_ = x.reshape(1, B * C, H, W)
        delta = F.conv2d(x_, weight=mixedW, bias=None, stride=1, padding=0, groups=B)
        delta = delta.view(B, C, H, W)

        return x + delta
