import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ..nafnet_utils import NAFBlock
from .dispatcher import SparseDispatcher
from .routing import RouteFunc


class MoCEExpert(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x.squeeze(-1).squeeze(-1)


class MoCEMoE(nn.Module):

    def __init__(self, in_channels, text_dim, img_dim, out_channels, num_experts=4, noisy_gating=True, use_experts=2):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.k = use_experts
        self.block = NAFBlock(in_channels)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)
        if num_experts == 1:
            self.is_moe = False
            self.conv = MoCEExpert(in_channels)
        else:
            self.route_func = RouteFunc(in_channels, text_dim, img_dim)
            self.experts = nn.ModuleList([MoCEExpert(in_channels) for _ in range(self.num_experts)])
            reduced_dim = 64
            self.w_gate = nn.Parameter(torch.randn(reduced_dim, num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(reduced_dim, num_experts), requires_grad=True)
            self.is_moe = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        if self.is_moe:
            assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, dtype=top_k_gates.dtype, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, clean_logits

    def forward(self, x, text_embd, img_embd):

        x = self.block(x)
        if not self.is_moe:
            y = x * self.conv(x).unsqueeze(-1).unsqueeze(-1) * self.gamma + self.beta
            load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return y, load_balancing_loss

        B, C, H, W = x.shape

        x_gating = self.route_func(x, text_embd, img_embd)

        gates, load, clean_logits = self.noisy_top_k_gating(x_gating, self.training)
        importance = gates.sum(0)

        load_balancing_loss = self.cv_squared(importance) * B / self.num_experts + self.cv_squared(load) * B * self.k / self.num_experts

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs).unsqueeze(-1).unsqueeze(-1)

        y = x * y * self.gamma + self.beta

        return y, load_balancing_loss
