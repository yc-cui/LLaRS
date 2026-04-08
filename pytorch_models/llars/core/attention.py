import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..moe import ConvMoE, LoRAExpert1x1, RouteFunc
from ..loss_collector import call_moe_with_auto_loss, get_current_collector as _get_current_collector


class Attention(nn.Module):

    def __init__(self, dim, text_dim, img_dim, num_heads, bias, num_experts=4,
                 rank=16, use_experts=2, noisy_gating=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.k = use_experts

        self.qk = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
            groups=dim * 2, bias=bias
        )

        self.v = ConvMoE(
            dim, text_dim, img_dim, dim,
            kernel_size=3, stride=1, padding=1, groups=dim, bias=bias,
            num_experts=num_experts, noisy_gating=noisy_gating, use_experts=use_experts
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        if self.num_experts == 1:
            self.is_moe = False
        else:
            self.route_func = RouteFunc(dim, text_dim, img_dim)
            reduced_dim = 64
            self.w_gate = nn.Parameter(torch.randn(reduced_dim, num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(reduced_dim, num_experts), requires_grad=True)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
            self.lora_q = LoRAExpert1x1(dim, rank=rank, alpha=rank, num_experts=num_experts)
            self.lora_k = LoRAExpert1x1(dim, rank=rank, alpha=rank, num_experts=num_experts)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def noisy_top_k_gating(self, x_reduced, train: bool, noise_epsilon: float = 1e-2):
        clean_logits = x_reduced @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x_reduced @ self.w_noise
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = F.softmax(top_k_logits, dim=1)

        B, E = logits.shape
        zeros = torch.zeros(B, E, device=top_k_gates.device, dtype=top_k_gates.dtype, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        load = (gates > 0).sum(0)
        return gates, load

    def cv_squared(self, x, eps=1e-10):
        if x.numel() <= 1:
            return torch.tensor(0., device=x.device, dtype=x.dtype)
        m = x.float().mean()
        v = x.float().var(unbiased=False)
        return v / (m * m + eps)

    def forward(self, x, text_embd, img_embd):
        B, C, H, W = x.shape

        qk = self.qk_dwconv(self.qk(x))
        q, k = qk.chunk(2, dim=1)

        if self.num_experts != 1:
            x_reduced = self.route_func(x, text_embd, img_embd)
            gates, load_qk = self.noisy_top_k_gating(x_reduced, self.training)
            importance_qk = gates.sum(0)
            load_loss_qk = (
                self.cv_squared(importance_qk) * B / self.num_experts +
                self.cv_squared(load_qk) * B * self.k / self.num_experts
            )
        else:
            gates = None
            load_loss_qk = torch.tensor(0.0, device=x.device)

        if gates is not None:
            q = self.lora_q(q, gates)
            k = self.lora_k(k, gates)

        v = call_moe_with_auto_loss(self.v, x, text_embd, img_embd)

        collector = _get_current_collector()
        if collector is not None:
            collector.add_loss(load_loss_qk)

        q = rearrange(q, 'b (h c) hgt wdt -> b h c (hgt wdt)', h=self.num_heads)
        k = rearrange(k, 'b (h c) hgt wdt -> b h c (hgt wdt)', h=self.num_heads)
        v = rearrange(v, 'b (h c) hgt wdt -> b h c (hgt wdt)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'b h c (hgt wdt) -> b (h c) hgt wdt',
                       h=self.num_heads, hgt=H, wdt=W)
        out = self.project_out(out)

        return out
