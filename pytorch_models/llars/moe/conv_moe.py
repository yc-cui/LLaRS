import sys
import os
from contextvars import ContextVar

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..loss_collector import get_batch_metadata
from .dispatcher import SparseDispatcher
from .routing import RouteFunc
from .expert_conv import ExpertConv2d

# ContextVar: batch metadata for NaN debug (see get_batch_metadata in loss_collector).
_batch_metadata: ContextVar = ContextVar('batch_metadata', default=None)


def _chk(name, t: torch.Tensor):
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    if not (has_nan or has_inf):
        return

    t_float = t.detach().float()
    finite = torch.isfinite(t_float)

    if finite.any():
        maxv = t_float[finite].max().item()
        minv = t_float[finite].min().item()
        maxabs = t_float[finite].abs().max().item()
    else:
        maxv = minv = maxabs = float("nan")

    metadata = _batch_metadata.get() if "_batch_metadata" in dir() else get_batch_metadata()

    print(f"[CHK] {name}: dtype={t.dtype}, shape={tuple(t.shape)}, "
          f"nan={has_nan}, inf={has_inf}, "
          f"min={minv:.6g}, max={maxv:.6g}, maxabs={maxabs:.6g}")

    if metadata is not None and len(t.shape) >= 1:
        batch_size = t.shape[0]
        nan_mask = torch.isnan(t.view(batch_size, -1)).any(dim=1)
        nan_indices = torch.where(nan_mask)[0].cpu().tolist()

        if nan_indices:
            print(f"  [NaN batch indices]: {nan_indices}")

            gt_paths = metadata.get('gt_path', [])
            deg_names = metadata.get('deg_name', [])
            prompts = metadata.get('prompt', [])

            if not isinstance(gt_paths, list):
                gt_paths = [gt_paths] if gt_paths is not None else []
            if not isinstance(deg_names, list):
                deg_names = [deg_names] if deg_names is not None else []
            if not isinstance(prompts, list):
                prompts = [prompts] if prompts is not None else []

            for idx in nan_indices:
                gt_path = gt_paths[idx] if idx < len(gt_paths) else 'unknown'
                deg_name = deg_names[idx] if idx < len(deg_names) else 'unknown'
                prompt = prompts[idx] if idx < len(prompts) else 'unknown'
                print(f"    sample {idx}:")
                print(f"      gt_path: {gt_path}")
                print(f"      deg_name: {deg_name}")
                print(f"      prompt: {prompt}")


class ConvMoE(nn.Module):

    def __init__(self, in_channels, text_dim, img_dim, out_channels, kernel_size, stride=1, padding=0, bias=False, dilation=1, groups=1, num_experts=4, noisy_gating=True, use_experts=2):
        super(ConvMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.k = use_experts

        if num_experts == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation,
                                  groups=groups, bias=bias)
            self.is_moe = False
        else:
            self.route_func = RouteFunc(in_channels, text_dim, img_dim)
            self.sparse_conv_expert = ExpertConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, num_experts=num_experts)
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

        has_nan1 = torch.isnan(clean_values).any().item()
        has_nan2 = torch.isnan(threshold_if_in).any().item()
        has_nan3 = torch.isnan(noise_stddev).any().item()

        if has_nan1 or has_nan2 or has_nan3:
            print(has_nan1, has_nan2, has_nan3)

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        _chk("w_gate", self.w_gate)
        _chk("w_noise", self.w_noise)

        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits

            _chk("clean_logits", clean_logits)
            _chk("raw_noise_stddev", raw_noise_stddev)

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

        if not self.is_moe:
            y = self.conv(x)
            load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=True)
            return y, load_balancing_loss

        B, C, H, W = x.shape

        x_gating = self.route_func(x, text_embd, img_embd)

        _chk("x", x)
        _chk("x_gating", x_gating)

        gates, load, clean_logits = self.noisy_top_k_gating(x_gating, self.training)
        importance = gates.sum(0)

        load_balancing_loss = self.cv_squared(importance) * B / self.num_experts + self.cv_squared(load) * B * self.k / self.num_experts

        y = self.sparse_conv_expert(x, gates)

        return y, load_balancing_loss
