from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

Tensor = torch.Tensor
RNG = np.random.Generator


# ============================================================================
# Noise Functions (copied from utils/sim_ops.py)
# ============================================================================

def gaussian_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak Gaussian noise (sigma=0.01)."""
    sigma = 0.01
    noise = torch.from_numpy(rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = torch.clamp(clean + noise, 0, 1)
    return inp, clean


def gaussian_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium Gaussian noise (sigma=0.05)."""
    sigma = 0.05
    noise = torch.from_numpy(rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = torch.clamp(clean + noise, 0, 1)
    return inp, clean


def gaussian_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong Gaussian noise (sigma=0.1)."""
    sigma = 0.1
    noise = torch.from_numpy(rng.normal(0.0, sigma, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = torch.clamp(clean + noise, 0, 1)
    return inp, clean


def uniform_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak uniform noise (max_abs=0.05)."""
    max_abs = 0.05
    noise = torch.from_numpy(rng.uniform(-max_abs, max_abs, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = clean * (1.0 + noise) if rng.random() < 0.5 else clean + noise
    return torch.clamp(inp, 0, 1), clean


def uniform_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium uniform noise (max_abs=0.1)."""
    max_abs = 0.1
    noise = torch.from_numpy(rng.uniform(-max_abs, max_abs, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = clean * (1.0 + noise) if rng.random() < 0.5 else clean + noise
    return torch.clamp(inp, 0, 1), clean


def uniform_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong uniform noise (max_abs=0.2)."""
    max_abs = 0.2
    noise = torch.from_numpy(rng.uniform(-max_abs, max_abs, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = clean * (1.0 + noise) if rng.random() < 0.5 else clean + noise
    return torch.clamp(inp, 0, 1), clean


def poisson_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak Poisson noise (lambda=500)."""
    lam = 500.0
    inp = torch.poisson(torch.clamp(clean, 0, 1) * lam) / lam
    return torch.clamp(inp, 0, 1), clean


def poisson_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium Poisson noise (lambda=1000)."""
    lam = 1000.0
    inp = torch.poisson(torch.clamp(clean, 0, 1) * lam) / lam
    return torch.clamp(inp, 0, 1), clean


def poisson_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong Poisson noise (lambda=2000)."""
    lam = 2000.0
    inp = torch.poisson(torch.clamp(clean, 0, 1) * lam) / lam
    return torch.clamp(inp, 0, 1), clean


def rayleigh_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak Rayleigh noise (scale=0.02)."""
    scale = 0.02
    u = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    rayleigh = scale * torch.sqrt(-2.0 * torch.log(torch.clamp(1.0 - u, min=1e-6)))
    if rng.random() < 0.5:
        inp = clean + rayleigh.expand_as(clean)
    else:
        mean_rayleigh = scale * 1.2533141373155001
        inp = clean * (1.0 + (rayleigh - mean_rayleigh).expand_as(clean))
    return torch.clamp(inp, 0, 1), clean


def rayleigh_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium Rayleigh noise (scale=0.1)."""
    scale = 0.1
    u = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    rayleigh = scale * torch.sqrt(-2.0 * torch.log(torch.clamp(1.0 - u, min=1e-6)))
    if rng.random() < 0.5:
        inp = clean + rayleigh.expand_as(clean)
    else:
        mean_rayleigh = scale * 1.2533141373155001
        inp = clean * (1.0 + (rayleigh - mean_rayleigh).expand_as(clean))
    return torch.clamp(inp, 0, 1), clean


def rayleigh_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong Rayleigh noise (scale=0.2)."""
    scale = 0.2
    u = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    rayleigh = scale * torch.sqrt(-2.0 * torch.log(torch.clamp(1.0 - u, min=1e-6)))
    if rng.random() < 0.5:
        inp = clean + rayleigh.expand_as(clean)
    else:
        mean_rayleigh = scale * 1.2533141373155001
        inp = clean * (1.0 + (rayleigh - mean_rayleigh).expand_as(clean))
    return torch.clamp(inp, 0, 1), clean


def gamma_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak Gamma noise (shape=80, scale=0.005)."""
    shape_k, scale_theta = 80.0, 0.005
    gamma_dist = torch.distributions.Gamma(torch.tensor(shape_k, device=clean.device, dtype=clean.dtype),
                                           torch.tensor(1.0 / scale_theta, device=clean.device, dtype=clean.dtype))
    noise = gamma_dist.sample(clean.shape)
    mean_gamma = shape_k * scale_theta
    inp = clean * (noise / mean_gamma) if rng.random() < 0.5 else clean + (noise - mean_gamma)
    return torch.clamp(inp, 0, 1), clean


def gamma_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium Gamma noise (shape=50, scale=0.01)."""
    shape_k, scale_theta = 50.0, 0.01
    gamma_dist = torch.distributions.Gamma(torch.tensor(shape_k, device=clean.device, dtype=clean.dtype),
                                           torch.tensor(1.0 / scale_theta, device=clean.device, dtype=clean.dtype))
    noise = gamma_dist.sample(clean.shape)
    mean_gamma = shape_k * scale_theta
    inp = clean * (noise / mean_gamma) if rng.random() < 0.5 else clean + (noise - mean_gamma)
    return torch.clamp(inp, 0, 1), clean


def gamma_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong Gamma noise (shape=20, scale=0.02)."""
    shape_k, scale_theta = 20.0, 0.02
    gamma_dist = torch.distributions.Gamma(torch.tensor(shape_k, device=clean.device, dtype=clean.dtype),
                                           torch.tensor(1.0 / scale_theta, device=clean.device, dtype=clean.dtype))
    noise = gamma_dist.sample(clean.shape)
    mean_gamma = shape_k * scale_theta
    inp = clean * (noise / mean_gamma) if rng.random() < 0.5 else clean + (noise - mean_gamma)
    return torch.clamp(inp, 0, 1), clean


def impulse_white_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak impulse white noise (prob=0.005)."""
    prob = 0.005
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype) < prob
    inp = clean.clone().masked_fill(mask.expand_as(clean), 1.0)
    return inp, clean


def impulse_white_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium impulse white noise (prob=0.02)."""
    prob = 0.02
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype) < prob
    inp = clean.clone().masked_fill(mask.expand_as(clean), 1.0)
    return inp, clean


def impulse_white_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong impulse white noise (prob=0.05)."""
    prob = 0.05
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype) < prob
    inp = clean.clone().masked_fill(mask.expand_as(clean), 1.0)
    return inp, clean


def salt_pepper_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak salt-pepper noise (prob=0.01)."""
    prob = 0.01
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    inp = clean.clone()
    inp = inp.masked_fill((mask < prob / 2).expand_as(clean), 1.0)
    inp = inp.masked_fill((mask > 1 - prob / 2).expand_as(clean), 0.0)
    return inp, clean


def salt_pepper_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium salt-pepper noise (prob=0.1)."""
    prob = 0.1
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    inp = clean.clone()
    inp = inp.masked_fill((mask < prob / 2).expand_as(clean), 1.0)
    inp = inp.masked_fill((mask > 1 - prob / 2).expand_as(clean), 0.0)
    return inp, clean


def salt_pepper_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong salt-pepper noise (prob=0.3)."""
    prob = 0.3
    mask = torch.rand((1, clean.shape[-2], clean.shape[-1]), device=clean.device, dtype=clean.dtype)
    inp = clean.clone()
    inp = inp.masked_fill((mask < prob / 2).expand_as(clean), 1.0)
    inp = inp.masked_fill((mask > 1 - prob / 2).expand_as(clean), 0.0)
    return inp, clean


def speckle_noise_weak(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Weak speckle noise (var=0.01)."""
    var = 0.01
    noise = torch.from_numpy(rng.normal(0.0, var, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = torch.clamp(clean * (1.0 + noise), 0, 1)
    return inp, clean


def speckle_noise_medium(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Medium speckle noise (var=0.05)."""
    var = 0.05
    noise = torch.from_numpy(rng.normal(0.0, var, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = torch.clamp(clean * (1.0 + noise), 0, 1)
    return inp, clean


def speckle_noise_strong(clean: Tensor, rng: RNG) -> Tuple[Tensor, Tensor]:
    """Strong speckle noise (var=0.1)."""
    var = 0.1
    noise = torch.from_numpy(rng.normal(0.0, var, size=tuple(clean.shape)).astype(np.float32)).to(clean.device)
    inp = torch.clamp(clean * (1.0 + noise), 0, 1)
    return inp, clean

