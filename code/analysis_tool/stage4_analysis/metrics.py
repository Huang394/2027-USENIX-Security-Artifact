"""Metric computation for mechanism analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ResidualMetrics:
    l1: float
    l2: float
    psnr: float
    ssim: float


def compute_residual_metrics(a: torch.Tensor, b: torch.Tensor) -> ResidualMetrics:
    diff = torch.abs(a - b)
    mse = torch.mean((a - b) ** 2).item()
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    return ResidualMetrics(
        l1=float(diff.mean().item()),
        l2=float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        psnr=psnr,
        ssim=_simple_ssim(a, b),
    )


def _simple_ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = a_flat.mean(dim=1)
    mu_b = b_flat.mean(dim=1)
    var_a = a_flat.var(dim=1, unbiased=False)
    var_b = b_flat.var(dim=1, unbiased=False)
    cov = ((a_flat - mu_a[:, None]) * (b_flat - mu_b[:, None])).mean(dim=1)
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return float(score.mean().item())


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a - b, dim=1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=1)
