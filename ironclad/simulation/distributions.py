"""Statistical distribution helpers for Monte Carlo draws."""
from __future__ import annotations

import numpy as np


def truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float = 0.0,
    high: float = np.inf,
    size: int = 1,
) -> np.ndarray:
    """Sample from a normal truncated to [low, high]."""
    samples = rng.normal(mean, std, size=size * 4)
    samples = samples[(samples >= low) & (samples <= high)]
    # If not enough samples, pad with mean
    while len(samples) < size:
        extra = rng.normal(mean, std, size=size)
        extra = extra[(extra >= low) & (extra <= high)]
        samples = np.concatenate([samples, extra])
    return samples[:size]


def beta_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    """Fit Beta(α, β) from a mean and std."""
    mean = max(0.01, min(0.99, mean))
    var = std ** 2
    var = min(var, mean * (1 - mean) * 0.99)
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta = (1 - mean) * (mean * (1 - mean) / var - 1)
    return max(0.1, alpha), max(0.1, beta)


def poisson_draw(rng: np.random.Generator, lam: float) -> int:
    lam = max(0.0, lam)
    return int(rng.poisson(lam))


def bernoulli(rng: np.random.Generator, p: float) -> bool:
    p = max(0.0, min(1.0, p))
    return bool(rng.random() < p)
