"""Tests for Monte Carlo distribution helpers."""
import numpy as np
import pytest
from ironclad.simulation.distributions import (
    truncated_normal,
    beta_from_mean_std,
    poisson_draw,
    bernoulli,
)


def test_truncated_normal_within_bounds():
    rng = np.random.default_rng(0)
    samples = truncated_normal(rng, mean=10, std=5, low=0, high=20, size=1000)
    assert all(0 <= s <= 20 for s in samples)


def test_truncated_normal_mean_approx():
    rng = np.random.default_rng(1)
    samples = truncated_normal(rng, mean=10, std=2, low=5, high=15, size=5000)
    assert abs(np.mean(samples) - 10) < 0.5


def test_beta_params_valid():
    a, b = beta_from_mean_std(0.55, 0.06)
    assert a > 0 and b > 0
    rng = np.random.default_rng(2)
    samples = rng.beta(a, b, size=1000)
    assert abs(np.mean(samples) - 0.55) < 0.05


def test_poisson_non_negative():
    rng = np.random.default_rng(3)
    for _ in range(100):
        assert poisson_draw(rng, 5.0) >= 0


def test_bernoulli_extremes():
    rng = np.random.default_rng(4)
    assert bernoulli(rng, 0.0) is False
    # p=1 should always return True
    results = [bernoulli(rng, 1.0) for _ in range(10)]
    assert all(results)
