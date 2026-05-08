"""Tests for game-script feedback in GameDraw."""
from __future__ import annotations

import numpy as np
import pytest

from ironclad.simulation.game_draw import GameDraw, _apply_game_script


BASE_RATE = 0.58


def _avg_effective(margin: float, n: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    rates = [_apply_game_script(rng, BASE_RATE, margin) for _ in range(n)]
    return float(np.mean(rates))


def test_neutral_margin_uses_base_rate():
    # Margin = 0 → halftime estimate is centered on 0 with std 7. With a 7-pt
    # threshold for urgency/comfort, ~16% of draws hit each tail. The blended
    # rate stays within ~0.005 of base because urgency and comfort roughly
    # cancel.
    avg = _avg_effective(0.0)
    assert abs(avg - BASE_RATE) < 0.01


def test_trailing_team_increases_pass_rate():
    # Margin -21 from home perspective → home_effective_pass_rate higher.
    avg = _avg_effective(-21.0)
    assert avg > BASE_RATE + 0.005


def test_leading_team_decreases_pass_rate():
    # Margin +21 → home_effective_pass_rate lower. Comfort effect is
    # asymmetrically smaller than urgency by design (clock-kill < urgency).
    avg = _avg_effective(21.0)
    assert avg < BASE_RATE - 0.004


def test_effective_rate_bounded():
    rng = np.random.default_rng(7)
    for m in np.linspace(-35.0, 35.0, 100):
        for _ in range(10):
            r = _apply_game_script(rng, BASE_RATE, m)
            assert 0.20 <= r <= 0.85


def test_pass_att_reflects_script():
    # In a blowout where home is drawn to lose by 35, paired draws should
    # show a script-adjusted home_pass_att higher than a neutral-script
    # baseline, all else equal.
    home_env = {"pass_rate_projected": BASE_RATE, "total_plays_projected": 64}
    away_env = {"pass_rate_projected": BASE_RATE, "total_plays_projected": 64}
    blowout_outcome = {
        "total_mean": 45.0, "total_std": 8.0,
        "home_margin_mean": -35.0, "home_margin_std": 2.0,
    }
    neutral_outcome = {
        "total_mean": 45.0, "total_std": 8.0,
        "home_margin_mean": 0.0, "home_margin_std": 2.0,
    }
    gd = GameDraw()
    rng_blow = np.random.default_rng(123)
    rng_neut = np.random.default_rng(123)

    blow, neut = [], []
    for _ in range(300):
        blow.append(gd.draw(rng_blow, blowout_outcome, home_env, away_env).home_pass_att)
        neut.append(gd.draw(rng_neut, neutral_outcome, home_env, away_env).home_pass_att)

    assert np.mean(blow) > np.mean(neut) + 0.5
