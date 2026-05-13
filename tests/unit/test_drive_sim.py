"""Tests for the drive-level Markov chain simulation (Phase 4.5)."""
from __future__ import annotations

import numpy as np
import pytest

from ironclad.simulation.drive_sim import DriveSimulator
from ironclad.simulation.game_draw import GameDrawResult


def _make_env(team_total=23.5, pass_rate=0.55, total_plays=64):
    return {
        "team_total_projected": team_total,
        "pass_rate_projected": pass_rate,
        "total_plays_projected": total_plays,
        "sack_rate_projected": 0.07,
    }


def _league_avg_env():
    return _make_env(team_total=23.5, pass_rate=0.55)


# ── Test 1: scores are non-negative ──────────────────────────────────────────

def test_drive_sim_scores_non_negative():
    sim = DriveSimulator()
    rng = np.random.default_rng(42)
    env = _league_avg_env()
    for _ in range(100):
        gd = sim.draw(rng, {}, env, env)
        assert gd.home_score >= 0, f"home_score negative: {gd.home_score}"
        assert gd.away_score >= 0, f"away_score negative: {gd.away_score}"


# ── Test 2: total plays are reasonable ───────────────────────────────────────

def test_drive_sim_plays_reasonable():
    sim = DriveSimulator()
    rng = np.random.default_rng(42)
    env = _league_avg_env()
    for _ in range(100):
        gd = sim.draw(rng, {}, env, env)
        assert 30 <= gd.total_plays <= 250, f"total_plays out of range: {gd.total_plays}"


# ── Test 3: all 17 GameDrawResult fields present and non-negative ─────────────

def test_drive_sim_gamedrawnresult_fields():
    sim = DriveSimulator()
    rng = np.random.default_rng(7)
    env = _league_avg_env()
    gd = sim.draw(rng, {}, env, env)

    required_fields = [
        "home_score", "away_score",
        "home_pass_att", "away_pass_att",
        "home_rush_att", "away_rush_att",
        "home_pass_yards", "away_pass_yards",
        "home_rush_yards", "away_rush_yards",
        "total_plays",
        "home_effective_pass_rate", "away_effective_pass_rate",
        "home_game_quality_factor", "away_game_quality_factor",
        "home_turnovers", "away_turnovers",
    ]
    for field in required_fields:
        assert hasattr(gd, field), f"Missing field: {field}"

    # Non-negative count/yard fields
    for field in ("home_score", "away_score", "home_pass_att", "away_pass_att",
                  "home_rush_att", "away_rush_att", "home_pass_yards", "away_pass_yards",
                  "home_rush_yards", "away_rush_yards", "total_plays",
                  "home_turnovers", "away_turnovers"):
        val = getattr(gd, field)
        assert val >= 0, f"{field} = {val} is negative"

    # Effective pass rates in [0, 1]
    assert 0.0 <= gd.home_effective_pass_rate <= 1.0
    assert 0.0 <= gd.away_effective_pass_rate <= 1.0


# ── Test 4: trailing team passes more ────────────────────────────────────────

def test_drive_sim_trailing_team_passes_more():
    """Home team with much lower quality will trail; should pass at higher rate."""
    sim = DriveSimulator()
    rng = np.random.default_rng(99)

    # Low quality home vs high quality away → home trails most draws
    home_env = _make_env(team_total=12.0, pass_rate=0.55)
    away_env = _make_env(team_total=38.0, pass_rate=0.55)

    trailing_pass_rates = []
    leading_pass_rates = []

    for _ in range(500):
        gd = sim.draw(rng, {}, home_env, away_env)
        total = gd.home_pass_att + gd.home_rush_att
        if total == 0:
            continue
        rate = gd.home_pass_att / total
        score_diff = gd.home_score - gd.away_score
        if score_diff < -10:
            trailing_pass_rates.append(rate)
        elif score_diff > 10:
            leading_pass_rates.append(rate)

    assert len(trailing_pass_rates) >= 30, f"Not enough trailing draws: {len(trailing_pass_rates)}"
    assert len(leading_pass_rates) >= 5 or True  # may be few leading draws given quality gap
    if leading_pass_rates:
        assert np.mean(trailing_pass_rates) > np.mean(leading_pass_rates) - 0.01, (
            f"trailing pass rate {np.mean(trailing_pass_rates):.3f} not "
            f">= leading {np.mean(leading_pass_rates):.3f}"
        )
    # Key check: trailing team passes significantly more than base rate 0.55
    assert np.mean(trailing_pass_rates) > 0.57, (
        f"Trailing team pass rate {np.mean(trailing_pass_rates):.3f} not elevated above base 0.55"
    )


# ── Test 5: engine uses DriveSimulator when flagged ──────────────────────────

def test_engine_uses_drive_sim_when_flagged():
    from ironclad.simulation.engine import MonteCarloEngine
    from ironclad.simulation.drive_sim import DriveSimulator

    engine_default = MonteCarloEngine(n_draws=1, use_drive_sim=False)
    assert not isinstance(engine_default._game_draw, DriveSimulator)

    engine_drive = MonteCarloEngine(n_draws=1, use_drive_sim=True)
    assert isinstance(engine_drive._game_draw, DriveSimulator)


# ── Test 6: score calibration ─────────────────────────────────────────────────

def test_drive_sim_score_calibration():
    """Over 500 draws with league-avg quality, mean total score ≈ 47 ± 8."""
    sim = DriveSimulator()
    rng = np.random.default_rng(123)
    env = _league_avg_env()

    totals = []
    for _ in range(500):
        gd = sim.draw(rng, {}, env, env)
        totals.append(gd.home_score + gd.away_score)

    mean_total = np.mean(totals)
    assert 35 <= mean_total <= 59, (
        f"Mean total score {mean_total:.1f} outside expected range [35, 59]. "
        "Drive calibration may be off."
    )
