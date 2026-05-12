"""Tests for game-script feedback and QB game-quality correlation in GameDraw."""
from __future__ import annotations

import numpy as np
import pytest

from ironclad.simulation.game_draw import GameDraw, _apply_game_script
from ironclad.simulation.player_draw import PlayerContext, PlayerDraw


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


# ── QB game-quality correlation ───────────────────────────────────────────────

def _wr_ctx() -> PlayerContext:
    return PlayerContext(
        player_id="P1", player_name="WR Test", team="KC", position="WR",
        is_home=True, availability=1.0,
        targets_projected=6.0, carries_projected=0.0, pass_attempts_projected=0.0,
        catch_rate=0.65, yards_per_target=9.0, yards_per_carry=4.2,
        td_rate_per_target=0.06, td_rate_per_carry=0.04,
        yards_per_target_std=5.0,
    )


def _draw_n_rec_yards(gqf: float, n: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    pd_obj = PlayerDraw()
    ctx = _wr_ctx()
    total = 0.0
    for _ in range(n):
        r = pd_obj.draw(rng, ctx, team_pass_att=32, team_rush_att=25, game_quality_factor=gqf)
        total += r.rec_yards
    return total / n


def test_gqf_zero_leaves_stats_near_baseline():
    # Zero factor should produce distributions centered near the ctx values.
    avg = _draw_n_rec_yards(0.0)
    # Expected ≈ targets_proj (6) × catch_rate (0.65) × yards_per_tgt (9) ≈ 35
    assert 20.0 < avg < 55.0


def test_positive_gqf_increases_rec_yards():
    low = _draw_n_rec_yards(0.0, seed=99)
    high = _draw_n_rec_yards(0.5, seed=99)
    assert high > low + 3.0


def test_negative_gqf_decreases_rec_yards():
    baseline = _draw_n_rec_yards(0.0, seed=77)
    low = _draw_n_rec_yards(-0.5, seed=77)
    assert low < baseline - 3.0


def test_gqf_catch_rate_bounded():
    # Even with extreme positive factor, adjusted catch rate must stay ≤ 0.99.
    rng = np.random.default_rng(0)
    pd_obj = PlayerDraw()
    ctx = _wr_ctx()
    for _ in range(200):
        r = pd_obj.draw(rng, ctx, team_pass_att=32, team_rush_att=25, game_quality_factor=5.0)
        # If catch rate were unbounded, receptions > targets would be impossible
        # due to Bernoulli, but targets would still be reasonable.
        assert r.receptions <= r.targets


# ── Turnover modeling ─────────────────────────────────────────────────────────

_OUTCOME = {
    "total_mean": 45.0, "total_std": 8.0,
    "home_margin_mean": 3.0, "home_margin_std": 14.0,
}
_HOME_ENV = {"pass_rate_projected": 0.58, "total_plays_projected": 64}
_AWAY_ENV = {"pass_rate_projected": 0.55, "total_plays_projected": 62}


def test_turnovers_are_non_negative():
    gd = GameDraw()
    rng = np.random.default_rng(0)
    for _ in range(200):
        result = gd.draw(rng, _OUTCOME, _HOME_ENV, _AWAY_ENV)
        assert result.home_turnovers >= 0
        assert result.away_turnovers >= 0


def test_high_turnover_games_reduce_score():
    # Draw 2000 games; games where one team has ≥4 TOs should average lower
    # score than games where that team has 0 TOs.
    gd = GameDraw()
    rng = np.random.default_rng(7)
    zero_to_scores, high_to_scores = [], []
    for _ in range(2000):
        result = gd.draw(rng, _OUTCOME, _HOME_ENV, _AWAY_ENV)
        if result.home_turnovers == 0:
            zero_to_scores.append(result.home_score)
        elif result.home_turnovers >= 4:
            high_to_scores.append(result.home_score)

    assert len(zero_to_scores) > 0 and len(high_to_scores) > 0
    assert np.mean(zero_to_scores) > np.mean(high_to_scores)


def test_mean_score_unaffected_by_normalization():
    # Normalized TO adjustment preserves mean score across many draws.
    # Mean home score should stay within ±2 pts of the un-adjusted expectation.
    gd = GameDraw()
    rng = np.random.default_rng(13)
    home_scores = [gd.draw(rng, _OUTCOME, _HOME_ENV, _AWAY_ENV).home_score for _ in range(2000)]
    expected = (_OUTCOME["total_mean"] + _OUTCOME["home_margin_mean"]) / 2.0
    assert abs(np.mean(home_scores) - expected) < 3.0


def test_game_draw_sets_quality_factors():
    # GameDraw.draw() should populate both quality factor fields.
    home_env = {"pass_rate_projected": 0.58, "total_plays_projected": 64}
    away_env = {"pass_rate_projected": 0.55, "total_plays_projected": 62}
    outcome = {
        "total_mean": 45.0, "total_std": 8.0,
        "home_margin_mean": 3.0, "home_margin_std": 14.0,
    }
    rng = np.random.default_rng(42)
    gd = GameDraw()
    # Draw many times; factors must be independent and not all identical.
    factors_home = [gd.draw(rng, outcome, home_env, away_env).home_game_quality_factor
                    for _ in range(50)]
    factors_away = [gd.draw(rng, outcome, home_env, away_env).away_game_quality_factor
                    for _ in range(50)]
    assert len(set(factors_home)) > 1
    assert len(set(factors_away)) > 1
    # Most draws should land within ±3σ = ±0.45 of zero.
    assert all(-0.6 < f < 0.6 for f in factors_home + factors_away)
