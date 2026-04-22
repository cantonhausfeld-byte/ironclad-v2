"""Smoke test: run Monte Carlo engine with synthetic features, assert structural invariants."""
import numpy as np
import pandas as pd
import pytest
from ironclad.simulation.engine import MonteCarloEngine


def _make_team_features(team, is_home, total=47.5, spread=-3.0):
    return pd.DataFrame([{
        "team": team,
        "is_home": is_home,
        "off_epa_per_play_l4": 0.05,
        "off_pass_rate_l4": 0.58,
        "def_epa_per_play_l4": -0.02,
        "def_sack_rate_l4": 0.06,
        "implied_total_from_odds": total,
        "spread_from_odds": spread,
        "home_win_prob_from_odds": 0.58 if is_home else 0.42,
        "altitude_ft": 0,
        "is_dome": False,
        "temp_f": 65.0,
        "wind_mph": 8.0,
        "precip_in": 0.0,
        "surface_grass": True,
    }])


def _make_player_features(team, n_players=6, is_home=True):
    positions = ["QB", "RB", "WR", "WR", "TE", "RB"][:n_players]
    rows = []
    for i, pos in enumerate(positions):
        rows.append({
            "player_id": f"{team}_p{i}",
            "player_name": f"{team} Player {i}",
            "team": team,
            "position": pos,
            "is_home": is_home,
            "availability": 1.0,
            "target_share_l4": 0.18 if pos in ("WR", "TE") else 0.1,
            "carry_share_l4": 0.45 if pos == "RB" else 0.0,
            "catch_rate_l4": 0.65,
            "yards_per_target_l4": 8.0,
            "yards_per_carry_l4": 4.2,
            "td_rate_per_target_l4": 0.05,
            "td_rate_per_carry_l4": 0.04,
            "opp_def_pass_epa_l4": -0.02,
            "opp_def_rush_epa_l4": -0.01,
            "opp_def_sack_rate_l4": 0.06,
            "team_off_pass_rate_l4": 0.58,
            "team_off_epa_l4": 0.05,
            "team_implied_total": 23.75,
        })
    return pd.DataFrame(rows)


N_DRAWS = 200  # small for speed


def test_win_probabilities_sum_to_one():
    engine = MonteCarloEngine(n_draws=N_DRAWS, seed=0)
    result = engine.run(
        home_team="KC", away_team="BAL",
        home_features=_make_team_features("KC", True),
        away_features=_make_team_features("BAL", False, spread=3.0),
        home_player_features=_make_player_features("KC", is_home=True),
        away_player_features=_make_player_features("BAL", is_home=False),
    )
    home_p, away_p = result.win_probability()
    assert abs(home_p + away_p - 1.0) < 1e-6
    assert 0.0 <= home_p <= 1.0


def test_scores_non_negative():
    engine = MonteCarloEngine(n_draws=N_DRAWS, seed=1)
    result = engine.run(
        home_team="KC", away_team="BAL",
        home_features=_make_team_features("KC", True),
        away_features=_make_team_features("BAL", False),
        home_player_features=_make_player_features("KC", is_home=True),
        away_player_features=_make_player_features("BAL", is_home=False),
    )
    for draw in result._draws:
        assert draw.home_score >= 0
        assert draw.away_score >= 0


def test_player_summary_has_expected_columns():
    engine = MonteCarloEngine(n_draws=N_DRAWS, seed=2)
    result = engine.run(
        home_team="KC", away_team="BAL",
        home_features=_make_team_features("KC", True),
        away_features=_make_team_features("BAL", False),
        home_player_features=_make_player_features("KC", is_home=True),
        away_player_features=_make_player_features("BAL", is_home=False),
    )
    psum = result.player_summary()
    if not psum.empty:
        assert "player_name" in psum.columns
        assert "metric" in psum.columns
        assert "p50" in psum.columns
        assert (psum["p25"] <= psum["p50"]).all()
        assert (psum["p50"] <= psum["p75"]).all()


def test_tds_non_negative_per_draw():
    engine = MonteCarloEngine(n_draws=N_DRAWS, seed=3)
    result = engine.run(
        home_team="KC", away_team="BAL",
        home_features=_make_team_features("KC", True),
        away_features=_make_team_features("BAL", False),
        home_player_features=_make_player_features("KC", is_home=True),
        away_player_features=_make_player_features("BAL", is_home=False),
    )
    for draw in result._draws:
        for stat in draw.player_stats:
            assert stat["tds"] >= 0
            assert stat["rec_yards"] >= 0
            assert stat["rush_yards"] >= 0
