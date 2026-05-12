"""Tests for EloComputer and Elo-related model features."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables


def _conn_with_games(games: list[dict]):
    conn = in_memory_connection()
    create_all_tables(conn)
    for g in games:
        conn.execute("""
            INSERT INTO silver.games
                (game_id, season, season_type, week, gameday, home_team, away_team,
                 home_score, away_score)
            VALUES (?, ?, 'REG', ?, ?, ?, ?, ?, ?)
        """, [g["game_id"], g["season"], g["week"], g["gameday"],
              g["home"], g["away"], g["home_score"], g["away_score"]])
    return conn


# ── Initial ratings ───────────────────────────────────────────────────────────

def test_elo_initial_rating_is_1500():
    """Both teams start at 1500 for their very first game."""
    conn = _conn_with_games([
        {"game_id": "2016_01_KC_NE", "season": 2016, "week": 1,
         "gameday": date(2016, 9, 11), "home": "NE", "away": "KC",
         "home_score": 27, "away_score": 14},
    ])
    from ironclad.features.elo import EloComputer
    ec = EloComputer(conn)
    rows = ec._compute_all()
    assert not rows.empty
    ne_row = rows[rows["team"] == "NE"].iloc[0]
    kc_row = rows[rows["team"] == "KC"].iloc[0]
    assert ne_row["elo_pre_game"] == pytest.approx(1500.0)
    assert kc_row["elo_pre_game"] == pytest.approx(1500.0)


# ── Winner gains, loser loses ─────────────────────────────────────────────────

def test_elo_winner_gains_loser_loses():
    """After a game, the winner's post-game Elo > pre-game; loser's < pre-game."""
    conn = _conn_with_games([
        {"game_id": "2016_01_KC_NE", "season": 2016, "week": 1,
         "gameday": date(2016, 9, 11), "home": "NE", "away": "KC",
         "home_score": 27, "away_score": 14},
    ])
    from ironclad.features.elo import EloComputer
    rows = EloComputer(conn)._compute_all()
    ne = rows[rows["team"] == "NE"].iloc[0]
    kc = rows[rows["team"] == "KC"].iloc[0]
    assert ne["elo_post_game"] > ne["elo_pre_game"]
    assert kc["elo_post_game"] < kc["elo_pre_game"]
    # Zero-sum: gains cancel
    assert (ne["elo_post_game"] - ne["elo_pre_game"]) == pytest.approx(
        -(kc["elo_post_game"] - kc["elo_pre_game"]), abs=1e-4
    )


# ── Season regression ─────────────────────────────────────────────────────────

def test_elo_season_regression_toward_1500():
    """A team above 1500 after season 1 should start season 2 closer to 1500."""
    conn = _conn_with_games([
        {"game_id": "2016_01_NE_NY", "season": 2016, "week": 1,
         "gameday": date(2016, 9, 11), "home": "NE", "away": "NYG",
         "home_score": 36, "away_score": 7},
        {"game_id": "2017_01_NE_KC", "season": 2017, "week": 1,
         "gameday": date(2017, 9, 7), "home": "NE", "away": "KC",
         "home_score": 27, "away_score": 14},
    ])
    from ironclad.features.elo import EloComputer
    rows = EloComputer(conn)._compute_all()
    ne_2016 = rows[(rows["team"] == "NE") & (rows["season"] == 2016)].iloc[0]
    ne_2017 = rows[(rows["team"] == "NE") & (rows["season"] == 2017)].iloc[0]
    # NE won big in 2016 → elo_post_game > 1500
    assert ne_2016["elo_post_game"] > 1500
    # Season 2 pre-game should be between 1500 and 2016 post-game (regression)
    assert 1500 < ne_2017["elo_pre_game"] < ne_2016["elo_post_game"]


# ── compute_and_write ─────────────────────────────────────────────────────────

def test_elo_compute_and_write_returns_row_count():
    conn = _conn_with_games([
        {"game_id": "2016_01_KC_NE", "season": 2016, "week": 1,
         "gameday": date(2016, 9, 11), "home": "NE", "away": "KC",
         "home_score": 27, "away_score": 14},
        {"game_id": "2016_02_NE_MIA", "season": 2016, "week": 2,
         "gameday": date(2016, 9, 18), "home": "NE", "away": "MIA",
         "home_score": 31, "away_score": 24},
    ])
    from ironclad.features.elo import EloComputer
    n = EloComputer(conn).compute_and_write()
    assert n == 4  # 2 teams × 2 games
    stored = conn.execute("SELECT COUNT(*) FROM gold.elo_ratings").fetchone()[0]
    assert stored == 4


# ── Model diff features ───────────────────────────────────────────────────────

def test_elo_diff_in_build_diff_features():
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{"home_elo_pre_game": 1550.0, "away_elo_pre_game": 1480.0}])
    Xf = _build_diff_features(X)
    assert Xf["elo_diff"].iloc[0] == pytest.approx(70.0, abs=1e-4)


def test_game_week_in_build_diff_features():
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{"home_week": 14}])
    Xf = _build_diff_features(X)
    assert Xf["game_week"].iloc[0] == pytest.approx(14.0)


def test_game_week_default_when_absent():
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{}])
    Xf = _build_diff_features(X)
    assert Xf["game_week"].iloc[0] == pytest.approx(9.0)


# ── Phase 2G: defense-side diffs ─────────────────────────────────────────────

def test_def_success_rate_diff_in_build_diff_features():
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{"home_def_success_rate_l4": 0.38, "away_def_success_rate_l4": 0.45}])
    Xf = _build_diff_features(X)
    assert Xf["def_success_rate_diff"].iloc[0] == pytest.approx(0.38 - 0.45, abs=1e-4)


def test_yards_per_play_diff_in_build_diff_features():
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{"home_off_yards_per_play_l4": 5.8, "away_off_yards_per_play_l4": 5.2}])
    Xf = _build_diff_features(X)
    assert Xf["yards_per_play_diff"].iloc[0] == pytest.approx(0.6, abs=1e-4)


def test_def_epa_pass_early_diff_in_build_diff_features():
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{"home_def_epa_pass_early_l4": 0.12, "away_def_epa_pass_early_l4": 0.08}])
    Xf = _build_diff_features(X)
    assert Xf["def_epa_pass_early_diff"].iloc[0] == pytest.approx(0.04, abs=1e-4)


# ── Phase 2H: momentum features ──────────────────────────────────────────────

def test_off_epa_momentum_diff_in_build_diff_features():
    """Momentum = (home_L1 - home_L4) - (away_L1 - away_L4)."""
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{
        "home_off_epa_per_play_l1": 0.20,
        "home_off_epa_per_play_l4": 0.10,
        "away_off_epa_per_play_l1": 0.05,
        "away_off_epa_per_play_l4": 0.10,
    }])
    Xf = _build_diff_features(X)
    # home trending up +0.10, away trending down -0.05 → diff = 0.15
    assert Xf["off_epa_momentum_diff"].iloc[0] == pytest.approx(0.15, abs=1e-4)


def test_momentum_diff_defaults_zero_when_absent():
    """When L1 columns are absent the momentum diff should be 0."""
    from ironclad.models.team.game_outcome import _build_diff_features
    X = pd.DataFrame([{}])
    Xf = _build_diff_features(X)
    assert Xf["off_epa_momentum_diff"].iloc[0] == pytest.approx(0.0, abs=1e-4)
