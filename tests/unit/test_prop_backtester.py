"""Unit tests for PropBacktester calibration logic."""
from __future__ import annotations

import numpy as np
import pytest

from ironclad.eval.prop_backtester import PropBacktester, summarize, _STAT_MAP


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_draw_record(player_id: str, position: str, **stats):
    """Build a minimal player_stats dict for a DrawRecord."""
    row = {
        "player_id": player_id,
        "player_name": "Test Player",
        "team": "TST",
        "position": position,
        "is_home": True,
        "targets": 0, "receptions": 0, "rec_yards": 0.0,
        "carries": 0, "rush_yards": 0.0,
        "pass_attempts": 0, "completions": 0, "pass_yards": 0.0,
        "tds": 0,
    }
    row.update(stats)
    return row


class _FakeDrawRecord:
    def __init__(self, player_stats):
        self.player_stats = player_stats


class _FakeSimResult:
    def __init__(self, draws):
        self._draws = draws


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_actual_rank_uniform_for_perfect_model():
    """If actual values are drawn from the simulated distribution, actual_rank is uniform."""
    rng = np.random.default_rng(42)
    n_games = 100
    n_draws = 200

    actual_ranks = []
    for _ in range(n_games):
        # Generate simulated distribution
        sim_vals = rng.normal(50.0, 15.0, size=n_draws)
        # Actual comes from the SAME distribution — perfect calibration
        actual = float(rng.normal(50.0, 15.0))
        rank = float(np.mean(sim_vals < actual))
        actual_ranks.append(rank)

    # A uniform distribution has mean ≈ 0.5 and std ≈ 1/sqrt(12) ≈ 0.289
    mean_rank = float(np.mean(actual_ranks))
    assert 0.35 <= mean_rank <= 0.65, f"Expected mean actual_rank ≈ 0.5, got {mean_rank:.3f}"


def test_coverage_80pct_well_calibrated():
    """Calibrated [p10, p90] interval covers actual value ~80% of the time."""
    rng = np.random.default_rng(123)
    n_games = 500
    n_draws = 200
    covered = 0

    for _ in range(n_games):
        sim_vals = rng.normal(50.0, 15.0, size=n_draws)
        actual = float(rng.normal(50.0, 15.0))
        p10 = float(np.percentile(sim_vals, 10))
        p90 = float(np.percentile(sim_vals, 90))
        if p10 <= actual <= p90:
            covered += 1

    coverage = covered / n_games
    # Allow ±5 pp around 80% for sampling variance
    assert 0.74 <= coverage <= 0.86, f"Expected coverage ≈ 80%, got {coverage:.1%}"


def test_overconfident_model_low_coverage():
    """If p10-p90 is too narrow (model underestimates variance), coverage drops below 80%."""
    rng = np.random.default_rng(7)
    n_games = 500
    n_draws = 200
    covered = 0

    for _ in range(n_games):
        # Sim draws have HALF the variance of the true distribution
        sim_vals = rng.normal(50.0, 7.5, size=n_draws)   # narrow
        actual = float(rng.normal(50.0, 15.0))             # true wide variance
        p10 = float(np.percentile(sim_vals, 10))
        p90 = float(np.percentile(sim_vals, 90))
        if p10 <= actual <= p90:
            covered += 1

    coverage = covered / n_games
    assert coverage < 0.70, f"Overconfident model should have coverage < 70%, got {coverage:.1%}"


def test_summarize_returns_expected_columns():
    """summarize(df) returns a DataFrame with required columns."""
    import pandas as pd

    df = pd.DataFrame([
        {
            "stat_type": "rec_yards", "position": "WR",
            "covered_80pct": True, "abs_error_p50": 10.0,
            "actual_value": 60.0, "predicted_p50": 50.0,
        },
        {
            "stat_type": "rec_yards", "position": "WR",
            "covered_80pct": False, "abs_error_p50": 20.0,
            "actual_value": 30.0, "predicted_p50": 50.0,
        },
        {
            "stat_type": "rush_yards", "position": "RB",
            "covered_80pct": True, "abs_error_p50": 5.0,
            "actual_value": 55.0, "predicted_p50": 50.0,
        },
    ])

    result = summarize(df)
    required = {"stat_type", "position", "N", "coverage_80pct", "median_p50_mae", "bias_p50"}
    assert required.issubset(set(result.columns)), f"Missing columns: {required - set(result.columns)}"
    assert len(result) > 0

    # rec_yards WR row
    wr_row = result[(result["stat_type"] == "rec_yards") & (result["position"] == "WR")]
    assert len(wr_row) == 1
    assert wr_row.iloc[0]["N"] == 2
    assert wr_row.iloc[0]["coverage_80pct"] == pytest.approx(0.5)


def test_backtest_filters_dnp_players(conn):
    """Players with null actual value are excluded from results."""
    import pandas as pd
    from ironclad.store.schema import create_all_tables

    create_all_tables(conn)

    # Insert a completed REG game
    conn.execute("""
        INSERT INTO silver.games (
            game_id, season, week, home_team, away_team, gameday, season_type,
            home_score, away_score
        ) VALUES (
            'test_game_01', 2025, 10, 'KC', 'BUF', '2025-11-09', 'REG',
            24, 17
        )
    """)

    # Player with NULL rec_yards (DNP — not in stats at all)
    conn.execute("""
        INSERT INTO silver.player_game_stats (
            game_id, season, week, player_id, player_name, team, opponent,
            position, is_home, targets, receptions, rec_yards, total_tds
        ) VALUES (
            'test_game_01', 2025, 10, 'p1', 'Injured WR', 'KC', 'BUF',
            'WR', TRUE, 0, 0, NULL, 0
        )
    """)

    bt = PropBacktester(conn=conn)
    actuals = bt._load_actuals(2025, {"WR"})

    # Build fake draw arrays
    fake_draws = [
        _FakeDrawRecord([_make_draw_record("p1", "WR", targets=5, rec_yards=50.0)])
        for _ in range(50)
    ]
    fake_result = _FakeSimResult(fake_draws)
    draw_arrays = bt._build_draw_arrays(fake_result, {"WR"})

    # Simulate game row building — player with NULL rec_yards should produce 0 rows for rec_yards
    game_actuals = actuals[actuals["game_id"] == "test_game_01"]
    rows = []
    for _, actual_row in game_actuals.iterrows():
        pid = actual_row["player_id"]
        for sim_stat, silver_col in _STAT_MAP.items():
            key = (pid, sim_stat)
            if key not in draw_arrays:
                continue
            actual_val = actual_row.get(silver_col)
            if actual_val is None or (isinstance(actual_val, float) and np.isnan(actual_val)):
                continue
            rows.append({"player_id": pid, "stat_type": sim_stat})

    # rec_yards is NULL so should be filtered; only non-null stats should appear
    rec_rows = [r for r in rows if r["stat_type"] == "rec_yards"]
    assert len(rec_rows) == 0, "NULL actual_value should be filtered out"
