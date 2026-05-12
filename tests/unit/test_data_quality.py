"""Tests for ironclad/store/data_quality.py."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ironclad.store.data_quality import (
    check_completeness_score,
    check_epa_range,
    check_null_rates,
    check_row_counts,
    run_all_checks,
)


def _insert_schedule_row(conn, season: int = 2023) -> None:
    conn.execute("""
        INSERT INTO bronze.schedules
            (game_id, season, season_type, week, gameday, away_team, home_team)
        VALUES ('2023_01_KC_BAL', ?, 'REG', 1, '2023-09-07', 'KC', 'BAL')
    """, [season])


def _insert_silver_stats(conn, season: int = 2023,
                          epa_per_play: float = 0.1,
                          epa_pass: float = 0.2,
                          epa_rush: float = -0.1) -> None:
    conn.execute("""
        INSERT INTO silver.team_game_stats
            (game_id, season, week, team, opponent, is_home,
             epa_per_play, epa_pass, epa_rush)
        VALUES ('2023_01_KC_BAL', ?, 1, 'KC', 'BAL', true, ?, ?, ?)
    """, [season, epa_per_play, epa_pass, epa_rush])


def _insert_gold_features(conn, season: int = 2023,
                           off_epa=0.1, implied_total=45.0, pass_rate=0.6) -> None:
    cutoff = datetime(2023, 9, 7, 17, 30, tzinfo=timezone.utc)
    conn.execute("""
        INSERT INTO gold.team_game_features
            (game_id, season, week, team, opponent, is_home,
             cutoff_ts, feature_version,
             off_epa_per_play_l4, implied_total_from_odds, off_pass_rate_l4)
        VALUES ('2023_01_KC_BAL', ?, 1, 'KC', 'BAL', true, ?, 'v1.0', ?, ?, ?)
    """, [season, cutoff, off_epa, implied_total, pass_rate])


# ── check_row_counts ──────────────────────────────────────────────────────────

def test_check_row_counts_empty(conn):
    result = check_row_counts(conn, season=2023)
    assert result["passed"] is False
    for table, count in result["counts"].items():
        assert count == 0, f"Expected 0 rows for {table}, got {count}"


def test_check_row_counts_with_data(conn):
    _insert_schedule_row(conn, season=2023)
    result = check_row_counts(conn, season=2023)
    assert result["counts"]["bronze.schedules"] == 1


# ── check_null_rates ──────────────────────────────────────────────────────────

def test_check_null_rates_all_null(conn):
    _insert_gold_features(conn, off_epa=None, implied_total=None, pass_rate=None)
    result = check_null_rates(conn, season=2023)
    nr = result["null_rates"]
    assert nr["gold.team_game_features.off_epa_per_play_l4"] == pytest.approx(1.0)
    assert nr["gold.team_game_features.implied_total_from_odds"] == pytest.approx(1.0)
    assert nr["gold.team_game_features.off_pass_rate_l4"] == pytest.approx(1.0)
    assert result["passed"] is False


def test_check_null_rates_populated(conn):
    _insert_gold_features(conn)
    result = check_null_rates(conn, season=2023)
    for key, pct in result["null_rates"].items():
        assert pct == pytest.approx(0.0), f"{key} should have 0% nulls"
    assert result["passed"] is True


# ── check_epa_range ───────────────────────────────────────────────────────────

def test_check_epa_range_clean(conn):
    _insert_silver_stats(conn, epa_per_play=0.05, epa_pass=0.15, epa_rush=-0.08)
    result = check_epa_range(conn, season=2023)
    assert result["out_of_range"]["silver.team_game_stats"] == 0
    assert result["passed"] is True


def test_check_epa_range_out_of_bounds(conn):
    _insert_silver_stats(conn, epa_per_play=3.5, epa_pass=0.1, epa_rush=-0.1)
    result = check_epa_range(conn, season=2023)
    assert result["out_of_range"]["silver.team_game_stats"] == 1
    assert result["passed"] is False


# ── run_all_checks ────────────────────────────────────────────────────────────

def test_run_all_checks_returns_passed_key(conn):
    result = run_all_checks(conn, season=2023)
    assert "passed" in result
    assert result["passed"] is False  # empty DB
    assert "row_counts" in result
    assert "null_rates" in result
    assert "epa_range" in result
    assert "completeness_score" in result
    assert "completeness_passed" in result
