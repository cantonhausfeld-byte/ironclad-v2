"""Tests for NGS ingestor and NGS feature integration."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from ironclad.ingest.ngs_stats import _clean_receiving, _clean_passing
from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables


# ── _clean_receiving ──────────────────────────────────────────────────────────

def _make_raw_receiving(**kwargs):
    defaults = {
        "season": [2024], "week": [5], "season_type": ["REG"],
        "player_gsis_id": ["P001"], "player_display_name": ["Test Player"],
        "player_position": ["WR"], "team_abbr": ["KC"],
        "avg_separation": [2.3], "avg_cushion": [4.1],
        "avg_intended_air_yards": [8.5], "avg_yac": [3.2],
        "avg_yac_above_expectation": [0.8],
        "targets": [8], "receptions": [6],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_clean_receiving_renames_player_id():
    raw = _make_raw_receiving()
    df = _clean_receiving(raw)
    assert "player_id" in df.columns
    assert "player_gsis_id" not in df.columns


def test_clean_receiving_filters_to_reg_season():
    raw = _make_raw_receiving(
        season=[2024, 2024], week=[5, 19],
        season_type=["REG", "POST"],
        player_gsis_id=["P001", "P001"],
        player_display_name=["TP", "TP"],
        player_position=["WR", "WR"],
        team_abbr=["KC", "KC"],
        avg_separation=[2.3, 3.0],
        avg_cushion=[4.1, 3.0],
        avg_intended_air_yards=[8.5, 9.0],
        avg_yac=[3.2, 3.5],
        avg_yac_above_expectation=[0.8, 0.9],
        targets=[8, 10], receptions=[6, 8],
    )
    df = _clean_receiving(raw)
    assert len(df) == 1
    assert df.iloc[0]["week"] == 5


def test_clean_receiving_casts_types():
    raw = _make_raw_receiving()
    df = _clean_receiving(raw)
    assert df["season"].dtype in (int, "int64", "int32")
    assert df["week"].dtype in (int, "int64", "int32")
    assert df["targets"].dtype in (int, "int64", "int32")
    assert df["avg_separation"].dtype in (float, "float64")


def test_clean_receiving_missing_optional_cols():
    """Missing optional NGS columns are filled with None, not an error."""
    raw = _make_raw_receiving()
    raw = raw.drop(columns=["avg_yac_above_expectation"])
    df = _clean_receiving(raw)
    assert "avg_yac_above_expectation" in df.columns
    assert df["avg_yac_above_expectation"].isna().all()


# ── _clean_passing ────────────────────────────────────────────────────────────

def _make_raw_passing(**kwargs):
    defaults = {
        "season": [2024], "week": [5], "season_type": ["REG"],
        "player_gsis_id": ["Q001"], "player_display_name": ["QB Test"],
        "team_abbr": ["KC"],
        "avg_time_to_throw": [2.6], "avg_intended_air_yards": [8.3],
        "aggressiveness": [14.2],
        "completion_percentage_above_expectation": [3.5],
        "attempts": [38], "completions": [26],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def test_clean_passing_renames_player_id():
    raw = _make_raw_passing()
    df = _clean_passing(raw)
    assert "player_id" in df.columns


def test_clean_passing_cpoe_is_float():
    raw = _make_raw_passing()
    df = _clean_passing(raw)
    assert df["completion_percentage_above_expectation"].dtype in (float, "float64")


# ── Schema tables exist ───────────────────────────────────────────────────────

def test_bronze_ngs_tables_created():
    conn = in_memory_connection()
    create_all_tables(conn)
    for tbl in ("ngs_receiving", "ngs_passing"):
        count = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'bronze' AND table_name = ?",
            [tbl],
        ).fetchone()[0]
        assert count == 1, f"bronze.{tbl} not created"


def test_gold_player_features_has_ngs_columns():
    conn = in_memory_connection()
    create_all_tables(conn)
    cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'gold' AND table_name = 'player_game_features'"
        ).fetchall()
    }
    for col in ("separation_l4", "yac_above_expected_l4", "cpoe_l4"):
        assert col in cols, f"Missing column: {col}"


# ── FeatureSnapshot NGS methods ───────────────────────────────────────────────

def _insert_game(conn, game_id="2023_05_KC_BAL", season=2023, week=5, gameday=date(2023, 10, 12)):
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday, away_team, home_team)
        VALUES (?, ?, 'REG', ?, ?, 'KC', 'BAL')
    """, [game_id, season, week, gameday])


def _insert_ngs_receiving(conn, player_id="P001", season=2023, week=4,
                           separation=2.5, yac_above=0.6):
    conn.execute("""
        INSERT INTO bronze.ngs_receiving
            (season, week, player_id, player_name, position, team, season_type,
             avg_separation, avg_yac_above_expectation, targets, receptions)
        VALUES (?, ?, ?, 'Test', 'WR', 'KC', 'REG', ?, ?, 8, 6)
    """, [season, week, player_id, separation, yac_above])


def test_snapshot_player_ngs_receiving_returns_data():
    from ironclad.features.snapshot import FeatureSnapshot
    conn = in_memory_connection()
    create_all_tables(conn)
    _insert_game(conn)
    _insert_ngs_receiving(conn, player_id="P001", season=2023, week=4, separation=2.5)

    snap = FeatureSnapshot(datetime(2023, 10, 12, 12, 0, tzinfo=timezone.utc), conn)
    df = snap.player_ngs_receiving("P001")
    assert not df.empty
    assert abs(float(df.iloc[0]["avg_separation"]) - 2.5) < 0.01


def test_snapshot_player_ngs_receiving_empty_for_unknown_player():
    from ironclad.features.snapshot import FeatureSnapshot
    conn = in_memory_connection()
    create_all_tables(conn)
    _insert_game(conn)
    snap = FeatureSnapshot(datetime(2023, 10, 12, 12, 0, tzinfo=timezone.utc), conn)
    df = snap.player_ngs_receiving("UNKNOWN")
    assert df.empty
