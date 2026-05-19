"""Tests for FTN charting ingestor and silver enrichment."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.ingest.ftn_charting import _clean_ftn_charting
from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.silver import SilverTransformer
from ironclad.store.writer import BronzeWriter


def _make_raw_ftn(**kwargs):
    defaults = {
        "game_id":        ["2024_05_KC_BAL"] * 4,
        "season":         [2024] * 4,
        "week":           [5] * 4,
        "n_blitzers":     [4, 5, 3, 6],
        "n_pass_rushers": [4, 4, 4, 5],
        "n_defense_box":  [6, 7, 6, 6],
        "is_play_action": [1, 0, 0, 1],
        "is_motion":      [1, 1, 0, 1],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ── Test 1: bronze table created ──────────────────────────────────────────────

def test_ftn_table_created():
    conn = in_memory_connection()
    create_all_tables(conn)
    count = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'bronze' AND table_name = 'ftn_charting'"
    ).fetchone()[0]
    assert count == 1, "bronze.ftn_charting not created"


# ── Test 2: clean function filters REG rows ───────────────────────────────────

def test_clean_ftn_filters_reg():
    """FTN data has no season_type column; all 4 play-level rows should be kept."""
    raw = _make_raw_ftn()
    df = _clean_ftn_charting(raw)
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"
    assert "team" not in df.columns, "team column should not be in cleaned FTN data"


# ── Test 3: silver enrichment updates play_action_rate ────────────────────────

def _insert_silver_game(conn, game_id="2024_05_KC_BAL", season=2024, week=5):
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday, away_team, home_team)
        VALUES (?, ?, 'REG', ?, '2024-10-07', 'KC', 'BAL')
    """, [game_id, season, week])


def _insert_silver_team_stats(conn, game_id, team, season=2024, week=5):
    conn.execute("""
        INSERT INTO silver.team_game_stats
            (game_id, season, week, team, opponent, is_home,
             pass_attempts, plays_total)
        VALUES (?, ?, ?, ?, 'OPP', false, 35, 60)
    """, [game_id, season, week, team])


def test_ftn_silver_enrich():
    conn = in_memory_connection()
    create_all_tables(conn)
    _insert_silver_game(conn)
    _insert_silver_team_stats(conn, "2024_05_KC_BAL", "KC")

    # Insert FTN rows — 2 plays, 1 play-action = 50% rate (no team column)
    conn.execute("""
        INSERT INTO bronze.ftn_charting
            (game_id, season, week, n_blitzers, n_defense_box,
             is_play_action, is_motion)
        VALUES
            ('2024_05_KC_BAL', 2024, 5, 4, 6, 1, 0),
            ('2024_05_KC_BAL', 2024, 5, 5, 7, 0, 1)
    """)

    transformer = SilverTransformer(conn)
    transformer._enrich_with_ftn(2024)

    row = conn.execute(
        "SELECT play_action_rate FROM silver.team_game_stats "
        "WHERE game_id = '2024_05_KC_BAL' AND team = 'KC'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 0.50) < 0.01, f"Expected play_action_rate ~0.50, got {row[0]}"


# ── Test 4: gold columns exist in schema ──────────────────────────────────────

def test_gold_ftn_columns_exist():
    conn = in_memory_connection()
    create_all_tables(conn)
    cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'gold' AND table_name = 'team_game_features'"
        ).fetchall()
    }
    for col in ("game_play_action_rate_l4", "game_avg_blitzers_l4", "game_avg_box_count_l4"):
        assert col in cols, f"Missing gold column: {col}"
