"""Tests for PFR pressure ingestor and silver enrichment."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.ingest.pfr_pressure import _clean_pfr_pressure
from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.silver import SilverTransformer
from ironclad.store.writer import BronzeWriter


def _make_raw_pfr(**kwargs):
    defaults = {
        "season": [2024, 2024],
        "week":   [5, 5],
        "game_id": ["2024_05_KC_BAL", "2024_05_KC_BAL"],
        "team":   ["KC", "BAL"],
        "season_type": ["REG", "REG"],
        "times_pressured":     [8,  6],
        "times_pressured_pct": [0.21, 0.16],
        "times_blitzed":       [10, 12],
        "times_hurried":       [5,  4],
        "times_hit":           [3,  2],
        "times_sacked":        [2,  1],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ── Test 1: bronze table created ──────────────────────────────────────────────

def test_pfr_pressure_table_created():
    conn = in_memory_connection()
    create_all_tables(conn)
    count = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'bronze' AND table_name = 'pfr_pressure_weekly'"
    ).fetchone()[0]
    assert count == 1, "bronze.pfr_pressure_weekly not created"


# ── Test 2: clean function filters to REG season ──────────────────────────────

def test_clean_pfr_pressure_filters_reg():
    raw = _make_raw_pfr(
        season=[2024, 2024, 2024],
        week=[5, 5, 19],
        game_id=["2024_05_KC_BAL", "2024_05_KC_BAL", "2024_19_XX_YY"],
        team=["KC", "BAL", "KC"],
        season_type=["REG", "REG", "POST"],
        times_pressured=[8, 6, 5],
        times_pressured_pct=[0.21, 0.16, 0.19],
        times_blitzed=[10, 12, 9],
        times_hurried=[5, 4, 3],
        times_hit=[3, 2, 1],
        times_sacked=[2, 1, 0],
    )
    df = _clean_pfr_pressure(raw)
    assert len(df) == 2, "POST row should be filtered out"
    assert all(df["week"] < 19)


# ── Test 3: silver enrichment updates pressure_rate ───────────────────────────

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


def test_silver_enrichment_updates_pressure():
    conn = in_memory_connection()
    create_all_tables(conn)
    _insert_silver_game(conn)
    _insert_silver_team_stats(conn, "2024_05_KC_BAL", "KC")

    # Insert PFR row
    conn.execute("""
        INSERT INTO bronze.pfr_pressure_weekly
            (season, week, game_id, team, times_pressured, times_pressured_pct,
             times_blitzed, times_hurried, times_hit, times_sacked)
        VALUES (2024, 5, '2024_05_KC_BAL', 'KC', 8, 0.21, 10, 5, 3, 2)
    """)

    transformer = SilverTransformer(conn)
    transformer._enrich_with_pfr(2024)

    row = conn.execute(
        "SELECT pressure_rate FROM silver.team_game_stats "
        "WHERE game_id = '2024_05_KC_BAL' AND team = 'KC'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 0.21) < 0.001, f"Expected pressure_rate ~0.21, got {row[0]}"


# ── Test 4: gold columns exist in schema ──────────────────────────────────────

def test_gold_pressure_features_present():
    conn = in_memory_connection()
    create_all_tables(conn)
    cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'gold' AND table_name = 'team_game_features'"
        ).fetchall()
    }
    for col in ("off_pressure_rate_l4", "def_pressure_rate_allowed_l4",
                "off_blitz_rate_l4", "def_blitz_rate_allowed_l4"):
        assert col in cols, f"Missing gold column: {col}"
