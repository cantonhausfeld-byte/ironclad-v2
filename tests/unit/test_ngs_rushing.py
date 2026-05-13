"""Tests for NGS Rushing ingestor and ryoe_per_att_l4 gold feature."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.ingest.ngs_stats import _clean_rushing
from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.writer import BronzeWriter


def _make_raw_rushing(**kwargs):
    defaults = {
        "player_gsis_id":              ["00-0033873", "00-0033873"],
        "player_display_name":         ["Player A", "Player A"],
        "player_position":             ["RB", "RB"],
        "team_abbr":                   ["KC", "KC"],
        "season":                      [2024, 2024],
        "week":                        [1, 2],
        "season_type":                 ["REG", "REG"],
        "avg_rush_yards_over_expected": [0.8, -0.3],
        "avg_time_to_los":             [2.1, 2.3],
        "efficiency":                  [0.15, 0.10],
        "rush_attempts":               [18, 22],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ── Test 1: bronze.ngs_rushing table is created ───────────────────────────────

def test_ngs_rushing_table_created():
    conn = in_memory_connection()
    create_all_tables(conn)
    count = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'bronze' AND table_name = 'ngs_rushing'"
    ).fetchone()[0]
    assert count == 1, "bronze.ngs_rushing table not created"


# ── Test 2: _clean_rushing filters to REG season only ────────────────────────

def test_clean_ngs_rushing_filters_reg():
    raw = _make_raw_rushing(
        season_type=["REG", "POST"],
        week=[1, 22],
    )
    df = _clean_rushing(raw)
    assert len(df) == 1, f"Expected 1 REG row, got {len(df)}"
    assert df["season_type"].iloc[0] == "REG"


# ── Test 3: write_ngs_rushing upserts by (season, week, player_id) ───────────

def test_write_ngs_rushing_upserts():
    conn = in_memory_connection()
    create_all_tables(conn)
    writer = BronzeWriter(conn)

    df = _clean_rushing(_make_raw_rushing())
    writer.write_ngs_rushing(df)
    # Write same rows again — upsert should not add duplicates
    writer.write_ngs_rushing(df)

    count = conn.execute("SELECT COUNT(*) FROM bronze.ngs_rushing").fetchone()[0]
    assert count == 2, f"Expected 2 rows after upsert, got {count}"


# ── Test 4: gold.player_game_features has ryoe_per_att_l4 column ─────────────

def test_ryoe_per_att_l4_column_exists():
    conn = in_memory_connection()
    create_all_tables(conn)
    cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'gold' AND table_name = 'player_game_features'"
        ).fetchall()
    }
    assert "ryoe_per_att_l4" in cols, "ryoe_per_att_l4 missing from gold.player_game_features"
