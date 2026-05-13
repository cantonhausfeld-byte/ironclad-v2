"""Tests for player ID mapping ingestor and bronze.player_id_mapping schema."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.ingest.player_ids import _clean_player_ids
from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.writer import BronzeWriter


def _make_raw(**kwargs):
    defaults = {
        "gsis_id":     ["00-0033873", "00-0030506"],
        "pfr_id":      ["MahoP00",    "KelcT00"],
        "nfl_id":      ["32004436",   "32004310"],
        "espn_id":     ["3139477",    "3054145"],
        "player_name": ["Patrick Mahomes", "Travis Kelce"],
        "position":    ["QB",          "TE"],
        "team":        ["KC",          "KC"],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ── Test 1: table created ─────────────────────────────────────────────────────

def test_player_id_table_created():
    conn = in_memory_connection()
    create_all_tables(conn)
    count = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'bronze' AND table_name = 'player_id_mapping'"
    ).fetchone()[0]
    assert count == 1, "bronze.player_id_mapping not created"


# ── Test 2: upsert is idempotent ──────────────────────────────────────────────

def test_write_player_id_mapping_upserts():
    conn = in_memory_connection()
    create_all_tables(conn)
    writer = BronzeWriter(conn=conn)

    df = _make_raw()
    writer.write_player_id_mapping(df)
    writer.write_player_id_mapping(df)  # second write with same gsis_ids → no duplicates

    n = conn.execute("SELECT count(*) FROM bronze.player_id_mapping").fetchone()[0]
    assert n == 2, f"Expected 2 rows after two identical writes, got {n}"


# ── Test 3: _clean_player_ids selects expected columns ────────────────────────

def test_clean_player_ids_selects_columns():
    raw = _make_raw()
    # Add an extra column that should be dropped
    raw["db_season"] = [2024, 2024]
    df = _clean_player_ids(raw)

    assert "gsis_id" in df.columns
    assert "pfr_id" in df.columns
    assert "player_name" in df.columns
    assert "db_season" not in df.columns, "Extra column should be dropped"
    assert len(df) == 2


def test_clean_player_ids_drops_null_gsis():
    raw = _make_raw(gsis_id=["00-0033873", None])
    df = _clean_player_ids(raw)
    assert len(df) == 1
    assert df.iloc[0]["gsis_id"] == "00-0033873"


def test_clean_player_ids_deduplicates():
    raw = _make_raw(
        gsis_id=["00-0033873", "00-0033873"],
        pfr_id=["MahoP00_old", "MahoP00"],
    )
    df = _clean_player_ids(raw)
    assert len(df) == 1
    assert df.iloc[0]["pfr_id"] == "MahoP00"
