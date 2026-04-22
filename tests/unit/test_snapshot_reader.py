"""Tests for time-travel-safe SnapshotReader."""
from datetime import datetime, timezone
import pandas as pd
from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.reader import SnapshotReader


def _seed_games(conn):
    early = datetime(2024, 1, 1, tzinfo=timezone.utc)
    late = datetime(2024, 9, 1, tzinfo=timezone.utc)
    conn.execute("""
        INSERT INTO silver.games (
            game_id, season, season_type, week, gameday,
            away_team, home_team, _silver_ts
        ) VALUES
            ('game1', 2023, 'REG', 1, '2023-09-07', 'KC', 'BAL', ?),
            ('game2', 2024, 'REG', 1, '2024-09-05', 'SF', 'NYJ', ?)
    """, [early, late])


def test_read_as_of_excludes_future_rows():
    conn = in_memory_connection()
    create_all_tables(conn)
    _seed_games(conn)

    cutoff = datetime(2024, 3, 1, tzinfo=timezone.utc)
    reader = SnapshotReader(cutoff, conn)
    df = reader.read_as_of("silver.games", ts_col="_silver_ts")
    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "game1"


def test_read_as_of_includes_all_past_rows():
    conn = in_memory_connection()
    create_all_tables(conn)
    _seed_games(conn)

    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    reader = SnapshotReader(cutoff, conn)
    df = reader.read_as_of("silver.games", ts_col="_silver_ts")
    assert len(df) == 2


def test_read_as_of_empty_before_any_data():
    conn = in_memory_connection()
    create_all_tables(conn)
    _seed_games(conn)

    cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)
    reader = SnapshotReader(cutoff, conn)
    df = reader.read_as_of("silver.games", ts_col="_silver_ts")
    assert df.empty
