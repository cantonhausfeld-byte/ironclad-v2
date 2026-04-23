"""Tests for DuckDB schema creation."""
import pytest
from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables


def test_create_all_tables_runs_without_error():
    conn = in_memory_connection()
    create_all_tables(conn)


def test_all_expected_tables_exist():
    conn = in_memory_connection()
    create_all_tables(conn)
    expected = [
        "bronze.schedules", "bronze.play_by_play", "bronze.rosters",
        "bronze.injuries", "bronze.depth_charts", "bronze.odds",
        "bronze.weather", "bronze.stadiums",
        "bronze.snap_counts", "bronze.player_stats_weekly",
        "silver.games", "silver.team_game_stats", "silver.player_game_stats",
        "silver.player_weekly_status",
        "gold.team_game_features", "gold.player_game_features",
        "gold.model_predictions",
    ]
    for table in expected:
        schema, tbl = table.split(".")
        result = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [schema, tbl],
        ).fetchone()[0]
        assert result == 1, f"Table {table} not found"


def test_idempotent_creation():
    conn = in_memory_connection()
    create_all_tables(conn)
    create_all_tables(conn)  # should not raise
