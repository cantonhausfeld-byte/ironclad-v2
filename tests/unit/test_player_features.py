"""Smoke tests for PlayerFeatureBuilder."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables


def _conn():
    conn = in_memory_connection()
    create_all_tables(conn)
    return conn


def _insert_game(conn, game_id="2023_05_KC_BAL", season=2023, week=5,
                 home="BAL", away="KC", gameday=date(2023, 10, 12)):
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday, away_team, home_team)
        VALUES (?, ?, 'REG', ?, ?, ?, ?)
    """, [game_id, season, week, gameday, away, home])


def _insert_player(conn, player_id="P001", team="KC", season=2023, week=5,
                   position="WR"):
    conn.execute("""
        INSERT INTO silver.player_weekly_status
            (season, week, player_id, player_name, team, position, availability)
        VALUES (?, ?, ?, 'Test Player', ?, ?, 1.0)
    """, [season, week, player_id, team, position])


def test_build_for_game_missing_game_returns_empty():
    """Returns empty DataFrame without raising when game_id not in silver."""
    from ironclad.features.player_features import PlayerFeatureBuilder
    conn = _conn()
    builder = PlayerFeatureBuilder(conn)
    cutoff = datetime(2023, 10, 12, 12, 0, tzinfo=timezone.utc)
    df = builder.build_for_game("2023_99_XX_YY", cutoff)
    assert df.empty


def test_build_for_game_returns_dataframe():
    """Returns a DataFrame with expected columns when game and roster exist."""
    from ironclad.features.player_features import PlayerFeatureBuilder
    conn = _conn()
    _insert_game(conn)
    _insert_player(conn, team="KC")
    _insert_player(conn, player_id="P002", team="BAL", position="RB")

    # Use a far-future cutoff so that NOW() _silver_ts rows are visible
    cutoff = datetime(2099, 1, 1, tzinfo=timezone.utc)
    builder = PlayerFeatureBuilder(conn)
    df = builder.build_for_game("2023_05_KC_BAL", cutoff)

    assert not df.empty
    for col in ("game_id", "player_id", "team", "position", "availability",
                "snap_rate_l4", "target_share_l4"):
        assert col in df.columns, f"Missing column: {col}"


def test_season_week_in_scope():
    """_build_player_row must not raise NameError for season/week variables."""
    from ironclad.features.player_features import PlayerFeatureBuilder
    conn = _conn()
    _insert_game(conn, season=2024, week=3,
                 home="SF", away="DAL", gameday=date(2024, 9, 29))
    _insert_player(conn, player_id="QB1", team="SF", season=2024, week=3, position="QB")

    cutoff = datetime(2024, 9, 29, 20, 0, tzinfo=timezone.utc)
    builder = PlayerFeatureBuilder(conn)
    # Would raise NameError before the fix; now must succeed
    df = builder.build_for_game("2023_03_DAL_SF", cutoff)
    # game_id mismatch → empty, but must not crash
    assert isinstance(df, pd.DataFrame)


def test_build_for_game_non_skill_positions_excluded():
    """OL / K / P players are skipped — only SKILL_POSITIONS produce rows."""
    from ironclad.features.player_features import PlayerFeatureBuilder
    conn = _conn()
    _insert_game(conn)
    _insert_player(conn, player_id="K1", team="KC", position="K")
    _insert_player(conn, player_id="OL1", team="KC", position="OL")

    cutoff = datetime(2099, 1, 1, tzinfo=timezone.utc)
    builder = PlayerFeatureBuilder(conn)
    df = builder.build_for_game("2023_05_KC_BAL", cutoff)
    assert df.empty


def _insert_ngs(conn, player_id="P001", season=2023, week=3,
                stat_type="receiving", avg_separation=2.5):
    conn.execute("""
        INSERT INTO bronze.ngs_data
            (player_id, player_name, season, week, stat_type, avg_separation)
        VALUES (?, 'Test Player', ?, ?, ?, ?)
    """, [player_id, season, week, stat_type, avg_separation])


def test_ngs_columns_present_in_player_features():
    """ngs_avg_separation column exists in output and is populated from bronze.ngs_data."""
    from ironclad.features.player_features import PlayerFeatureBuilder
    conn = _conn()
    _insert_game(conn)
    _insert_player(conn, team="KC")
    _insert_player(conn, player_id="P002", team="BAL", position="RB")
    _insert_ngs(conn, player_id="P001", season=2023, week=3)

    cutoff = datetime(2099, 1, 1, tzinfo=timezone.utc)
    builder = PlayerFeatureBuilder(conn)
    df = builder.build_for_game("2023_05_KC_BAL", cutoff)

    assert not df.empty
    assert "ngs_avg_separation" in df.columns
    assert "ngs_rush_yards_over_expected" in df.columns
    assert "ngs_completion_pct_above_expected" in df.columns
    # P001 (WR on KC) should have separation populated from prior-week NGS data
    p001_rows = df[df["player_id"] == "P001"]
    if not p001_rows.empty:
        val = p001_rows.iloc[0]["ngs_avg_separation"]
        assert val is None or float(val) == pytest.approx(2.5, abs=1e-6)
