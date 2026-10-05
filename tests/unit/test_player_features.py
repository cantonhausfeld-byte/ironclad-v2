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


# ── Weather context tests ─────────────────────────────────────────────────────

def test_player_game_features_has_weather_columns():
    """gold.player_game_features schema includes weather/venue columns."""
    conn = _conn()
    cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'gold' AND table_name = 'player_game_features'"
        ).fetchall()
    }
    for col in ("wind_mph", "temp_f", "is_dome", "altitude_ft"):
        assert col in cols, f"{col} missing from gold.player_game_features"


def test_player_features_weather_populated_from_game():
    """build_for_game propagates weather fields from silver.games into feature rows."""
    from ironclad.features.player_features import PlayerFeatureBuilder

    conn = _conn()
    game_id = "2024_10_KC_DEN"
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday, away_team, home_team,
             temp_f, wind_mph, is_dome, altitude_ft)
        VALUES (?, 2024, 'REG', 10, '2024-11-10', 'KC', 'DEN',
                28.0, 12.5, false, 5280)
    """, [game_id])
    conn.execute("""
        INSERT INTO silver.player_weekly_status
            (season, week, player_id, player_name, team, position, availability)
        VALUES (2024, 10, 'RB1', 'Test RB', 'KC', 'RB', 1.0)
    """)

    cutoff = datetime(2024, 11, 10, 20, 0, tzinfo=timezone.utc)
    builder = PlayerFeatureBuilder(conn)
    df = builder.build_for_game(game_id, cutoff)

    assert not df.empty, "Expected at least one player row"
    row = df.iloc[0]
    assert row["temp_f"] == pytest.approx(28.0)
    assert row["wind_mph"] == pytest.approx(12.5)
    assert row["is_dome"] == False  # noqa: E712 (strict: None/NaN must fail)
    assert row["altitude_ft"] == 5280


# ── Team code normalization regression ───────────────────────────────────────

def test_supplement_from_roster_normalizes_team_code():
    """bronze.rosters may use 'LA' (Rams transition); roster lookup must normalize."""
    from datetime import timezone

    from ironclad.features.player_features import _supplement_from_roster
    from ironclad.features.snapshot import FeatureSnapshot

    conn = _conn()
    cutoff = datetime(2099, 1, 1, tzinfo=timezone.utc)
    snap = FeatureSnapshot(cutoff, conn)

    conn.execute("""
        INSERT INTO bronze.rosters
            (season, week, player_id, player_name, team, position, _ingest_ts)
        VALUES (2024, 1, 'P999', 'Test Ram', 'LA', 'WR',
                '2024-01-01 00:00:00+00')
    """)

    # Query with normalized code 'LAR' — must find the 'LA'-coded roster entry
    result = _supplement_from_roster(pd.DataFrame(), "LAR", 2024, 1, snap)
    assert not result.empty, "Roster lookup with 'LA'→'LAR' normalization should find the player"
    assert result.iloc[0]["team"] == "LAR"


def test_supplement_from_roster_only_adds_active_players():
    from datetime import timezone

    from ironclad.features.player_features import _supplement_from_roster
    from ironclad.features.snapshot import FeatureSnapshot

    conn = _conn()
    for pid, status in [("ACT1", "ACT"), ("IR1", "RES"), ("PS1", "DEV"), ("INA1", "INA"), ("CUT1", "CUT")]:
        conn.execute(
            "INSERT INTO bronze.rosters (season, week, player_id, player_name, team, position, "
            "status, _ingest_ts) VALUES (2025, 5, ?, ?, 'KC', 'WR', ?, '2025-10-01 00:00:00+00')",
            [pid, pid, status],
        )
    snap = FeatureSnapshot(datetime(2099, 1, 1, tzinfo=timezone.utc), conn)
    result = _supplement_from_roster(pd.DataFrame(), "KC", 2025, 5, snap)
    assert list(result["player_id"]) == ["ACT1"]


def test_drop_inactive_removes_injured_reserve_and_practice_squad():
    from datetime import timezone

    from ironclad.features.player_features import _drop_inactive
    from ironclad.features.snapshot import FeatureSnapshot

    conn = _conn()
    for pid, status in [("STAR", "RES"), ("PS", "DEV"), ("WR1", "ACT")]:
        conn.execute(
            "INSERT INTO bronze.rosters (season, week, player_id, player_name, team, position, "
            "status, _ingest_ts) VALUES (2025, 10, ?, ?, 'MIA', 'WR', ?, '2025-11-01 00:00:00+00')",
            [pid, pid, status],
        )
    players = pd.DataFrame({"player_id": ["STAR", "PS", "WR1", "NOROSTER"], "team": "MIA"})
    snap = FeatureSnapshot(datetime(2099, 1, 1, tzinfo=timezone.utc), conn)
    kept = _drop_inactive(players, "MIA", 2025, 10, snap)
    assert list(kept["player_id"]) == ["WR1", "NOROSTER"]


def test_rebuild_removes_players_who_no_longer_qualify():
    from ironclad.features.player_features import PlayerFeatureBuilder

    conn = _conn()
    conn.execute(
        "INSERT INTO silver.games (game_id, season, season_type, week, gameday, home_team, away_team) "
        "VALUES ('2025_10_BUF_MIA', 2025, 'REG', 10, '2025-11-09', 'MIA', 'BUF')"
    )
    conn.execute(
        "INSERT INTO gold.player_game_features (game_id, player_id, player_name, team, opponent, "
        "season, week, position, is_home, cutoff_ts, feature_version) "
        "VALUES ('2025_10_BUF_MIA', 'STALE', 'Stale', 'MIA', 'BUF', 2025, 10, 'WR', true, NOW(), 'v')"
    )
    PlayerFeatureBuilder(conn).build_for_game("2025_10_BUF_MIA", datetime(2025, 11, 9, 17, 30))
    ids = conn.execute(
        "SELECT player_id FROM gold.player_game_features WHERE game_id = '2025_10_BUF_MIA'"
    ).fetchall()
    assert ("STALE",) not in ids
