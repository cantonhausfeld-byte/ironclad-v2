"""Tests for TargetBackfiller."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.targets import TargetBackfiller


def _conn():
    conn = in_memory_connection()
    create_all_tables(conn)
    return conn


def _insert_game(conn, game_id="2024_01_KC_BAL", season=2024, week=1,
                 home_score=27, away_score=20):
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday,
             away_team, home_team,
             away_score, home_score, total_score, home_margin, home_win)
        VALUES (?, ?, 'REG', ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [game_id, season, week, date(2024, 9, 5),
          "KC", "BAL",
          away_score, home_score,
          home_score + away_score,
          home_score - away_score,
          home_score > away_score])


def _insert_team_stats(conn, game_id="2024_01_KC_BAL", team="BAL",
                       season=2024, week=1):
    conn.execute("""
        INSERT INTO silver.team_game_stats
            (game_id, season, week, team, opponent, is_home,
             points_scored, total_yards, pass_rate)
        VALUES (?, ?, ?, ?, ?, true, 27, 350.0, 0.58)
    """, [game_id, season, week, team, "KC"])


def _insert_team_feature(conn, game_id="2024_01_KC_BAL", team="BAL",
                         season=2024, week=1):
    cutoff = datetime(2024, 9, 5, 18, 0, tzinfo=timezone.utc)
    conn.execute("""
        INSERT INTO gold.team_game_features
            (game_id, season, week, team, opponent, is_home,
             cutoff_ts, feature_version)
        VALUES (?, ?, ?, ?, ?, true, ?, '1.0')
    """, [game_id, season, week, team, "KC", cutoff])


def _insert_player_stats(conn, game_id="2024_01_KC_BAL", player_id="P001",
                          season=2024, week=1):
    conn.execute("""
        INSERT INTO silver.player_game_stats
            (game_id, season, week, player_id, player_name,
             team, opponent, position, is_home,
             targets, carries, receptions, rec_yards, rush_yards, total_tds)
        VALUES (?, ?, ?, ?, 'Test Player', 'BAL', 'KC', 'WR', true,
                8, 0, 6, 92, 0, 1)
    """, [game_id, season, week, player_id])


def _insert_player_feature(conn, game_id="2024_01_KC_BAL", player_id="P001",
                            season=2024, week=1):
    cutoff = datetime(2024, 9, 5, 18, 0, tzinfo=timezone.utc)
    conn.execute("""
        INSERT INTO gold.player_game_features
            (game_id, season, week, player_id, player_name,
             team, opponent, position, is_home, cutoff_ts, feature_version,
             availability)
        VALUES (?, ?, ?, ?, 'Test Player', 'BAL', 'KC', 'WR', true,
                ?, '1.0', 1.0)
    """, [game_id, season, week, player_id, cutoff])


# ── team targets ──────────────────────────────────────────────────────────────

def test_team_targets_filled():
    """Backfiller writes all six team target columns including new win/margin/total."""
    conn = _conn()
    _insert_game(conn)
    _insert_team_stats(conn)
    _insert_team_feature(conn)

    counts = TargetBackfiller(conn).run([2024])
    assert counts["team_targets"] == 1

    row = conn.execute(
        "SELECT * FROM gold.team_game_features WHERE game_id = '2024_01_KC_BAL'"
    ).df().iloc[0]

    assert row["target_points_scored"] == pytest.approx(27.0)
    assert row["target_yards_total"] == pytest.approx(350.0)
    assert row["target_home_win"] == True
    assert row["target_home_margin"] == 7        # 27 - 20
    assert row["target_total_score"] == 47       # 27 + 20


def test_team_targets_idempotent():
    """Running backfiller twice does not double-count rows (NULL guard)."""
    conn = _conn()
    _insert_game(conn)
    _insert_team_stats(conn)
    _insert_team_feature(conn)

    TargetBackfiller(conn).run([2024])
    counts2 = TargetBackfiller(conn).run([2024])
    assert counts2["team_targets"] == 0


def test_team_targets_empty_gold():
    """Returns 0 when gold table has no rows."""
    conn = _conn()
    _insert_game(conn)
    _insert_team_stats(conn)
    counts = TargetBackfiller(conn).run([2024])
    assert counts["team_targets"] == 0


# ── player targets ────────────────────────────────────────────────────────────

def test_player_targets_filled():
    """Backfiller writes all six player target columns."""
    conn = _conn()
    _insert_game(conn)
    _insert_player_stats(conn)
    _insert_player_feature(conn)

    counts = TargetBackfiller(conn).run([2024])
    assert counts["player_targets"] == 1

    row = conn.execute(
        "SELECT * FROM gold.player_game_features WHERE player_id = 'P001'"
    ).df().iloc[0]

    assert row["target_targets"] == pytest.approx(8.0)
    assert row["target_receptions"] == pytest.approx(6.0)
    assert row["target_rec_yards"] == pytest.approx(92.0)
    assert row["target_total_tds"] == pytest.approx(1.0)


def test_backfiller_no_seasons_filter():
    """run(None) processes all seasons."""
    conn = _conn()
    _insert_game(conn)
    _insert_team_stats(conn)
    _insert_team_feature(conn)

    counts = TargetBackfiller(conn).run(None)
    assert counts["team_targets"] == 1
