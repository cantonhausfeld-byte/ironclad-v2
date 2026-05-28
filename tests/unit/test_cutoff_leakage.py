"""Knowledge-cutoff leakage guard.

FeatureSnapshot enforces the cutoff via game *date* (gameday < cutoff date), not
via _ingest_ts: a bulk backfill stamps every row with _ingest_ts = NOW(), so an
ingest-time filter would erase all historical data. The substantive guarantee
for backtest validity is therefore that no game played on/after the cutoff can
leak into the features built for that cutoff. These tests lock that in.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ironclad.features.snapshot import FeatureSnapshot


def _add_game(conn, game_id, season, week, gameday, home, away):
    conn.execute(
        "INSERT INTO silver.games (game_id, season, season_type, week, gameday, "
        "away_team, home_team) VALUES (?, ?, 'REG', ?, ?, ?, ?)",
        [game_id, season, week, gameday, away, home],
    )


def _add_team_stat(conn, game_id, team, season, week):
    conn.execute(
        "INSERT INTO silver.team_game_stats (game_id, season, week, team, "
        "opponent, is_home) VALUES (?, ?, ?, ?, 'BBB', true)",
        [game_id, season, week, team],
    )


def _add_player_stat(conn, game_id, player_id, season, week):
    conn.execute(
        "INSERT INTO silver.player_game_stats (game_id, season, week, player_id, "
        "player_name, team, opponent, position, is_home) "
        "VALUES (?, ?, ?, ?, 'Player One', 'AAA', 'BBB', 'WR', true)",
        [game_id, season, week, player_id],
    )


@pytest.fixture
def cutoff_db(conn):
    # Past game (week 1) and a future game (week 3); cutoff sits between them.
    _add_game(conn, "2024_01_BBB_AAA", 2024, 1, "2024-09-08", "AAA", "BBB")
    _add_game(conn, "2024_03_BBB_AAA", 2024, 3, "2024-09-22", "AAA", "BBB")
    for gid, wk in [("2024_01_BBB_AAA", 1), ("2024_03_BBB_AAA", 3)]:
        _add_team_stat(conn, gid, "AAA", 2024, wk)
        _add_player_stat(conn, gid, "P1", 2024, wk)
    return conn


# Cutoff = kickoff of the week-3 game minus margin → strictly after week 1,
# strictly before week 3's gameday.
CUTOFF = datetime(2024, 9, 15, 12, 0, tzinfo=timezone.utc)


def test_past_game_ids_excludes_future(cutoff_db):
    snap = FeatureSnapshot(CUTOFF, cutoff_db)
    ids = set(snap._past_game_ids())
    assert "2024_01_BBB_AAA" in ids
    assert "2024_03_BBB_AAA" not in ids


def test_team_recent_games_excludes_future(cutoff_db):
    snap = FeatureSnapshot(CUTOFF, cutoff_db)
    games = snap.team_recent_games("AAA", n=4)
    assert list(games["game_id"]) == ["2024_01_BBB_AAA"]


def test_player_recent_games_excludes_future(cutoff_db):
    snap = FeatureSnapshot(CUTOFF, cutoff_db)
    games = snap.player_recent_games("P1", n=4)
    assert list(games["game_id"]) == ["2024_01_BBB_AAA"]


def test_cutoff_after_all_games_includes_all(cutoff_db):
    late = datetime(2024, 12, 1, tzinfo=timezone.utc)
    snap = FeatureSnapshot(late, cutoff_db)
    assert len(snap.team_recent_games("AAA", n=4)) == 2


def test_cutoff_before_all_games_includes_none(cutoff_db):
    early = datetime(2024, 9, 1, tzinfo=timezone.utc)
    snap = FeatureSnapshot(early, cutoff_db)
    assert snap.team_recent_games("AAA", n=4).empty
    assert snap.player_recent_games("P1", n=4).empty
