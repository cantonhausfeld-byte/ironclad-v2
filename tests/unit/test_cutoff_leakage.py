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


# ── Night games and weekly (season, week) tables ──────────────────────────────
#
# nflverse gamedays and kickoff times are US Eastern. An 8:20pm ET Sunday game
# is past midnight UTC, so comparing gamedays against the cutoff's UTC date
# treated the game being predicted (and every other same-day game) as played.

def test_kickoff_parse_handles_daylight_saving():
    from ironclad.features.team_features import _parse_kickoff
    # 1:00pm EDT (UTC-4) in September, 1:00pm EST (UTC-5) in December
    assert _parse_kickoff("2024-09-22", "13:00") == datetime(2024, 9, 22, 17, 0, tzinfo=timezone.utc)
    assert _parse_kickoff("2024-12-22", "13:00") == datetime(2024, 12, 22, 18, 0, tzinfo=timezone.utc)


def test_night_game_does_not_see_itself(cutoff_db):
    from datetime import timedelta

    from ironclad.config import KNOWLEDGE_CUTOFF_MARGIN_MINUTES
    from ironclad.features.team_features import _parse_kickoff

    for gametime in ("20:20", "20:40"):
        cutoff = _parse_kickoff("2024-09-22", gametime) - timedelta(
            minutes=KNOWLEDGE_CUTOFF_MARGIN_MINUTES
        )
        snap = FeatureSnapshot(cutoff, cutoff_db)
        assert "2024_03_BBB_AAA" not in set(snap._past_game_ids()), gametime
        assert list(snap.team_recent_games("AAA")["game_id"]) == ["2024_01_BBB_AAA"]


def test_week_counts_as_past_only_when_all_its_games_are_played(conn):
    # Week 3: Thursday game on 9/19, Sunday game on 9/22
    _add_game(conn, "2024_03_CCC_DDD", 2024, 3, "2024-09-19", "DDD", "CCC")
    _add_game(conn, "2024_03_BBB_AAA", 2024, 3, "2024-09-22", "AAA", "BBB")
    _add_game(conn, "2024_02_BBB_AAA", 2024, 2, "2024-09-15", "AAA", "BBB")

    sunday_cutoff = datetime(2024, 9, 22, 16, 30, tzinfo=timezone.utc)
    weeks = FeatureSnapshot(sunday_cutoff, conn)._past_week_pairs()
    assert sorted(map(tuple, weeks.values.tolist())) == [(2024, 2)]
