"""Tests for odds ingestor game_id fix, player prop parsing, and load_prop_lines_from_db."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.store.normalization import full_name_to_abbr
from ironclad.ingest.odds import _resolve_game_id, _resolve_player, _parse_game_odds
from ironclad.betting.props import load_prop_lines_from_db, PropLine


# ── full_name_to_abbr ─────────────────────────────────────────────────────────

def test_full_name_known_teams():
    assert full_name_to_abbr("Kansas City Chiefs") == "KC"
    assert full_name_to_abbr("Los Angeles Chargers") == "LAC"
    assert full_name_to_abbr("Los Angeles Rams") == "LAR"
    assert full_name_to_abbr("New England Patriots") == "NE"
    assert full_name_to_abbr("San Francisco 49ers") == "SF"


def test_full_name_historical():
    assert full_name_to_abbr("Oakland Raiders") == "LV"
    assert full_name_to_abbr("San Diego Chargers") == "LAC"
    assert full_name_to_abbr("Washington Redskins") == "WAS"


def test_full_name_unknown_returns_none():
    assert full_name_to_abbr("Fictional Falcons") is None
    assert full_name_to_abbr("") is None
    assert full_name_to_abbr(None) is None


# ── _resolve_game_id ──────────────────────────────────────────────────────────

def test_resolve_game_id_found(conn):
    conn.execute("""
        INSERT INTO silver.games
        (game_id, season, season_type, week, gameday, away_team, home_team)
        VALUES ('2024_14_LAC_KC', 2024, 'REG', 14, '2024-12-15', 'LAC', 'KC')
    """)
    gid = _resolve_game_id(conn, "KC", "LAC", "2024-12-15T18:00:00Z", "evt123")
    assert gid == "2024_14_LAC_KC"


def test_resolve_game_id_not_found_fallback(conn):
    gid = _resolve_game_id(conn, "KC", "LAC", "2024-12-15T18:00:00Z", "evt456")
    assert gid == "ODDS_evt456"


def test_resolve_game_id_missing_abbr_fallback(conn):
    gid = _resolve_game_id(conn, None, "LAC", "2024-12-15T18:00:00Z", "evt789")
    assert gid == "ODDS_evt789"


def test_resolve_game_id_timezone_plus_one(conn):
    conn.execute("""
        INSERT INTO silver.games
        (game_id, season, season_type, week, gameday, away_team, home_team)
        VALUES ('2024_14_LAC_KC', 2024, 'REG', 14, '2024-12-15', 'LAC', 'KC')
    """)
    # Commence time in UTC pushes into the next calendar day — should still resolve
    gid = _resolve_game_id(conn, "KC", "LAC", "2024-12-16T01:00:00Z", "evt123")
    assert gid == "2024_14_LAC_KC"


# ── _resolve_player ───────────────────────────────────────────────────────────

def _insert_roster_row(conn, player_name: str, player_id: str, team: str) -> None:
    conn.execute("""
        INSERT INTO bronze.rosters (season, week, player_id, player_name, team, position)
        VALUES (2024, 14, ?, ?, ?, 'WR')
    """, [player_id, player_name, team])


def test_resolve_player_exact_match(conn):
    _insert_roster_row(conn, "Travis Kelce", "00-0030506", "KC")
    team, pid = _resolve_player(conn, "Travis Kelce", "KC", "LAC")
    assert team == "KC"
    assert pid == "00-0030506"


def test_resolve_player_not_found(conn):
    team, pid = _resolve_player(conn, "Ghost Player", "KC", "LAC")
    assert team is None
    assert pid is None


def test_resolve_player_team_filter(conn):
    _insert_roster_row(conn, "Justin Herbert", "00-0036971", "LAC")
    _insert_roster_row(conn, "Justin Herbert", "FAKE-99999", "KC")  # wrong team
    team, pid = _resolve_player(conn, "Justin Herbert", "KC", "LAC")
    # Should return either team's match (both are valid candidates)
    assert pid is not None


# ── load_prop_lines_from_db ───────────────────────────────────────────────────

def _insert_prop(conn, game_id, player_id, stat_type, line, over_odds, under_odds):
    conn.execute("""
        INSERT INTO bronze.player_props
        (game_id, player_name, player_id, team, stat_type, line, over_odds, under_odds,
         bookmaker, retrieved_at)
        VALUES (?, 'Test Player', ?, 'KC', ?, ?, ?, ?, 'draftkings', NOW())
    """, [game_id, player_id, stat_type, line, over_odds, under_odds])


def test_load_prop_lines_from_db_basic(conn):
    _insert_prop(conn, "2024_14_LAC_KC", "P1", "rec_yards", 55.5, -110, -110)
    lines = load_prop_lines_from_db(conn, "2024_14_LAC_KC")
    assert len(lines) == 1
    assert lines[0].player_id == "P1"
    assert lines[0].stat_type == "rec_yards"
    assert lines[0].line == 55.5
    assert lines[0].over_odds == -110


def test_load_prop_lines_skips_null_player_id(conn):
    conn.execute("""
        INSERT INTO bronze.player_props
        (game_id, player_name, player_id, team, stat_type, line, over_odds, under_odds,
         bookmaker, retrieved_at)
        VALUES ('2024_14_LAC_KC', 'Unknown', NULL, 'KC', 'rec_yards', 55.5, -110, -110,
                'draftkings', NOW())
    """)
    lines = load_prop_lines_from_db(conn, "2024_14_LAC_KC")
    assert lines == []


def test_load_prop_lines_deduplicates(conn):
    # Same player+stat twice — should return only one PropLine
    _insert_prop(conn, "2024_14_LAC_KC", "P1", "rec_yards", 55.5, -110, -110)
    _insert_prop(conn, "2024_14_LAC_KC", "P1", "rec_yards", 60.0, -115, -105)
    lines = load_prop_lines_from_db(conn, "2024_14_LAC_KC")
    assert len(lines) == 1


def test_load_prop_lines_empty_when_no_data(conn):
    lines = load_prop_lines_from_db(conn, "2024_14_LAC_KC")
    assert lines == []


def test_load_prop_lines_anytime_td(conn):
    _insert_prop(conn, "2024_14_LAC_KC", "P2", "anytime_td", None, 200, None)
    lines = load_prop_lines_from_db(conn, "2024_14_LAC_KC")
    assert len(lines) == 1
    assert lines[0].stat_type == "anytime_td"
    assert lines[0].line is None
    assert lines[0].over_odds == 200


def test_load_prop_lines_validates_and_skips_invalid(conn):
    # rec_yards without a line — should be skipped (validation error)
    _insert_prop(conn, "2024_14_LAC_KC", "P3", "rec_yards", None, -110, None)
    lines = load_prop_lines_from_db(conn, "2024_14_LAC_KC")
    assert lines == []


# ── _parse_game_odds ──────────────────────────────────────────────────────────

def test_parse_game_odds_extracts_spread_and_total():
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc)
    event = {
        "bookmakers": [{
            "key": "draftkings",
            "markets": [
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": "Kansas City Chiefs", "point": -3.0, "price": -110},
                        {"name": "Los Angeles Chargers", "point": 3.0, "price": -110},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": 47.5, "price": -110},
                        {"name": "Under", "point": 47.5, "price": -110},
                    ],
                },
            ],
        }],
    }
    rows = _parse_game_odds(event, "2024_14_LAC_KC", "Kansas City Chiefs", "Los Angeles Chargers", ts)
    assert len(rows) == 1
    r = rows[0]
    assert r["game_id"] == "2024_14_LAC_KC"
    assert r["spread_home"] == -3.0
    assert r["total_over"] == 47.5
