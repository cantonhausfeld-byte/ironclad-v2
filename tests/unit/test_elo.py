"""Elo ratings: update rules and knowledge-cutoff behaviour."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ironclad.features.elo import INITIAL_ELO, compute_ratings
from ironclad.features.snapshot import FeatureSnapshot
from ironclad.models.team.game_outcome import _build_diff_features


def _games(rows):
    return pd.DataFrame(
        rows,
        columns=["game_id", "season", "gameday", "home_team", "away_team",
                 "home_score", "away_score"],
    ).assign(gameday=lambda d: pd.to_datetime(d["gameday"]))


def test_no_games_means_no_ratings():
    assert compute_ratings(_games([]), as_of_season=2026) == {}


def test_updates_are_zero_sum():
    r = compute_ratings(_games([
        ("g1", 2025, "2025-09-07", "KC", "BAL", 27, 20),
        ("g2", 2025, "2025-09-14", "BAL", "CIN", 10, 31),
    ]), as_of_season=2025)
    assert sum(r.values()) == pytest.approx(3 * INITIAL_ELO)
    assert r["KC"] > INITIAL_ELO > r["BAL"]


def test_home_field_means_road_wins_move_ratings_more():
    home_win = compute_ratings(_games([("g", 2025, "2025-09-07", "A", "B", 24, 17)]), 2025)
    road_win = compute_ratings(_games([("g", 2025, "2025-09-07", "B", "A", 17, 24)]), 2025)
    assert road_win["A"] - INITIAL_ELO > home_win["A"] - INITIAL_ELO


def test_tie_lowers_home_team_expected_to_win():
    r = compute_ratings(_games([("g", 2025, "2025-09-07", "A", "B", 20, 20)]), 2025)
    assert r["A"] < INITIAL_ELO < r["B"]


def test_bigger_margin_moves_ratings_more():
    close = compute_ratings(_games([("g", 2025, "2025-09-07", "A", "B", 21, 20)]), 2025)
    blowout = compute_ratings(_games([("g", 2025, "2025-09-07", "A", "B", 45, 3)]), 2025)
    assert blowout["A"] > close["A"]


def test_regresses_one_third_at_new_season():
    games = _games([("g", 2024, "2024-12-01", "A", "B", 42, 0)])
    end_2024 = compute_ratings(games, as_of_season=2024)
    start_2025 = compute_ratings(games, as_of_season=2025)
    for team in ("A", "B"):
        expected = end_2024[team] + (INITIAL_ELO - end_2024[team]) / 3
        assert start_2025[team] == pytest.approx(expected)


def test_unplayed_games_are_ignored():
    r = compute_ratings(_games([("g", 2025, "2025-09-07", "A", "B", None, None)]), 2025)
    assert r == {}


# ── Snapshot integration ──────────────────────────────────────────────────────

def _insert_game(conn, game_id, season, week, gameday, home, away, hs, as_):
    conn.execute(
        """INSERT INTO silver.games
           (game_id, season, season_type, week, gameday, home_team, away_team,
            home_score, away_score)
           VALUES (?, ?, 'REG', ?, ?, ?, ?, ?, ?)""",
        [game_id, season, week, gameday, home, away, hs, as_],
    )


def test_snapshot_elo_only_uses_games_before_cutoff(conn):
    _insert_game(conn, "2026_01_B_A", 2026, 1, "2026-09-13", "A", "B", 31, 3)
    _insert_game(conn, "2026_02_C_A", 2026, 2, "2026-09-20", "A", "C", 0, 35)

    # Cutoff before week 2 kickoff: week-2 blowout loss must not be visible
    snap = FeatureSnapshot(datetime(2026, 9, 20, 16, 30, tzinfo=timezone.utc), conn)
    before_wk2 = snap.team_elo("A", 2026)
    assert before_wk2 > INITIAL_ELO
    assert snap.team_elo("C", 2026) == INITIAL_ELO

    later = FeatureSnapshot(datetime(2026, 9, 27, tzinfo=timezone.utc), conn)
    assert later.team_elo("A", 2026) < before_wk2


def test_elo_diff_from_pivoted_rows():
    X = pd.DataFrame({"home_elo_pre_game": [1600.0, None], "away_elo_pre_game": [1550.0, 1450.0]})
    out = _build_diff_features(X)
    assert list(out["elo_diff"]) == [50.0, INITIAL_ELO - 1450.0]


def test_elo_diff_absent_columns_is_zero():
    out = _build_diff_features(pd.DataFrame({"home_off_epa_per_play_l4": [0.1]}))
    assert out["elo_diff"].iloc[0] == 0.0



def test_team_feature_row_carries_pregame_elo(conn):
    from ironclad.features.team_features import TeamFeatureBuilder

    _insert_game(conn, "2026_01_B_A", 2026, 1, "2026-09-13", "A", "B", 31, 3)
    _insert_game(conn, "2026_02_C_A", 2026, 2, "2026-09-20", "A", "C", None, None)

    df = TeamFeatureBuilder(conn).build_for_game(
        "2026_02_C_A", datetime(2026, 9, 20, 16, 30, tzinfo=timezone.utc),
    )
    elo = df.set_index("team")["elo_pre_game"]
    assert elo["A"] > INITIAL_ELO
    assert elo["C"] == INITIAL_ELO

    stored = conn.execute(
        "SELECT elo_pre_game FROM gold.team_game_features WHERE game_id = '2026_02_C_A' AND team = 'A'"
    ).fetchone()[0]
    assert stored == pytest.approx(elo["A"])
