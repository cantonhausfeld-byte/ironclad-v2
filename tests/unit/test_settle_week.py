"""Unit tests for auto_settle_week() and the retroactive settlement loop."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ironclad.eval.performance_tracker import auto_settle_week, load_results, pnl_summary, save_edges
from ironclad.store.schema import create_all_tables


@pytest.fixture()
def conn():
    import duckdb
    c = duckdb.connect(":memory:")
    c.execute("CREATE SCHEMA bronze")
    c.execute("CREATE SCHEMA silver")
    c.execute("CREATE SCHEMA gold")
    create_all_tables(c)
    return c


def _seed_game(conn, game_id: str = "2025_18_KC_LV", season: int = 2025, week: int = 18) -> None:
    conn.execute(f"""
        INSERT INTO silver.games (game_id, season, week, season_type, home_team, away_team, gameday)
        VALUES ('{game_id}', {season}, {week}, 'REG', 'KC', 'LV', '2025-01-05')
    """)


def _seed_edge(conn, game_id: str, player_id: str, stat_type: str, side: str,
               market_line: float, odds: int = -110) -> str:
    edges_df = pd.DataFrame([{
        "player_id":   player_id,
        "player_name": "Test Player",
        "team":        "KC",
        "position":    "QB",
        "stat_type":   stat_type,
        "side":        side,
        "market_line": market_line,
        "odds":        odds,
        "model_prob":  0.55,
        "market_prob": 0.52,
        "edge":        0.03,
        "ev":          0.06,
        "kelly":       0.05,
        "model_p10":   float(market_line) * 0.7,
        "model_p50":   float(market_line) * 1.05,
        "model_p90":   float(market_line) * 1.4,
    }])
    return save_edges(conn, game_id, 500, edges_df)[0]


def _seed_actual(conn, player_id: str, season: int, week: int, passing_yards: float) -> None:
    conn.execute(f"""
        INSERT INTO bronze.player_stats_weekly
            (player_id, player_name, season, week, passing_yards, team, position,
             passing_tds, rushing_yards, rushing_tds, receptions, receiving_yards,
             receiving_tds, carries, completions, attempts, targets)
        VALUES ('{player_id}', 'Test Player', {season}, {week}, {passing_yards},
                'KC', 'QB', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    """)


# ── Basic win/loss/push detection ─────────────────────────────────────────────

def test_settle_week_over_win(conn):
    """Edge: pass_yards OVER 250.5 → actual 310 → WIN."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "player-001", "pass_yards", "over", 250.5)
    _seed_actual(conn, "player-001", 2025, 18, 310.0)

    settled = auto_settle_week(conn, 2025, 18)
    assert len(settled) == 1
    assert settled[0]["result"] == "win"
    assert settled[0]["profit_units"] > 0


def test_settle_week_over_loss(conn):
    """Edge: pass_yards OVER 250.5 → actual 180 → LOSS."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "player-002", "pass_yards", "over", 250.5)
    _seed_actual(conn, "player-002", 2025, 18, 180.0)

    settled = auto_settle_week(conn, 2025, 18)
    assert len(settled) == 1
    assert settled[0]["result"] == "loss"
    assert settled[0]["profit_units"] < 0


def test_settle_week_under_win(conn):
    """Edge: rush_yards UNDER 80.5 → actual 55 → WIN."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "player-003", "rush_yards", "under", 80.5)
    _seed_actual(conn, "player-003", 2025, 18, 55.0)

    settled = auto_settle_week(conn, 2025, 18)
    assert len(settled) == 1
    assert settled[0]["result"] == "win"


def test_settle_week_push(conn):
    """Edge: pass_yards OVER 250.5 → actual exactly 250.5 → PUSH."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "player-004", "pass_yards", "over", 250.5)
    _seed_actual(conn, "player-004", 2025, 18, 250.5)

    settled = auto_settle_week(conn, 2025, 18)
    assert len(settled) == 1
    assert settled[0]["result"] == "push"
    assert settled[0]["profit_units"] == 0.0


def test_settle_week_multiple_edges(conn):
    """Multiple edges for the same week are all settled in one call."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "player-005", "pass_yards", "over", 250.5)
    _seed_edge(conn, "2025_18_KC_LV", "player-006", "rush_yards", "under", 80.5)
    _seed_actual(conn, "player-005", 2025, 18, 310.0)  # win
    _seed_actual(conn, "player-006", 2025, 18, 55.0)   # win

    settled = auto_settle_week(conn, 2025, 18)
    assert len(settled) == 2
    assert all(r["result"] == "win" for r in settled)


def test_settle_week_no_double_settle(conn):
    """Already-settled edges are skipped on a second call."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "player-007", "pass_yards", "over", 250.5)
    _seed_actual(conn, "player-007", 2025, 18, 310.0)

    settled1 = auto_settle_week(conn, 2025, 18)
    settled2 = auto_settle_week(conn, 2025, 18)
    assert len(settled1) == 1
    assert len(settled2) == 0  # already settled


def test_settle_week_unknown_player_skipped(conn):
    """Edges for players not in bronze.player_stats_weekly are skipped."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "nobody-999", "pass_yards", "over", 250.5)
    # No actual row for nobody-999

    settled = auto_settle_week(conn, 2025, 18)
    assert len(settled) == 0


def test_full_loop_pnl_summary(conn):
    """Full loop: seed game → edges → settle → pnl_summary shows correct ROI."""
    _seed_game(conn)
    _seed_edge(conn, "2025_18_KC_LV", "player-010", "pass_yards", "over", 250.5, odds=-110)
    _seed_actual(conn, "player-010", 2025, 18, 310.0)

    auto_settle_week(conn, 2025, 18, units_wagered=1.0)

    df = load_results(conn)
    assert not df.empty
    summary = pnl_summary(df)
    assert summary["total_bets"] == 1
    assert summary["wins"] == 1
    assert summary["win_rate"] == 1.0
    assert summary["roi"] > 0  # won at -110 odds
