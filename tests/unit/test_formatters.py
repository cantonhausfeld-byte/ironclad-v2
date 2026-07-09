"""Tests for shared Discord+CLI formatters."""
from __future__ import annotations

import pandas as pd

from ironclad.notifications.formatters import (
    format_team_comparison_block,
    format_top_edges_block,
)


# ── format_top_edges_block ────────────────────────────────────────────────────

def test_top_edges_empty():
    assert format_top_edges_block(None) == "No edges found."
    assert format_top_edges_block(pd.DataFrame()) == "No edges found."


def test_top_edges_sorts_by_ev_desc():
    df = pd.DataFrame([
        {"player_name": "Alice", "stat_type": "pass_yards", "side": "over",
         "market_line": 250.5, "ev": 0.02, "kelly": 0.01},
        {"player_name": "Bob", "stat_type": "rec_yards", "side": "under",
         "market_line": 80.5, "ev": 0.10, "kelly": 0.05},
        {"player_name": "Carla", "stat_type": "carries", "side": "over",
         "market_line": 15.5, "ev": 0.05, "kelly": 0.03},
    ])
    out = format_top_edges_block(df, limit=10)
    lines = out.splitlines()
    assert "Top 3 edges (3 total)" in lines[0]
    bob_idx = next(i for i, line in enumerate(lines) if line.startswith("Bob"))
    alice_idx = next(i for i, line in enumerate(lines) if line.startswith("Alice"))
    carla_idx = next(i for i, line in enumerate(lines) if line.startswith("Carla"))
    assert bob_idx < carla_idx < alice_idx  # sorted by EV desc


def test_top_edges_respects_limit():
    df = pd.DataFrame([
        {"player_name": f"P{i}", "stat_type": "pass_yards", "side": "over",
         "market_line": float(i), "ev": 0.10 - i * 0.01, "kelly": 0.01}
        for i in range(20)
    ])
    out = format_top_edges_block(df, limit=5)
    # Strip the header row ("Player   Stat  ..."), count only the P0..P19 rows
    player_lines = [line for line in out.splitlines()
                    if line[:2] in {f"P{i}" for i in range(20)}]
    assert len(player_lines) == 5
    assert "Top 5 edges (20 total)" in out


def test_top_edges_handles_none_values():
    df = pd.DataFrame([{
        "player_name": "Alice", "stat_type": "pass_yards", "side": "over",
        "market_line": None, "ev": 0.02, "kelly": None,
    }])
    out = format_top_edges_block(df)
    assert "Alice" in out
    assert "—" in out  # None values rendered as em-dash


# ── format_team_comparison_block ──────────────────────────────────────────────

def _seed_team_stats(conn, team: str, season: int, rows: list[dict]):
    for i, r in enumerate(rows):
        conn.execute("""
            INSERT INTO silver.team_game_stats
                (game_id, season, week, team, opponent, is_home,
                 points_scored, total_yards, pass_yards, rush_yards, turnovers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [f"{season}_{i+1}_{team}_X", season, i + 1, team, "X", True,
              r["points_scored"], r["total_yards"], r["pass_yards"],
              r["rush_yards"], r["turnovers"]])


def test_team_comparison_no_data(conn):
    assert format_team_comparison_block(conn, "KC", "BAL", 2024) is None


def test_team_comparison_renders_two_columns(conn):
    _seed_team_stats(conn, "KC", 2024, [
        {"points_scored": 30, "total_yards": 400, "pass_yards": 280, "rush_yards": 120, "turnovers": 1},
        {"points_scored": 24, "total_yards": 350, "pass_yards": 250, "rush_yards": 100, "turnovers": 2},
    ])
    _seed_team_stats(conn, "BAL", 2024, [
        {"points_scored": 20, "total_yards": 320, "pass_yards": 200, "rush_yards": 120, "turnovers": 0},
    ])
    out = format_team_comparison_block(conn, "KC", "BAL", 2024)
    assert out is not None
    assert "KC vs BAL — 2024" in out
    assert "Games" in out and "Points/G" in out
    # KC average points = 27.0, BAL = 20.0
    assert "27.0" in out
    assert "20.0" in out


def test_team_comparison_handles_one_side_missing(conn):
    _seed_team_stats(conn, "KC", 2024, [
        {"points_scored": 30, "total_yards": 400, "pass_yards": 280, "rush_yards": 120, "turnovers": 1},
    ])
    out = format_team_comparison_block(conn, "KC", "BAL", 2024)
    assert out is not None
    assert "KC" in out and "BAL" in out
    assert "—" in out  # BAL side is missing
