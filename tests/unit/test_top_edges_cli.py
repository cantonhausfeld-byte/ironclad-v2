"""Tests for `ironclad top-edges` and `ironclad compare-teams` CLI commands."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cli.main import cli


def _seed_edge(conn, *, edge_id: str, game_id: str, player: str, ev: float, stat: str = "pass_yards"):
    conn.execute("""
        INSERT INTO gold.betting_edges
            (edge_id, analyzed_at, game_id, n_draws, player_id, player_name,
             stat_type, market_line, side, odds, model_prob, market_prob, ev, kelly)
        VALUES (?, NOW(), ?, 5000, ?, ?, ?, 250.5, 'over', -110, 0.55, 0.52, ?, 0.03)
    """, [edge_id, game_id, player.lower(), player, stat, ev])


def _seed_game(conn, *, game_id: str, season: int, week: int, home: str = "CIN", away: str = "CLE"):
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday, home_team, away_team)
        VALUES (?, ?, 'REG', ?, '2025-01-04', ?, ?)
    """, [game_id, season, week, home, away])


def test_top_edges_prints_sorted_by_ev(conn):
    _seed_edge(conn, edge_id="e1", game_id="g1", player="Alice", ev=0.02)
    _seed_edge(conn, edge_id="e2", game_id="g1", player="Bob", ev=0.10)
    _seed_edge(conn, edge_id="e3", game_id="g1", player="Carla", ev=0.05)

    runner = CliRunner()
    with patch("ironclad.store.connection.get_connection", return_value=conn), \
         patch("cli.main.get_connection", return_value=conn, create=True):
        result = runner.invoke(cli, ["top-edges", "--limit", "5"])
    assert result.exit_code == 0, result.output
    out = result.output
    bob_idx = out.find("Bob")
    carla_idx = out.find("Carla")
    alice_idx = out.find("Alice")
    assert bob_idx != -1 and carla_idx != -1 and alice_idx != -1
    assert bob_idx < carla_idx < alice_idx


def test_top_edges_filters_by_game_id(conn):
    _seed_edge(conn, edge_id="e1", game_id="g1", player="Alice", ev=0.05)
    _seed_edge(conn, edge_id="e2", game_id="g2", player="Bob", ev=0.10)

    runner = CliRunner()
    with patch("ironclad.store.connection.get_connection", return_value=conn):
        result = runner.invoke(cli, ["top-edges", "--game-id", "g1"])
    assert result.exit_code == 0, result.output
    assert "Alice" in result.output
    assert "Bob" not in result.output


def test_top_edges_filters_by_min_ev(conn):
    _seed_edge(conn, edge_id="e1", game_id="g1", player="Alice", ev=0.02)
    _seed_edge(conn, edge_id="e2", game_id="g1", player="Bob", ev=0.10)

    runner = CliRunner()
    with patch("ironclad.store.connection.get_connection", return_value=conn):
        result = runner.invoke(cli, ["top-edges", "--min-ev", "0.05"])
    assert result.exit_code == 0, result.output
    assert "Bob" in result.output
    assert "Alice" not in result.output


def test_top_edges_filters_by_season_week(conn):
    _seed_game(conn, game_id="g1", season=2024, week=1)
    _seed_game(conn, game_id="g2", season=2025, week=18)
    _seed_edge(conn, edge_id="e1", game_id="g1", player="Alice", ev=0.05)
    _seed_edge(conn, edge_id="e2", game_id="g2", player="Bob", ev=0.05)

    runner = CliRunner()
    with patch("ironclad.store.connection.get_connection", return_value=conn):
        result = runner.invoke(cli, ["top-edges", "--season", "2025", "--week", "18"])
    assert result.exit_code == 0, result.output
    assert "Bob" in result.output
    assert "Alice" not in result.output


def test_top_edges_empty_ok(conn):
    runner = CliRunner()
    with patch("ironclad.store.connection.get_connection", return_value=conn):
        result = runner.invoke(cli, ["top-edges"])
    assert result.exit_code == 0, result.output
    assert "No edges found." in result.output


def test_compare_teams_prints_stats(conn):
    for i in range(2):
        conn.execute("""
            INSERT INTO silver.team_game_stats
                (game_id, season, week, team, opponent, is_home,
                 points_scored, total_yards, pass_yards, rush_yards, turnovers)
            VALUES (?, 2024, ?, 'KC', 'X', TRUE, 28, 400, 280, 120, 1)
        """, [f"g_kc_{i}", i + 1])
    for i in range(2):
        conn.execute("""
            INSERT INTO silver.team_game_stats
                (game_id, season, week, team, opponent, is_home,
                 points_scored, total_yards, pass_yards, rush_yards, turnovers)
            VALUES (?, 2024, ?, 'BAL', 'X', TRUE, 20, 320, 200, 120, 0)
        """, [f"g_bal_{i}", i + 1])

    runner = CliRunner()
    with patch("ironclad.store.connection.get_connection", return_value=conn):
        result = runner.invoke(cli, ["compare-teams", "KC", "BAL", "--season", "2024"])
    assert result.exit_code == 0, result.output
    assert "KC vs BAL — 2024" in result.output
    assert "28.0" in result.output
    assert "20.0" in result.output


def test_compare_teams_no_data(conn):
    runner = CliRunner()
    with patch("ironclad.store.connection.get_connection", return_value=conn):
        result = runner.invoke(cli, ["compare-teams", "KC", "BAL", "--season", "2024"])
    assert result.exit_code == 0, result.output
    assert "No stats found" in result.output
