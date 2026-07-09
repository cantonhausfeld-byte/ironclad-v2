"""Tests for the ironclad FastAPI REST layer."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ironclad.api.app import app

client = TestClient(app)


# ── GET /api/v1/games ─────────────────────────────────────────────────────────

def test_get_games_empty_db(conn):
    """Empty silver.games → 200 with empty list."""
    with patch("ironclad.api.app.get_connection", return_value=conn):
        resp = client.get("/api/v1/games", params={"season": 2024, "week": 14})
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_games_returns_data(conn):
    """After inserting a game row, the endpoint returns it."""
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday, home_team, away_team)
        VALUES ('2024_14_LAC_KC', 2024, 'REG', 14, '2024-12-15', 'KC', 'LAC')
    """)
    with patch("ironclad.api.app.get_connection", return_value=conn):
        resp = client.get("/api/v1/games", params={"season": 2024, "week": 14})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["game_id"] == "2024_14_LAC_KC"
    assert data[0]["home_team"] == "KC"


# ── POST /api/v1/simulate ─────────────────────────────────────────────────────

def test_simulate_missing_game():
    """Game not in DB → 404."""
    with patch("ironclad.api.app.MatchupWorkflow") as MockWF:
        MockWF.return_value.simulate.side_effect = ValueError("Game not found")
        resp = client.post("/api/v1/simulate", json={"game_id": "2024_14_LAC_KC", "n_draws": 100})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_simulate_success():
    """Monkeypatched simulate → 200 with home_win_prob."""
    from ironclad.simulation.results import DrawRecord, SimulationResult

    draws = [DrawRecord(home_score=24, away_score=17,
                        home_pass_yards=250, away_pass_yards=200,
                        home_rush_yards=80, away_rush_yards=60,
                        home_pass_att=32, away_pass_att=28)
             for _ in range(100)]
    mock_result = SimulationResult("KC", "LAC", draws)

    with patch("ironclad.api.app.MatchupWorkflow") as MockWF:
        MockWF.return_value.simulate.return_value = (mock_result, {}, None, MagicMock())
        resp = client.post("/api/v1/simulate", json={"game_id": "2024_14_LAC_KC", "n_draws": 100})

    assert resp.status_code == 200
    body = resp.json()
    assert "home_win_prob" in body
    assert body["home_team"] == "KC"
    assert body["away_team"] == "LAC"
    assert 0.0 <= body["home_win_prob"] <= 1.0


# ── POST /api/v1/edges ────────────────────────────────────────────────────────

def test_edges_no_props(conn):
    """No props in DB → 404 with helpful message."""
    with patch("ironclad.api.app.get_connection", return_value=conn), \
         patch("ironclad.api.app.load_prop_lines_from_db", return_value=[]):
        resp = client.post("/api/v1/edges", json={"game_id": "2024_14_LAC_KC", "n_draws": 100})
    assert resp.status_code == 404
    assert "ironclad odds" in resp.json()["detail"]


# ── GET /api/v1/results ───────────────────────────────────────────────────────

def test_results_empty(conn):
    """No settled bets → 200 with zero summary."""
    with patch("ironclad.api.app.get_connection", return_value=conn):
        resp = client.get("/api/v1/results")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["total_bets"] == 0
    assert body["bets"] == []


# ── GET /api/v1/health ────────────────────────────────────────────────────────

def test_health_ok():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── GET /api/v1/status ────────────────────────────────────────────────────────

def test_status_returns_table_counts(conn):
    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, season_type, week, gameday, home_team, away_team)
        VALUES ('2024_1_KC_BAL', 2024, 'REG', 1, '2024-09-05', 'KC', 'BAL')
    """)
    with patch("ironclad.api.app.get_connection", return_value=conn):
        resp = client.get("/api/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "tables" in body and "models" in body and "scheduler" in body
    assert body["tables"]["silver.games"] == 1
    assert isinstance(body["models"], dict)
    assert "game_outcome" in body["models"]


# ── GET /api/v1/edges/stored ──────────────────────────────────────────────────

def _seed_stored_edge(conn, *, edge_id: str, game_id: str, player: str, ev: float):
    conn.execute("""
        INSERT INTO gold.betting_edges
            (edge_id, analyzed_at, game_id, n_draws, player_id, player_name,
             stat_type, market_line, side, odds, model_prob, market_prob, ev, kelly)
        VALUES (?, NOW(), ?, 5000, ?, ?, 'pass_yards', 250.5, 'over',
                -110, 0.55, 0.52, ?, 0.03)
    """, [edge_id, game_id, player.lower(), player, ev])


def test_edges_stored_returns_seeded_rows(conn):
    _seed_stored_edge(conn, edge_id="e1", game_id="g1", player="Alice", ev=0.02)
    _seed_stored_edge(conn, edge_id="e2", game_id="g1", player="Bob", ev=0.10)
    with patch("ironclad.api.app.get_connection", return_value=conn):
        resp = client.get("/api/v1/edges/stored", params={"limit": 10})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["player_name"] == "Bob"  # sorted by EV desc
    assert data[1]["player_name"] == "Alice"


def test_edges_stored_filters_by_game_id(conn):
    _seed_stored_edge(conn, edge_id="e1", game_id="g1", player="Alice", ev=0.05)
    _seed_stored_edge(conn, edge_id="e2", game_id="g2", player="Bob", ev=0.10)
    with patch("ironclad.api.app.get_connection", return_value=conn):
        resp = client.get("/api/v1/edges/stored", params={"game_id": "g1"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["player_name"] == "Alice"


def test_edges_stored_min_ev_filter(conn):
    _seed_stored_edge(conn, edge_id="e1", game_id="g1", player="Alice", ev=0.02)
    _seed_stored_edge(conn, edge_id="e2", game_id="g1", player="Bob", ev=0.10)
    with patch("ironclad.api.app.get_connection", return_value=conn):
        resp = client.get("/api/v1/edges/stored", params={"min_ev": 0.05})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["player_name"] == "Bob"
