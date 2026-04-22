"""Tests for report builder context assembly."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from ironclad.simulation.results import SimulationResult, DrawRecord
from ironclad.report.builder import build_report_context


def _make_result(n: int = 50, seed: int = 0) -> SimulationResult:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        home_score = int(rng.integers(10, 45))
        away_score = int(rng.integers(10, 45))
        draws.append(DrawRecord(
            home_score=home_score,
            away_score=away_score,
            home_pass_yards=float(rng.integers(150, 350)),
            away_pass_yards=float(rng.integers(150, 350)),
            home_rush_yards=float(rng.integers(60, 200)),
            away_rush_yards=float(rng.integers(60, 200)),
            home_pass_att=int(rng.integers(25, 45)),
            away_pass_att=int(rng.integers(25, 45)),
            player_stats=[
                {
                    "player_id": "QB1", "player_name": "Home QB", "team": "KC",
                    "position": "QB", "is_home": True, "played": True,
                    "targets": 0, "receptions": 0, "rec_yards": 0,
                    "carries": 0, "rush_yards": 0,
                    "pass_attempts": int(rng.integers(20, 40)),
                    "completions": int(rng.integers(12, 30)),
                    "pass_yards": float(rng.integers(150, 320)),
                    "tds": int(rng.integers(0, 4)),
                },
                {
                    "player_id": "WR1", "player_name": "Home WR", "team": "KC",
                    "position": "WR", "is_home": True, "played": True,
                    "targets": int(rng.integers(4, 12)),
                    "receptions": int(rng.integers(2, 9)),
                    "rec_yards": float(rng.integers(20, 120)),
                    "carries": 0, "rush_yards": 0,
                    "pass_attempts": 0, "completions": 0, "pass_yards": 0,
                    "tds": int(rng.integers(0, 2)),
                },
            ],
        ))
    return SimulationResult("KC", "BAL", draws)


_GAME_META = {
    "season": 2023, "week": 1, "gameday": "2023-09-07",
    "stadium_id": "M&T Bank Stadium", "stadium": "M&T Bank Stadium",
    "is_dome": False, "surface": "Grass", "altitude_ft": 0,
    "temp_f": 72.0, "wind_mph": 5.0, "precip_in": 0.0,
    "total_consensus": 48.5, "spread_consensus": -3.5,
    "data_completeness_score": 0.8,
}

_CUTOFF = datetime(2023, 9, 7, 11, 30, tzinfo=timezone.utc)


def test_context_has_required_keys():
    result = _make_result()
    ctx = build_report_context(result, _GAME_META, _CUTOFF)
    for key in ["home_team", "away_team", "home_win_prob", "away_win_prob",
                "home_score", "away_score", "total_projected",
                "home_stats", "away_stats", "confidence", "n_draws"]:
        assert key in ctx


def test_win_probs_sum_to_100():
    result = _make_result()
    ctx = build_report_context(result, _GAME_META, _CUTOFF)
    hp = float(ctx["home_win_prob"].rstrip("%"))
    ap = float(ctx["away_win_prob"].rstrip("%"))
    assert abs(hp + ap - 100.0) < 0.5


def test_confidence_tier_high():
    result = _make_result()
    ctx = build_report_context(result, {**_GAME_META, "data_completeness_score": 0.9}, _CUTOFF)
    assert ctx["confidence"] == "HIGH"


def test_confidence_tier_low():
    result = _make_result()
    ctx = build_report_context(result, {**_GAME_META, "data_completeness_score": 0.1}, _CUTOFF)
    assert ctx["confidence"] == "LOW"


def test_player_table_qb_has_range_keys():
    result = _make_result(n=100)
    ctx = build_report_context(result, _GAME_META, _CUTOFF)
    qb_rows = ctx["home_qb"]
    if qb_rows:
        row = qb_rows[0]
        assert "pass_attempts" in row
        assert "pass_attempts_range" in row
        assert "pass_yards_range" in row
        assert "tds_range" in row


def test_home_stats_numeric():
    result = _make_result(n=100)
    ctx = build_report_context(result, _GAME_META, _CUTOFF)
    stats = ctx["home_stats"]
    assert isinstance(stats["total_yards"], float)
    assert stats["total_yards"] > 0


def test_dome_weather_desc():
    result = _make_result()
    ctx = build_report_context(result, {**_GAME_META, "is_dome": True}, _CUTOFF)
    assert "dome" in ctx["weather_desc"].lower() or "indoor" in ctx["weather_desc"].lower()
