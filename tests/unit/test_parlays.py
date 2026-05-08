"""Tests for same-game parlay probability engine."""
from __future__ import annotations

from pathlib import Path

import pytest

from ironclad.betting.ev import prob_to_american
from ironclad.betting.parlays import (
    ParlayLeg,
    _leg_hit,
    load_parlay_legs,
    market_independence_prob,
    parlay_probability,
)
from ironclad.simulation.results import DrawRecord, SimulationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stats(player_id: str = "P1", rec_yards: float = 0.0, rush_yards: float = 0.0,
           pass_yards: float = 0.0, tds: int = 0, played: bool = True) -> dict:
    return {
        "player_id": player_id, "player_name": "Test", "team": "KC",
        "position": "WR", "is_home": True, "played": played,
        "targets": 5, "receptions": 3, "rec_yards": rec_yards,
        "carries": 4, "rush_yards": rush_yards,
        "pass_attempts": 0, "completions": 0, "pass_yards": pass_yards,
        "tds": tds,
    }


def _sim(n_over: int, n_total: int, stat: str = "rec_yards",
         line: float = 60.0, player_id: str = "P1") -> SimulationResult:
    draws = []
    for i in range(n_total):
        val = 80.0 if i < n_over else 40.0
        draws.append(DrawRecord(
            home_score=21, away_score=17,
            home_pass_yards=250.0, away_pass_yards=200.0,
            home_rush_yards=100.0, away_rush_yards=90.0,
            home_pass_att=32, away_pass_att=28,
            player_stats=[_stats(player_id=player_id, **{stat: val})],
        ))
    return SimulationResult("KC", "LAC", draws)


# ── _leg_hit ──────────────────────────────────────────────────────────────────

def test_leg_hit_over():
    leg = ParlayLeg("P1", "rec_yards", "over", 60.0)
    assert _leg_hit(_stats(rec_yards=80.0), leg) is True
    assert _leg_hit(_stats(rec_yards=40.0), leg) is False


def test_leg_hit_under():
    leg = ParlayLeg("P1", "rec_yards", "under", 60.0)
    assert _leg_hit(_stats(rec_yards=40.0), leg) is True
    assert _leg_hit(_stats(rec_yards=80.0), leg) is False


def test_leg_hit_yes_anytime_td():
    leg = ParlayLeg("P1", "anytime_td", "yes", None)
    assert _leg_hit(_stats(tds=1), leg) is True
    assert _leg_hit(_stats(tds=2), leg) is True
    assert _leg_hit(_stats(tds=0), leg) is False


def test_leg_hit_player_not_played():
    leg = ParlayLeg("P1", "rec_yards", "over", 60.0)
    assert _leg_hit(_stats(rec_yards=80.0, played=False), leg) is False


def test_leg_hit_missing_player():
    leg = ParlayLeg("P1", "rec_yards", "over", 60.0)
    assert _leg_hit({}, leg) is False


# ── parlay_probability ────────────────────────────────────────────────────────

def test_single_leg_prob_matches_count():
    sim = _sim(n_over=800, n_total=1000)
    leg = ParlayLeg("P1", "rec_yards", "over", 60.0)
    assert parlay_probability(sim, [leg]) == pytest.approx(0.8)


def test_multi_leg_prob_all_hit():
    # Both players hit in every draw.
    draws = []
    for _ in range(200):
        draws.append(DrawRecord(
            home_score=21, away_score=17,
            home_pass_yards=250.0, away_pass_yards=200.0,
            home_rush_yards=100.0, away_rush_yards=90.0,
            home_pass_att=32, away_pass_att=28,
            player_stats=[
                _stats(player_id="P1", rec_yards=80.0),
                _stats(player_id="P2", rush_yards=70.0),
            ],
        ))
    sim = SimulationResult("KC", "LAC", draws)
    legs = [
        ParlayLeg("P1", "rec_yards", "over", 60.0),
        ParlayLeg("P2", "rush_yards", "over", 60.0),
    ]
    assert parlay_probability(sim, legs) == pytest.approx(1.0)


def test_multi_leg_prob_none_hit():
    # No draw satisfies both legs simultaneously.
    draws = []
    for i in range(200):
        draws.append(DrawRecord(
            home_score=21, away_score=17,
            home_pass_yards=250.0, away_pass_yards=200.0,
            home_rush_yards=100.0, away_rush_yards=90.0,
            home_pass_att=32, away_pass_att=28,
            player_stats=[
                _stats(player_id="P1", rec_yards=80.0),   # over 60 ✓
                _stats(player_id="P2", rush_yards=40.0),  # over 60 ✗
            ],
        ))
    sim = SimulationResult("KC", "LAC", draws)
    legs = [
        ParlayLeg("P1", "rec_yards", "over", 60.0),
        ParlayLeg("P2", "rush_yards", "over", 60.0),
    ]
    assert parlay_probability(sim, legs) == pytest.approx(0.0)


def test_unknown_player_never_hits():
    sim = _sim(n_over=1000, n_total=1000)
    leg = ParlayLeg("UNKNOWN", "rec_yards", "over", 60.0)
    assert parlay_probability(sim, [leg]) == pytest.approx(0.0)


# ── market_independence_prob ──────────────────────────────────────────────────

def test_market_independence_prob_all_present():
    legs = [
        ParlayLeg("P1", "rec_yards", "over", 60.0, market_odds=-110),
        ParlayLeg("P2", "rush_yards", "over", 50.0, market_odds=-110),
    ]
    # Each -110 leg has implied prob 0.5238; product ≈ 0.2744
    prob = market_independence_prob(legs)
    assert prob == pytest.approx(0.5238 * 0.5238, rel=1e-2)


def test_market_independence_prob_missing_odds():
    legs = [
        ParlayLeg("P1", "rec_yards", "over", 60.0, market_odds=-110),
        ParlayLeg("P2", "rush_yards", "over", 50.0, market_odds=None),
    ]
    assert market_independence_prob(legs) is None


# ── prob_to_american ──────────────────────────────────────────────────────────

def test_prob_to_american_favorite():
    # p=0.6 → -150
    assert prob_to_american(0.6) == -150


def test_prob_to_american_underdog():
    # p=0.4 → +150
    assert prob_to_american(0.4) == 150


def test_prob_to_american_invalid():
    with pytest.raises(ValueError):
        prob_to_american(0.0)
    with pytest.raises(ValueError):
        prob_to_american(1.0)


# ── load_parlay_legs ──────────────────────────────────────────────────────────

def test_load_parlay_legs(tmp_path: Path):
    csv = tmp_path / "legs.csv"
    csv.write_text(
        "player_id,stat_type,direction,line,market_odds\n"
        "P1,rec_yards,over,55,-110\n"
        "P2,rush_yards,under,60.5,-115\n"
        "P3,anytime_td,yes,,200\n"
    )
    legs = load_parlay_legs(csv)
    assert len(legs) == 3
    assert legs[0].direction == "over"
    assert legs[0].line == 55.0
    assert legs[0].market_odds == -110
    assert legs[2].stat_type == "anytime_td"
    assert legs[2].line is None
    assert legs[2].market_odds == 200


def test_parlay_leg_validate_unknown_stat():
    leg = ParlayLeg("P1", "weird_stat", "over", 10.0)
    with pytest.raises(ValueError):
        leg.validate()


def test_parlay_leg_validate_bad_direction():
    leg = ParlayLeg("P1", "rec_yards", "sideways", 10.0)
    with pytest.raises(ValueError):
        leg.validate()
