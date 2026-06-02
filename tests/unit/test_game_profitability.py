"""Tests for the game-level profitability harness (spread/total bets)."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.eval.profitability import (
    _american_to_decimal,
    _spread_to_prob,
    profitability_summary,
    simulate_spread_bets,
    simulate_total_bets,
)


def _pred_row(**kw):
    base = {
        "game_id": "2023_01_KC_BAL", "season": 2023, "week": 1,
        "home_team": "BAL", "away_team": "KC",
        "home_win_prob": 0.60, "home_win_prob_vegas": 0.55,
        "home_margin_pred": 3.5, "total_pred": 48.0,
        "home_win_actual": True, "home_margin_actual": 7.0,
        "total_actual": 51.0, "vegas_spread": -3.0, "vegas_total": 46.5,
    }
    base.update(kw)
    return base


# ── helpers ──────────────────────────────────────────────────────────────────

def test_spread_to_prob_even():
    assert _spread_to_prob(0.0) == pytest.approx(0.5, abs=1e-6)


def test_spread_to_prob_home_favored():
    # Negative spread (home favored) → prob > 0.5
    assert _spread_to_prob(-3.0) > 0.5


def test_american_to_decimal_minus110():
    # Risk 110 to win 100 → decimal = 1 + 100/110 ≈ 1.909
    assert _american_to_decimal(-110) == pytest.approx(1.0 + 100 / 110, rel=1e-4)


def test_american_to_decimal_plus120():
    assert _american_to_decimal(120) == pytest.approx(2.2, rel=1e-4)


# ── simulate_spread_bets ─────────────────────────────────────────────────────

def test_spread_bets_places_bet_when_edge_exceeds_threshold():
    df = pd.DataFrame([_pred_row(home_win_prob=0.65, vegas_spread=-3.0)])
    bets = simulate_spread_bets(df, min_edge=0.03)
    assert len(bets) == 1
    assert bets.iloc[0]["side"] == "home"


def test_spread_bets_skips_when_edge_below_threshold():
    # Model prob ≈ market prob → no edge
    df = pd.DataFrame([_pred_row(home_win_prob=_spread_to_prob(-3.0), vegas_spread=-3.0)])
    bets = simulate_spread_bets(df, min_edge=0.03)
    assert len(bets) == 0


def test_spread_bets_away_side():
    # Model says away is significantly better
    df = pd.DataFrame([_pred_row(home_win_prob=0.35, vegas_spread=-3.0)])
    bets = simulate_spread_bets(df, min_edge=0.05)
    if len(bets) > 0:
        assert bets.iloc[0]["side"] == "away"


def test_spread_bet_home_win_profit():
    df = pd.DataFrame([_pred_row(
        home_win_prob=0.65, vegas_spread=-3.0,
        home_margin_actual=7.0,  # home covers (margin 7 > -(-3) = 3)
    )])
    bets = simulate_spread_bets(df, min_edge=0.03)
    assert len(bets) == 1
    assert bets.iloc[0]["result"] == "win"
    assert bets.iloc[0]["profit_units"] > 0


def test_spread_bet_home_loss():
    df = pd.DataFrame([_pred_row(
        home_win_prob=0.65, vegas_spread=-3.0,
        home_margin_actual=1.0,  # home fails to cover (1 < 3)
    )])
    bets = simulate_spread_bets(df, min_edge=0.03)
    assert len(bets) == 1
    assert bets.iloc[0]["result"] == "loss"
    assert bets.iloc[0]["profit_units"] == -1.0


def test_spread_bets_missing_actuals_skipped():
    df = pd.DataFrame([_pred_row(home_margin_actual=float("nan"))])
    bets = simulate_spread_bets(df, min_edge=0.03)
    assert len(bets) == 0


# ── simulate_total_bets ───────────────────────────────────────────────────────

def test_total_bets_over():
    df = pd.DataFrame([_pred_row(total_pred=50.0, vegas_total=46.5, total_actual=51.0)])
    bets = simulate_total_bets(df, min_line_diff=1.5)
    assert len(bets) == 1
    assert bets.iloc[0]["side"] == "over"
    assert bets.iloc[0]["result"] == "win"


def test_total_bets_under():
    df = pd.DataFrame([_pred_row(total_pred=43.0, vegas_total=46.5, total_actual=40.0)])
    bets = simulate_total_bets(df, min_line_diff=1.5)
    assert len(bets) == 1
    assert bets.iloc[0]["side"] == "under"
    assert bets.iloc[0]["result"] == "win"


def test_total_bets_skipped_below_diff():
    df = pd.DataFrame([_pred_row(total_pred=47.0, vegas_total=46.5)])
    bets = simulate_total_bets(df, min_line_diff=1.5)
    assert len(bets) == 0


# ── profitability_summary ────────────────────────────────────────────────────

def test_summary_empty():
    s = profitability_summary(pd.DataFrame())
    assert s["total_bets"] == 0


def test_summary_correct_roi():
    # 2 wins at +0.909u, 1 loss at -1u → net = 0.818, roi = 0.818/3
    decimal_odds = _american_to_decimal(-110)
    win_profit = decimal_odds - 1.0
    bets = pd.DataFrame([
        {"bet_type": "spread", "result": "win",  "profit_units": win_profit, "edge": 0.05},
        {"bet_type": "spread", "result": "win",  "profit_units": win_profit, "edge": 0.05},
        {"bet_type": "spread", "result": "loss", "profit_units": -1.0,       "edge": 0.05},
    ])
    s = profitability_summary(bets)
    assert s["overall"]["win_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert s["overall"]["roi"] == pytest.approx((2 * win_profit - 1) / 3, rel=1e-3)
