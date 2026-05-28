"""Tests for the approximate prop-profitability harness."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.eval.prop_profitability import (
    _prob_over,
    approximate_prop_roi,
    summarize_roi,
)


def _row(p10, p25, p50, p75, p90, actual, **kw):
    base = {
        "player_id": "P1", "player_name": "Player One", "position": "WR",
        "stat_type": "rec_yards",
        "predicted_p10": p10, "predicted_p25": p25, "predicted_p50": p50,
        "predicted_p75": p75, "predicted_p90": p90, "actual_value": actual,
    }
    base.update(kw)
    return base


# ── P(over) interpolation ────────────────────────────────────────────────────

def test_prob_over_at_median_is_half():
    row = pd.Series(_row(10, 20, 30, 40, 50, actual=0))
    assert _prob_over(row, 30.0) == pytest.approx(0.5, abs=1e-9)


def test_prob_over_below_p10_is_high():
    row = pd.Series(_row(10, 20, 30, 40, 50, actual=0))
    assert _prob_over(row, 5.0) > 0.9


def test_prob_over_above_p90_is_low():
    row = pd.Series(_row(10, 20, 30, 40, 50, actual=0))
    assert _prob_over(row, 60.0) < 0.1


def test_prob_over_is_clamped():
    row = pd.Series(_row(10, 20, 30, 40, 50, actual=0))
    assert 0.02 <= _prob_over(row, 1000.0) <= 0.98
    assert 0.02 <= _prob_over(row, -1000.0) <= 0.98


# ── betting / settlement ─────────────────────────────────────────────────────

def test_empty_df_returns_empty():
    assert approximate_prop_roi(pd.DataFrame()).empty


def test_model_p50_line_bets_at_median():
    # With line == p50, model prob ≈ 0.5; default min_ev=0 means EV(-110, .5)<0,
    # so no bets are placed under the null baseline.
    df = pd.DataFrame([_row(10, 20, 30, 40, 50, actual=35)])
    bets = approximate_prop_roi(df, line_mode="model_p50")
    assert bets.empty


def test_winning_over_bet_pays_out():
    # Line well below the model's distribution → strong over; actual clears it.
    df = pd.DataFrame([
        _row(40, 50, 60, 70, 80, actual=75),  # game 1
        _row(40, 50, 60, 70, 80, actual=65),  # game 2 — same player/stat
    ])
    # season_mean line = (75+65)/2 = 70. Model p50=60 < 70 → favors UNDER.
    bets = approximate_prop_roi(df, line_mode="season_mean", min_ev=-1.0)
    assert len(bets) == 2
    assert set(bets["side"]) == {"under"}
    # actual 75 > 70 → under loses; actual 65 < 70 → under wins.
    results = dict(zip(bets["actual_value"], bets["result"]))
    assert results[75.0] == "loss"
    assert results[65.0] == "win"


def test_push_when_actual_equals_line():
    df = pd.DataFrame([
        _row(0, 1, 2, 3, 4, actual=2, stat_type="receptions"),
        _row(0, 1, 2, 3, 4, actual=2, stat_type="receptions"),
    ])
    # season_mean = 2.0; both actuals == line → pushes.
    bets = approximate_prop_roi(df, line_mode="season_mean", min_ev=-1.0)
    assert (bets["result"] == "push").all()
    assert bets["profit_units"].sum() == pytest.approx(0.0)


# ── summary ──────────────────────────────────────────────────────────────────

def test_summarize_roi_excludes_pushes_from_hit_rate():
    bets = pd.DataFrame([
        {"stat_type": "rec_yards", "position": "WR", "result": "win",
         "profit_units": 0.909, "ev": 0.1},
        {"stat_type": "rec_yards", "position": "WR", "result": "loss",
         "profit_units": -1.0, "ev": 0.1},
        {"stat_type": "rec_yards", "position": "WR", "result": "push",
         "profit_units": 0.0, "ev": 0.1},
    ])
    summary = summarize_roi(bets)
    wr = summary[(summary["stat_type"] == "rec_yards") & (summary["position"] == "WR")].iloc[0]
    assert wr["n_bets"] == 3
    assert wr["hit_rate"] == pytest.approx(0.5)  # 1 win / 2 decided
    assert wr["roi"] == pytest.approx((0.909 - 1.0) / 2)  # staked on 2 decided
    # An overall ALL row is always present.
    assert (summary["stat_type"] == "ALL").any()
