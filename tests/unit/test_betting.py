"""Tests for betting intelligence: ev, kelly, props."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from ironclad.betting.ev import (
    american_to_decimal,
    american_to_prob,
    decimal_to_prob,
    edge,
    expected_value,
    no_vig_prob,
)
from ironclad.betting.kelly import (
    KellyBankroll,
    fractional_kelly,
    kelly_fraction,
)
from ironclad.betting.props import PropAnalyzer, PropLine, load_prop_lines
from ironclad.simulation.results import DrawRecord, SimulationResult


# ── Odds conversions ──────────────────────────────────────────────────────────

def test_american_to_decimal_negative():
    assert american_to_decimal(-110) == pytest.approx(1.9091, rel=1e-3)


def test_american_to_decimal_positive():
    assert american_to_decimal(150) == pytest.approx(2.5)


def test_american_to_prob_negative_favorite():
    assert american_to_prob(-200) == pytest.approx(2 / 3, rel=1e-3)


def test_american_to_prob_positive_underdog():
    assert american_to_prob(150) == pytest.approx(0.4)


def test_american_to_prob_pickem():
    assert american_to_prob(-110) == pytest.approx(0.5238, rel=1e-3)


def test_decimal_to_prob():
    assert decimal_to_prob(2.0) == 0.5


def test_decimal_to_prob_invalid():
    with pytest.raises(ValueError):
        decimal_to_prob(1.0)


def test_zero_odds_invalid():
    with pytest.raises(ValueError):
        american_to_decimal(0)


# ── No-vig conversion ─────────────────────────────────────────────────────────

def test_no_vig_symmetric_market():
    p_over, p_under = no_vig_prob(-110, -110)
    assert p_over == pytest.approx(0.5)
    assert p_under == pytest.approx(0.5)


def test_no_vig_asymmetric_market():
    # Heavy favorite over: -200/+165 → over true ~2/3
    p_over, p_under = no_vig_prob(-200, 165)
    assert p_over + p_under == pytest.approx(1.0)
    assert p_over > p_under


# ── Expected value ────────────────────────────────────────────────────────────

def test_ev_breakeven_at_market_prob():
    # At -110, breakeven model prob is 0.5238. EV should be ~0.
    assert expected_value(0.5238, -110) == pytest.approx(0.0, abs=1e-3)


def test_ev_positive_with_edge():
    # 60% model prob at -110 (52.4% implied) is a strong edge.
    assert expected_value(0.60, -110) > 0


def test_ev_negative_without_edge():
    assert expected_value(0.40, -110) < 0


def test_edge_helper():
    assert edge(0.55, -110) == pytest.approx(0.55 - 0.5238, rel=1e-3)


# ── Kelly ─────────────────────────────────────────────────────────────────────

def test_kelly_no_edge_returns_zero():
    # If model_prob equals market implied prob, full-Kelly should be ~0.
    assert kelly_fraction(0.5238, -110) == pytest.approx(0.0, abs=1e-3)


def test_kelly_negative_edge_clamped_to_zero():
    assert kelly_fraction(0.40, -110) == 0.0


def test_kelly_positive_edge_known_value():
    # b=1, p=0.6, q=0.4 → f* = (1*0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 100) == pytest.approx(0.2)


def test_fractional_kelly_quarter():
    full = kelly_fraction(0.6, 100)
    quarter = fractional_kelly(0.6, 100, fraction=0.25)
    assert quarter == pytest.approx(full * 0.25)


def test_kelly_bankroll_settles_win():
    br = KellyBankroll(bankroll=1000.0, fraction=0.25)
    stake = br.stake(0.6, 100)
    assert stake > 0
    profit = br.settle(won=True, stake=stake, american_odds=100)
    assert profit == pytest.approx(stake)
    assert br.bankroll == pytest.approx(1000.0 + stake)


def test_kelly_bankroll_settles_loss():
    br = KellyBankroll(bankroll=1000.0, fraction=0.25)
    stake = br.stake(0.6, 100)
    br.settle(won=False, stake=stake, american_odds=100)
    assert br.bankroll == pytest.approx(1000.0 - stake)


# ── Prop CSV loading ──────────────────────────────────────────────────────────

def test_load_prop_lines(tmp_path: Path):
    csv = tmp_path / "props.csv"
    csv.write_text(
        "player_id,stat_type,line,over_odds,under_odds\n"
        "P1,rec_yards,72.5,-110,-110\n"
        "P2,anytime_td,,250,\n"
        "P3,rush_yards,84.5,-115,-105\n"
    )
    lines = load_prop_lines(csv)
    assert len(lines) == 3
    assert lines[0].line == 72.5
    assert lines[1].stat_type == "anytime_td"
    assert lines[1].line is None
    assert lines[1].under_odds is None


def test_prop_line_validation_anytime_td_requires_odds():
    pl = PropLine(player_id="P1", stat_type="anytime_td", line=None, over_odds=None)
    with pytest.raises(ValueError):
        pl.validate()


def test_prop_line_validation_unknown_stat():
    pl = PropLine(player_id="P1", stat_type="weird_stat", line=10.0, over_odds=-110)
    with pytest.raises(ValueError):
        pl.validate()


# ── PropAnalyzer end-to-end ───────────────────────────────────────────────────

def _fake_sim(rec_yards_values: list[float], tds_values: list[int]) -> SimulationResult:
    """Build a SimulationResult with one player and N draws of stats."""
    draws = []
    for ry, td in zip(rec_yards_values, tds_values):
        draws.append(DrawRecord(
            home_score=21, away_score=17,
            home_pass_yards=250.0, away_pass_yards=200.0,
            home_rush_yards=100.0, away_rush_yards=90.0,
            home_pass_att=32, away_pass_att=28,
            player_stats=[{
                "player_id": "P1", "player_name": "WR Test",
                "team": "KC", "position": "WR", "is_home": True,
                "rec_yards": ry, "tds": td,
                "targets": 8, "receptions": 5,
                "carries": 0, "rush_yards": 0.0,
                "pass_attempts": 0, "completions": 0, "pass_yards": 0.0,
            }],
        ))
    return SimulationResult("KC", "LAC", draws)


def test_analyzer_picks_over_when_distribution_above_line():
    # 80% of draws above 60 → strong over.
    rec_yards = [40.0] * 200 + [80.0] * 800
    tds = [0] * 1000
    sim = _fake_sim(rec_yards, tds)

    pl = PropLine(player_id="P1", stat_type="rec_yards", line=60.0,
                  over_odds=-110, under_odds=-110)
    df = PropAnalyzer().analyze(sim, [pl])

    assert len(df) == 1
    assert df.iloc[0]["side"] == "over"
    assert df.iloc[0]["model_prob"] == pytest.approx(0.8)
    assert df.iloc[0]["ev"] > 0
    assert df.iloc[0]["kelly"] > 0


def test_analyzer_picks_under_when_distribution_below_line():
    rec_yards = [40.0] * 800 + [80.0] * 200
    tds = [0] * 1000
    sim = _fake_sim(rec_yards, tds)

    pl = PropLine(player_id="P1", stat_type="rec_yards", line=60.0,
                  over_odds=-110, under_odds=-110)
    df = PropAnalyzer().analyze(sim, [pl])

    assert df.iloc[0]["side"] == "under"
    assert df.iloc[0]["ev"] > 0


def test_analyzer_anytime_td():
    # 35% of draws have a TD.
    rec_yards = [50.0] * 1000
    tds = [0] * 650 + [1] * 300 + [2] * 50
    sim = _fake_sim(rec_yards, tds)

    pl = PropLine(player_id="P1", stat_type="anytime_td", line=None, over_odds=200)
    df = PropAnalyzer().analyze(sim, [pl])

    assert df.iloc[0]["side"] == "yes"
    assert df.iloc[0]["model_prob"] == pytest.approx(0.35, abs=1e-3)
    # +200 implies 0.333; 0.35 is a small edge → positive EV.
    assert df.iloc[0]["ev"] > 0


def test_analyzer_skips_unknown_player():
    sim = _fake_sim([50.0] * 100, [0] * 100)
    pl = PropLine(player_id="UNKNOWN", stat_type="rec_yards", line=50,
                  over_odds=-110, under_odds=-110)
    df = PropAnalyzer().analyze(sim, [pl])
    assert df.empty
