"""Spread sign convention: silver/gold spreads are positive when the HOME team is favored.

(nflverse spread_line convention; Odds API spreads are negated in silver.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ironclad.eval.metrics import ats_record, ou_record
from ironclad.models.team.game_outcome import GameOutcomeModel
from ironclad.models.team.game_outcome_lgbm import GameOutcomeLGBM


def test_ats_grades_the_models_picks():
    spread = np.array([7.0, 7.0, -3.0, 3.0, 3.0])     # home favored by 7, 7; away by 3; home by 3, 3
    pred = np.array([10.0, 2.0, 0.0, 5.0, 3.0])        # model: home, away, home, home, no edge
    actual = np.array([14.0, 14.0, -7.0, 3.0, 0.0])
    # g1 home -7 covers (14>7): pick home -> win
    # g2 pick away +7; home won by 14 -> loss
    # g3 pick home +3; home lost by 7 -> loss
    # g4 pick home -3; won by exactly 3 -> push
    # g5 model margin == spread -> no pick
    assert ats_record(pred, spread, actual) == {
        "wins": 1, "losses": 2, "pushes": 1, "pct": round(1 / 3, 4), "n": 4,
    }


def test_ats_perfect_foresight_wins_every_bet():
    rng = np.random.default_rng(0)
    spread = rng.normal(2, 6, 300).round() + 0.5
    actual = rng.normal(2, 13, 300).round()
    rec = ats_record(actual.astype(float), spread, actual)
    assert rec["losses"] == 0 and rec["wins"] == rec["n"]


def test_ou_grades_the_models_picks():
    line = np.array([45.0, 45.0, 45.0, 45.0])
    pred = np.array([50.0, 40.0, 50.0, 46.0])  # over, under, over, over
    actual = np.array([52.0, 52.0, 30.0, 45.0])
    assert ou_record(pred, line, actual) == {
        "wins": 1, "losses": 2, "pushes": 1, "pct": round(1 / 3, 4), "n": 4,
    }


def _home_row(spread, win_prob):
    return pd.DataFrame([{
        "spread_from_odds": spread,
        "home_win_prob_from_odds": win_prob,
        "implied_total_from_odds": 45.0,
    }])


def test_stub_margin_follows_spread_sign():
    for model in (GameOutcomeModel(), GameOutcomeLGBM()):
        fav = model.predict(_home_row(7.0, 0.72))
        dog = model.predict(_home_row(-7.0, 0.28))
        assert fav["home_margin_mean"] == 7.0
        assert dog["home_margin_mean"] == -7.0
        # Margin and win probability must agree on who is favored
        assert (fav["home_margin_mean"] > 0) == (fav["home_win_prob"] > 0.5)


def test_stub_default_margin_is_home_field_advantage():
    from ironclad.config import LEAGUE_AVG_HOME_MARGIN
    out = GameOutcomeModel().predict(pd.DataFrame([{"temp_f": 60.0}]))
    assert out["home_margin_mean"] == LEAGUE_AVG_HOME_MARGIN > 0
