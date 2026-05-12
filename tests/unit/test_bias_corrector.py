"""Tests for TeamBiasCorrector and recency weight helper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ironclad.models.bias_corrector import TeamBiasCorrector
from ironclad.models.team.game_outcome import _sample_weights


# ── Recency weights ───────────────────────────────────────────────────────────

def test_recency_weights_most_recent_is_one():
    X = pd.DataFrame({"home_season": [2020, 2021, 2022, 2023]})
    w = _sample_weights(X)
    assert w is not None
    assert abs(w[-1] - 1.0) < 1e-6   # most recent season = weight 1.0


def test_recency_weights_decay_monotonically():
    X = pd.DataFrame({"home_season": [2019, 2020, 2021, 2022, 2023]})
    w = _sample_weights(X)
    assert w is not None
    assert list(w) == sorted(w)       # weights increase toward the most recent


def test_recency_weights_missing_col_returns_none():
    X = pd.DataFrame({"off_epa": [0.1, 0.2]})
    assert _sample_weights(X) is None


def test_recency_weights_player_season_col():
    X = pd.DataFrame({"season": [2021, 2022, 2023]})
    w = _sample_weights(X, season_col="season")
    assert w is not None
    assert abs(w[-1] - 1.0) < 1e-6


# ── TeamBiasCorrector ─────────────────────────────────────────────────────────

def _make_backtest(home_team="KC", away_team="LAC", pred=3.0, actual=10.0, n=20):
    return pd.DataFrame({
        "home_team":           [home_team] * n,
        "away_team":           [away_team] * n,
        "home_margin_pred":    [pred] * n,
        "home_margin_actual":  [actual] * n,
    })


def test_bias_corrector_fit_sets_fitted():
    bc = TeamBiasCorrector()
    bc.fit(_make_backtest())
    assert bc._fitted is True


def test_bias_corrector_correct_adjusts_toward_residual():
    # KC consistently outperforms by 7 pts when home → corrected margin higher
    bc = TeamBiasCorrector()
    bc.fit(_make_backtest(home_team="KC", pred=3.0, actual=10.0))  # residual +7
    raw = 3.0
    corrected = bc.correct("KC", "LAC", raw)
    assert corrected > raw


def test_bias_corrector_away_team_lowers_margin():
    # If KC (away) typically outperforms +7 at home, they should lower home margin
    bc = TeamBiasCorrector()
    bc.fit(_make_backtest(home_team="KC", pred=3.0, actual=10.0))
    raw = 3.0
    corrected = bc.correct("LAC", "KC", raw)   # KC is now away team
    assert corrected < raw


def test_bias_corrector_no_data_returns_unchanged():
    bc = TeamBiasCorrector()
    assert bc.correct("KC", "LAC", 5.0) == 5.0


def test_bias_corrector_unknown_team_zero_adjustment():
    bc = TeamBiasCorrector()
    bc.fit(_make_backtest(home_team="KC"))
    # NYG has no backtest data → no adjustment
    result = bc.correct("NYG", "NYJ", 3.0)
    assert result == 3.0


def test_bias_corrector_missing_columns_does_not_crash():
    bc = TeamBiasCorrector()
    bad_df = pd.DataFrame({"home_team": ["KC"], "away_team": ["LAC"]})
    bc.fit(bad_df)   # missing required columns → warning, no fit
    assert not bc._fitted
