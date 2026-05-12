"""Unit tests for GameOutcomeEnsemble meta-learner."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ironclad.models.team.ensemble import GameOutcomeEnsemble


def _mock_base_model(win_prob: float, margin: float, total: float) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = {
        "home_win_prob": win_prob,
        "away_win_prob": round(1.0 - win_prob, 4),
        "home_margin_mean": margin,
        "home_margin_std": 13.45,
        "total_mean": total,
        "total_std": 13.45,
    }
    return m


def _make_X(n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "home_off_epa_per_play_l4": rng.normal(0, 0.1, n),
        "away_off_epa_per_play_l4": rng.normal(0, 0.1, n),
    })


def _make_y(n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "home_win": rng.integers(0, 2, n),
        "home_margin": rng.normal(0, 10, n),
        "total_score": rng.normal(45, 10, n),
    })


def _fitted_ensemble(win_prob_xgb=0.6, win_prob_lgbm=0.55, margin_xgb=7.0, margin_lgbm=3.0,
                      total_xgb=48.0, total_lgbm=44.0, n=4):
    xgb_mock = _mock_base_model(win_prob_xgb, margin_xgb, total_xgb)
    lgbm_mock = _mock_base_model(win_prob_lgbm, margin_lgbm, total_lgbm)

    ensemble = GameOutcomeEnsemble()
    with patch("ironclad.models.registry.ModelRegistry") as MockReg:
        MockReg.return_value.load.side_effect = [xgb_mock, lgbm_mock]
        ensemble.fit(_make_X(n), _make_y(n))
    return ensemble, xgb_mock, lgbm_mock


def test_ensemble_predict_returns_required_keys():
    """predict() must return all 6 keys matching GameOutcomeModel's output contract."""
    ensemble, xgb_mock, lgbm_mock = _fitted_ensemble()
    ensemble._xgb = xgb_mock
    ensemble._lgbm = lgbm_mock

    result = ensemble.predict(_make_X(1))

    required = {"home_win_prob", "away_win_prob", "home_margin_mean",
                 "home_margin_std", "total_mean", "total_std"}
    assert required.issubset(result.keys())


def test_ensemble_win_prob_between_zero_and_one():
    """home_win_prob must be in [0, 1] after calibration."""
    ensemble, xgb_mock, lgbm_mock = _fitted_ensemble()
    ensemble._xgb = xgb_mock
    ensemble._lgbm = lgbm_mock

    result = ensemble.predict(_make_X(1))

    assert 0.0 <= result["home_win_prob"] <= 1.0
    assert abs(result["home_win_prob"] + result["away_win_prob"] - 1.0) < 1e-4


def test_ensemble_fit_trains_meta_lr():
    """After fit(), _meta.coef_ should be set (not None)."""
    ensemble, _, _ = _fitted_ensemble(n=6)
    assert ensemble._meta.coef_ is not None
    assert ensemble._fitted is True


def test_ensemble_averages_base_margins():
    """Ensemble margin = average of XGB and LightGBM margins."""
    xgb_margin = 7.0
    lgbm_margin = 3.0
    expected_avg = (xgb_margin + lgbm_margin) / 2.0

    ensemble, xgb_mock, lgbm_mock = _fitted_ensemble(
        margin_xgb=xgb_margin, margin_lgbm=lgbm_margin, n=6
    )
    ensemble._xgb = xgb_mock
    ensemble._lgbm = lgbm_mock

    result = ensemble.predict(_make_X(1))

    assert abs(result["home_margin_mean"] - expected_avg) < 1e-6


def test_ensemble_fallback_when_not_fitted():
    """predict() before fit() should return a dict without raising."""
    ensemble = GameOutcomeEnsemble()
    assert not ensemble._fitted

    result = ensemble.predict(_make_X(1))

    assert isinstance(result, dict)
    assert "home_win_prob" in result
