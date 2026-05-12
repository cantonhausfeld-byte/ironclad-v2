"""Ensemble game-outcome model: ridge meta-learner on XGB + LightGBM predictions."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ironclad.models.base import BaseModel
from ironclad.models.calibration import PlattCalibrator
from ironclad.models.team.game_outcome import GameOutcomeModel
from ironclad.models.team.game_outcome_lgbm import GameOutcomeLGBM

logger = logging.getLogger(__name__)

_MARGIN_STD_DEFAULT = 13.45
_TOTAL_STD_DEFAULT = 13.45


class GameOutcomeEnsemble(BaseModel):
    """Ridge meta-learner stacking XGB + LightGBM win probabilities.

    Both base models apply their own Vegas blend internally.  The meta-learner
    learns the optimal linear combination — no additional Vegas blend is applied
    on top (that would double-count the market signal).

    Prerequisites: GameOutcomeModel and GameOutcomeLGBM must be saved in the
    ModelRegistry before calling fit().
    """

    name = "game_outcome_ensemble"
    version = "v1.0"

    def __init__(self) -> None:
        self._meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, penalty="l2")
        self._calibrator = PlattCalibrator()
        self._xgb: GameOutcomeModel | None = None
        self._lgbm: GameOutcomeLGBM | None = None
        self._margin_std = _MARGIN_STD_DEFAULT
        self._total_std = _TOTAL_STD_DEFAULT
        self._fitted = False

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Load base models from registry; train meta-learner on X.

        X must be the same wide-pivoted DataFrame as GameOutcomeModel.fit() expects
        (one row per game, home_/away_ prefixed columns).
        y must have columns: home_win, home_margin, total_score.
        """
        from ironclad.models.registry import ModelRegistry
        reg = ModelRegistry()
        try:
            self._xgb = reg.load("game_outcome")
            logger.info("Loaded GameOutcomeModel from registry")
        except Exception as exc:
            raise RuntimeError(
                "GameOutcomeModel not found in registry. "
                "Run `ironclad train --model game-outcome` first."
            ) from exc
        try:
            self._lgbm = reg.load("game_outcome_lgbm")
            logger.info("Loaded GameOutcomeLGBM from registry")
        except Exception as exc:
            raise RuntimeError(
                "GameOutcomeLGBM not found in registry. "
                "Run `ironclad train --model game-outcome-lgbm` first."
            ) from exc

        meta_X, xgb_margins, lgbm_margins, xgb_totals, lgbm_totals = self._base_preds(X)
        labels = y["home_win"].astype(int).values
        self._meta.fit(meta_X, labels)
        logger.info("Meta-learner fitted. Coefficients: %s", self._meta.coef_)

        avg_margins = (xgb_margins + lgbm_margins) / 2
        avg_totals = (xgb_totals + lgbm_totals) / 2
        margin_resid = y["home_margin"].values - avg_margins
        total_resid = y["total_score"].values - avg_totals
        self._margin_std = float(np.std(margin_resid))
        self._total_std = float(np.std(total_resid))
        self._fitted = True
        logger.info("Ensemble fitted. margin_std=%.2f total_std=%.2f", self._margin_std, self._total_std)

    def calibrate_from_probs(self, probs: np.ndarray, labels: np.ndarray) -> None:
        self._calibrator.fit(probs, labels)

    def predict(self, X: pd.DataFrame) -> dict:
        if not self._fitted or self._xgb is None or self._lgbm is None:
            logger.warning("Ensemble not fitted; falling back to GameOutcomeModel stub")
            return GameOutcomeModel().predict(X)

        meta_X, xgb_margins, lgbm_margins, xgb_totals, lgbm_totals = self._base_preds(X)
        raw_prob = self._meta.predict_proba(meta_X)[:, 1]
        cal_prob = self._calibrator.transform(raw_prob)
        home_prob = float(np.clip(cal_prob[0], 0.02, 0.98))

        return {
            "home_win_prob":    round(home_prob, 4),
            "away_win_prob":    round(1.0 - home_prob, 4),
            "home_margin_mean": float((xgb_margins[0] + lgbm_margins[0]) / 2),
            "home_margin_std":  self._margin_std,
            "total_mean":       float(max(20.0, (xgb_totals[0] + lgbm_totals[0]) / 2)),
            "total_std":        self._total_std,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _base_preds(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (meta_X, xgb_margins, lgbm_margins, xgb_totals, lgbm_totals)."""
        xgb_probs, lgbm_probs = [], []
        xgb_margins, lgbm_margins = [], []
        xgb_totals, lgbm_totals = [], []
        for i in range(len(X)):
            row = X.iloc[[i]]
            xp = self._xgb.predict(row)
            lp = self._lgbm.predict(row)
            xgb_probs.append(xp["home_win_prob"])
            lgbm_probs.append(lp["home_win_prob"])
            xgb_margins.append(xp["home_margin_mean"])
            lgbm_margins.append(lp["home_margin_mean"])
            xgb_totals.append(xp["total_mean"])
            lgbm_totals.append(lp["total_mean"])

        meta_X = np.column_stack([xgb_probs, lgbm_probs])
        return (
            meta_X,
            np.array(xgb_margins), np.array(lgbm_margins),
            np.array(xgb_totals),  np.array(lgbm_totals),
        )
