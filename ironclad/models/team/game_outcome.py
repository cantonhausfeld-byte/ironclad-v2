"""Game outcome model: win probability, spread, total.

Two implementations:
- GameOutcomeModel (XGBoost): trained on historical gold features
- StubGameOutcomeModel: odds-implied fallback used before training
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ironclad.config import (
    LEAGUE_HOME_WIN_PROB,
    LEAGUE_AVG_TOTAL,
    LEAGUE_AVG_HOME_MARGIN,
)
from ironclad.models.base import BaseModel
from ironclad.models.calibration import IsotonicCalibrator

logger = logging.getLogger(__name__)

# Historical std devs (2016–2023)
_MARGIN_STD = 14.1
_TOTAL_STD = 10.2

# Feature columns used by the model (differences: home - away)
TEAM_FEATURES = [
    "off_epa_diff",
    "def_epa_diff",
    "off_pass_epa_diff",
    "def_pass_epa_diff",
    "off_rush_epa_diff",
    "def_rush_epa_diff",
    "pass_rate_diff",
    "success_rate_diff",
    "points_pg_diff",
    "home_rest_days",
    "away_rest_days",
    "rest_advantage",
    "is_divisional",
    "altitude_ft",
    "is_dome",
    "temp_f",
    "wind_mph",
    "precip_in",
    "implied_total_from_odds",
    "spread_from_odds",
    "home_win_prob_from_odds",
]


class GameOutcomeModel(BaseModel):
    """XGBoost game outcome model with isotonic calibration."""
    name = "game_outcome"
    version = "v1.0"

    def __init__(self) -> None:
        self._clf: XGBClassifier | None = None      # home_win classification
        self._reg_margin: XGBRegressor | None = None # home_margin regression
        self._reg_total: XGBRegressor | None = None  # total_score regression
        self._calibrator = IsotonicCalibrator()
        self._margin_std = _MARGIN_STD
        self._total_std = _TOTAL_STD
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        Xf = _build_diff_features(X)
        feat_cols = [c for c in TEAM_FEATURES if c in Xf.columns]
        Xm = Xf[feat_cols].fillna(0)

        y_win = y["home_win"].astype(int)
        y_margin = y["home_margin"].astype(float)
        y_total = y["total_score"].astype(float)

        self._clf = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, n_jobs=-1,
        )
        self._clf.fit(Xm, y_win)

        self._reg_margin = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        self._reg_margin.fit(Xm, y_margin)

        self._reg_total = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        self._reg_total.fit(Xm, y_total)

        # Compute residual std devs from training set
        margin_pred = self._reg_margin.predict(Xm)
        total_pred = self._reg_total.predict(Xm)
        self._margin_std = float(np.std(y_margin.values - margin_pred))
        self._total_std = float(np.std(y_total.values - total_pred))

        self._fitted = True
        logger.info("GameOutcomeModel fitted on %d rows", len(Xm))

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.DataFrame) -> None:
        if not self._fitted:
            raise RuntimeError("fit() before calibrate()")
        Xf = _build_diff_features(X_cal)
        feat_cols = [c for c in TEAM_FEATURES if c in Xf.columns]
        Xm = Xf[feat_cols].fillna(0)
        raw_probs = self._clf.predict_proba(Xm)[:, 1]
        self._calibrator.fit(raw_probs, y_cal["home_win"].astype(int).values)
        logger.info("Calibration fitted on %d rows", len(Xm))

    def predict(self, X: pd.DataFrame) -> dict:
        if self._fitted and self._clf is not None:
            return self._predict_trained(X)
        return self._predict_stub(X)

    def _predict_trained(self, X: pd.DataFrame) -> dict:
        Xf = _build_diff_features(X)
        feat_cols = [c for c in TEAM_FEATURES if c in Xf.columns]
        Xm = Xf[feat_cols].fillna(0)

        raw_prob = float(self._clf.predict_proba(Xm)[0, 1])
        home_win_prob = float(self._calibrator.transform(np.array([raw_prob]))[0])
        home_margin_mean = float(self._reg_margin.predict(Xm)[0])
        total_mean = float(self._reg_total.predict(Xm)[0])

        return {
            "home_win_prob": np.clip(home_win_prob, 0.02, 0.98),
            "away_win_prob": 1.0 - np.clip(home_win_prob, 0.02, 0.98),
            "home_margin_mean": home_margin_mean,
            "home_margin_std": self._margin_std,
            "total_mean": max(20.0, total_mean),
            "total_std": self._total_std,
        }

    def _predict_stub(self, X: pd.DataFrame) -> dict:
        home_win_prob = _col(X, "home_win_prob_from_odds", LEAGUE_HOME_WIN_PROB)
        total_mean = _col(X, "implied_total_from_odds", LEAGUE_AVG_TOTAL)
        spread = _col(X, "spread_from_odds", -LEAGUE_AVG_HOME_MARGIN)
        return {
            "home_win_prob": float(home_win_prob),
            "away_win_prob": 1.0 - float(home_win_prob),
            "home_margin_mean": -float(spread),
            "home_margin_std": _MARGIN_STD,
            "total_mean": float(total_mean),
            "total_std": _TOTAL_STD,
        }


def _build_diff_features(X: pd.DataFrame) -> pd.DataFrame:
    """Pivot wide: each row has home_ and away_ prefixed columns → compute diffs."""
    # If already pivoted (one row per game), just map directly
    out = X.copy()
    pairs = [
        ("off_epa_diff",        "off_epa_per_play_l4",     None),
        ("def_epa_diff",        "def_epa_per_play_l4",     None),
        ("off_pass_epa_diff",   "off_pass_epa_l4",         None),
        ("def_pass_epa_diff",   "def_pass_epa_l4",         None),
        ("off_rush_epa_diff",   "off_rush_epa_l4",         None),
        ("def_rush_epa_diff",   "def_rush_epa_l4",         None),
        ("pass_rate_diff",      "off_pass_rate_l4",        None),
        ("success_rate_diff",   "off_success_rate_l4",     None),
        ("points_pg_diff",      "off_points_per_game_l4",  None),
    ]
    # If input has home_ / away_ prefix form (from training loader):
    for diff_col, feat_col, _ in pairs:
        home_col = f"home_{feat_col}"
        away_col = f"away_{feat_col}"
        if home_col in X.columns and away_col in X.columns:
            out[diff_col] = X[home_col].fillna(0) - X[away_col].fillna(0)
        elif feat_col in X.columns:
            out[diff_col] = X[feat_col].fillna(0)
        else:
            out[diff_col] = 0.0

    # Rest
    if "home_rest_days" not in out.columns:
        out["home_rest_days"] = X.get("rest_days", pd.Series(7, index=X.index))
    if "away_rest_days" not in out.columns:
        out["away_rest_days"] = 7
    out["rest_advantage"] = out.get("home_rest_days", 7) - out.get("away_rest_days", 7)

    for col, default in [
        ("is_divisional", False), ("altitude_ft", 0),
        ("is_dome", False), ("temp_f", 65.0),
        ("wind_mph", 8.0), ("precip_in", 0.0),
        ("implied_total_from_odds", LEAGUE_AVG_TOTAL),
        ("spread_from_odds", 0.0),
        ("home_win_prob_from_odds", LEAGUE_HOME_WIN_PROB),
    ]:
        if col not in out.columns:
            out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    return out


def _col(X: pd.DataFrame, col: str, default: float) -> float:
    if col in X.columns and not X[col].isna().all():
        val = X[col].iloc[0]
        if pd.notna(val):
            return float(val)
    return default
