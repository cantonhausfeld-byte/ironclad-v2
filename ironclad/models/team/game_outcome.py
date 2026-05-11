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
from ironclad.models.calibration import IsotonicCalibrator, PlattCalibrator

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
    "turnover_diff",
    # Phase 2A: situational features
    "cpoe_diff",
    "neutral_epa_diff",
    "epa_pass_early_diff",
    "epa_rush_early_diff",
    "epa_third_down_diff",
    "third_down_pct_diff",
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
    # Phase 2B: season-to-date EPA, red zone, bye week, surface
    "off_epa_std_diff",
    "def_epa_std_diff",
    "off_epa_rz_diff",
    "def_epa_rz_diff",
    "home_is_post_bye",
    "away_is_post_bye",
    "surface_grass",
    # Phase 2E: Elo team rating + game week
    "elo_diff",
    "game_week",
    # Phase 2G: defense-side feature completion
    "def_success_rate_diff",
    "def_cpoe_allowed_diff",
    "def_neutral_epa_diff",
    "def_epa_rush_early_diff",
    "def_epa_third_down_diff",
    "def_third_down_pct_diff",
    "def_turnovers_forced_diff",
    "yards_per_play_diff",
    "def_epa_pass_early_diff",
    "def_sack_rate_diff",
    "fourth_down_att_diff",
]

# Win-probability classifier omits home_win_prob_from_odds to avoid circular dependency.
# XGBoost uses it as a near-perfect leaf feature, collapsing 75%+ of outputs to < 5% or > 95%.
# Spread encodes the same directional signal without causing output saturation.
CLF_FEATURES = [f for f in TEAM_FEATURES if f != "home_win_prob_from_odds"]


class GameOutcomeModel(BaseModel):
    """XGBoost game outcome model with isotonic calibration."""
    name = "game_outcome"
    version = "v1.0"

    def __init__(
        self, *,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
    ) -> None:
        self._hp = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
        )
        self._clf: XGBClassifier | None = None
        self._reg_margin: XGBRegressor | None = None
        self._reg_total: XGBRegressor | None = None
        self._calibrator = PlattCalibrator()
        self._margin_std = _MARGIN_STD
        self._total_std = _TOTAL_STD
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        Xf = _build_diff_features(X)
        clf_cols = [c for c in CLF_FEATURES if c in Xf.columns]
        reg_cols = [c for c in TEAM_FEATURES if c in Xf.columns]
        Xm_clf = Xf[clf_cols].fillna(0)
        Xm_reg = Xf[reg_cols].fillna(0)

        y_win = y["home_win"].astype(int)
        y_margin = y["home_margin"].astype(float)
        y_total = y["total_score"].astype(float)

        self._clf = XGBClassifier(
            **self._hp,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        self._clf.fit(Xm_clf, y_win)

        self._reg_margin = XGBRegressor(
            **self._hp,
            random_state=42, n_jobs=-1,
        )
        self._reg_margin.fit(Xm_reg, y_margin)

        self._reg_total = XGBRegressor(
            **self._hp,
            random_state=42, n_jobs=-1,
        )
        self._reg_total.fit(Xm_reg, y_total)

        # Compute residual std devs from training set
        margin_pred = self._reg_margin.predict(Xm_reg)
        total_pred = self._reg_total.predict(Xm_reg)
        self._margin_std = float(np.std(y_margin.values - margin_pred))
        self._total_std = float(np.std(y_total.values - total_pred))

        self._fitted = True
        logger.info("GameOutcomeModel fitted on %d rows (clf=%d feats, reg=%d feats)",
                    len(Xm_clf), len(clf_cols), len(reg_cols))

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.DataFrame) -> None:
        if not self._fitted:
            raise RuntimeError("fit() before calibrate()")
        Xf = _build_diff_features(X_cal)
        clf_cols = [c for c in CLF_FEATURES if c in Xf.columns]
        Xm = Xf[clf_cols].fillna(0)
        raw_probs = self._clf.predict_proba(Xm)[:, 1]
        self._calibrator.fit(raw_probs, y_cal["home_win"].astype(int).values)
        logger.info("Calibration fitted on %d rows", len(Xm))

    def calibrate_from_probs(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic calibrator from pre-computed probabilities (e.g. temperature-scaled)."""
        self._calibrator.fit(probs, labels)
        logger.info("Calibration fitted from %d pre-computed probs", len(probs))

    def predict(self, X: pd.DataFrame) -> dict:
        if self._fitted and self._clf is not None:
            return self._predict_trained(X)
        return self._predict_stub(X)

    def _predict_trained(self, X: pd.DataFrame) -> dict:
        Xf = _build_diff_features(X)
        clf_cols = [c for c in CLF_FEATURES if c in Xf.columns]
        reg_cols = [c for c in TEAM_FEATURES if c in Xf.columns]
        Xm_clf = Xf[clf_cols].fillna(0)
        Xm_reg = Xf[reg_cols].fillna(0)

        raw_prob = float(self._clf.predict_proba(Xm_clf)[0, 1])
        platt_prob = float(np.clip(self._calibrator.transform(np.array([raw_prob]))[0], 0.02, 0.98))

        # Blend win prob 50/50 with Vegas-implied win prob when available.
        # XGBoost remains overconfident at extreme spreads; anchoring to the market
        # reduces systematic error for heavy favorites/underdogs.
        vegas_win_prob = _col(X, "home_win_prob_from_odds", None)
        if vegas_win_prob is not None:
            home_win_prob = float(np.clip(0.5 * platt_prob + 0.5 * float(vegas_win_prob), 0.02, 0.98))
        else:
            home_win_prob = platt_prob

        home_margin_mean = float(self._reg_margin.predict(Xm_reg)[0])
        total_mean_model = float(self._reg_total.predict(Xm_reg)[0])
        # Blend model total with Vegas-implied total (60/40).
        vegas_total = _col(X, "implied_total_from_odds", total_mean_model)
        total_mean = 0.4 * total_mean_model + 0.6 * vegas_total

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": 1.0 - home_win_prob,
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
        ("turnover_diff",       "off_turnovers_l4",        None),
        # Phase 2A: situational features
        ("cpoe_diff",           "off_cpoe_l4",             None),
        ("neutral_epa_diff",    "off_neutral_epa_l4",      None),
        ("epa_pass_early_diff", "off_epa_pass_early_l4",   None),
        ("epa_rush_early_diff", "off_epa_rush_early_l4",   None),
        ("epa_third_down_diff", "off_epa_third_down_l4",   None),
        ("third_down_pct_diff", "off_third_down_pct_l4",   None),
        # Phase 2G: defense-side completions
        ("def_success_rate_diff",    "def_success_rate_l4",    None),
        ("def_cpoe_allowed_diff",    "def_cpoe_allowed_l4",    None),
        ("def_neutral_epa_diff",     "def_neutral_epa_l4",     None),
        ("def_epa_rush_early_diff",  "def_epa_rush_early_l4",  None),
        ("def_epa_third_down_diff",  "def_epa_third_down_l4",  None),
        ("def_third_down_pct_diff",  "def_third_down_pct_l4",  None),
        ("def_turnovers_forced_diff","def_turnovers_forced_l4",None),
        ("yards_per_play_diff",      "off_yards_per_play_l4",  None),
        ("def_epa_pass_early_diff",  "def_epa_pass_early_l4",  None),
        ("def_sack_rate_diff",       "def_sack_rate_l4",       None),
        ("fourth_down_att_diff",     "off_fourth_down_att_l4", None),
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
        ("surface_grass", False),
    ]:
        # Training data has home_/away_ prefixes — try home-side first
        home_col = f"home_{col}"
        if col not in out.columns or out[col].isna().all():
            if home_col in X.columns:
                out[col] = X[home_col]
            elif col not in out.columns:
                out[col] = default
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    # Phase 2B: season-to-date EPA diffs
    for diff_col, feat_col in [
        ("off_epa_std_diff", "off_epa_per_play_std"),
        ("def_epa_std_diff", "def_epa_per_play_std"),
        ("off_epa_rz_diff",  "off_epa_rz_l4"),
        ("def_epa_rz_diff",  "def_epa_rz_l4"),
    ]:
        home_col = f"home_{feat_col}"
        away_col = f"away_{feat_col}"
        if home_col in X.columns and away_col in X.columns:
            out[diff_col] = X[home_col].fillna(0) - X[away_col].fillna(0)
        elif feat_col in X.columns:
            out[diff_col] = X[feat_col].fillna(0)
        else:
            out[diff_col] = 0.0

    # Phase 2B: post-bye flags (rest_days >= 14 means team had a bye week)
    home_rest = pd.to_numeric(out.get("home_rest_days", 7), errors="coerce").fillna(7)
    away_rest = pd.to_numeric(out.get("away_rest_days", 7), errors="coerce").fillna(7)
    out["home_is_post_bye"] = (home_rest >= 14).astype(int)
    out["away_is_post_bye"] = (away_rest >= 14).astype(int)

    # Phase 2E: Elo diff (home_elo_pre_game - away_elo_pre_game)
    home_elo_col = "home_elo_pre_game"
    away_elo_col = "away_elo_pre_game"
    if home_elo_col in X.columns and away_elo_col in X.columns:
        out["elo_diff"] = X[home_elo_col].fillna(1500) - X[away_elo_col].fillna(1500)
    elif "elo_pre_game" in X.columns:
        out["elo_diff"] = X["elo_pre_game"].fillna(0)
    else:
        out["elo_diff"] = 0.0

    # Phase 2E: game week (same for both teams — raw week number, not a diff)
    if "home_week" in X.columns:
        out["game_week"] = pd.to_numeric(X["home_week"], errors="coerce").fillna(9)
    elif "week" in X.columns:
        out["game_week"] = pd.to_numeric(X["week"], errors="coerce").fillna(9)
    else:
        out["game_week"] = 9.0

    return out


def _col(X: pd.DataFrame, col: str, default: float | None) -> float | None:
    if col in X.columns and not X[col].isna().all():
        val = X[col].iloc[0]
        if pd.notna(val):
            return float(val)
    return default
