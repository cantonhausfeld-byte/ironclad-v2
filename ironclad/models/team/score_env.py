"""Score environment model: pace, pass rate, scoring context."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ironclad.config import LEAGUE_PRIORS
from ironclad.models.base import BaseModel
from ironclad.models.team.game_outcome import _sample_weights

logger = logging.getLogger(__name__)

FEATURES = [
    "off_epa_per_play_l4", "off_pass_epa_l4", "off_rush_epa_l4",
    "off_pass_rate_l4", "off_success_rate_l4", "off_points_per_game_l4",
    "def_epa_per_play_l4", "def_sack_rate_l4",
    "opp_def_pass_epa_l4", "opp_def_rush_epa_l4",
    "implied_total_from_odds", "spread_from_odds",
    "rest_days", "is_dome", "temp_f", "wind_mph",
]


class ScoreEnvironmentModel(BaseModel):
    """XGBoost multi-target score environment model."""
    name = "score_env"
    version = "v1.0"

    def __init__(self) -> None:
        self._reg_pass_rate: XGBRegressor | None = None
        self._reg_plays: XGBRegressor | None = None
        self._reg_team_total: XGBRegressor | None = None
        self._reg_sack_rate: XGBRegressor | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        feat_cols = [c for c in FEATURES if c in X.columns]
        Xm = X[feat_cols].fillna(0)
        w = _sample_weights(X, season_col="season")

        targets = {
            "_reg_pass_rate":  "target_pass_rate",
            "_reg_team_total": "target_points_scored",
        }
        for attr, tgt in targets.items():
            if tgt not in y.columns:
                logger.warning("Missing target %s, skipping", tgt)
                continue
            mask = y[tgt].notna()
            w_masked = w[mask] if w is not None else None
            reg = XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            )
            reg.fit(Xm[mask], y[tgt][mask], sample_weight=w_masked)
            setattr(self, attr, reg)

        self._fitted = True
        logger.info("ScoreEnvironmentModel fitted on %d rows", len(Xm))

    def predict(self, X: pd.DataFrame) -> dict:
        if self._fitted and self._reg_pass_rate is not None:
            return self._predict_trained(X)
        return self._predict_stub(X)

    def _predict_trained(self, X: pd.DataFrame) -> dict:
        # Use the model's own feature list so we never pass columns the trained
        # booster doesn't know about (FEATURES may have grown since training).
        train_cols = list(self._reg_pass_rate.feature_names_in_)
        Xm = X.reindex(columns=train_cols, fill_value=0.0).fillna(0.0).astype(float)

        def pred(reg, default):
            if reg is None:
                return default
            return max(0.0, float(reg.predict(Xm)[0]))

        pass_rate = np.clip(pred(self._reg_pass_rate, LEAGUE_PRIORS["pass_rate"]), 0.3, 0.85)
        total_plays = np.clip(pred(self._reg_plays, LEAGUE_PRIORS["total_plays"]), 40, 90)
        sack_rate = np.clip(pred(self._reg_sack_rate, LEAGUE_PRIORS["sack_rate"]), 0.01, 0.20)
        team_total = max(7.0, pred(self._reg_team_total, LEAGUE_PRIORS["points_per_game"]))

        return {
            "pass_rate_projected": float(pass_rate),
            "total_plays_projected": float(total_plays),
            "sack_rate_projected": float(sack_rate),
            "team_total_projected": float(team_total),
        }

    def _predict_stub(self, X: pd.DataFrame) -> dict:
        def col(c, default):
            if X is None or c not in X.columns:
                return default
            val = X[c].iloc[0] if not X.empty else None
            return float(val) if pd.notna(val) else default

        team_total = col("implied_total_from_odds", LEAGUE_PRIORS["points_per_game"] * 2) / 2.0
        return {
            "pass_rate_projected": col("off_pass_rate_l4", LEAGUE_PRIORS["pass_rate"]),
            "total_plays_projected": LEAGUE_PRIORS["total_plays"],
            "sack_rate_projected": col("def_sack_rate_l4", LEAGUE_PRIORS["sack_rate"]),
            "team_total_projected": max(7.0, team_total),
        }
