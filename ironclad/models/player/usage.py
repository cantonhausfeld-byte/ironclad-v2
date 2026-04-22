"""Player usage model: projected targets, carries, snap share."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ironclad.models.base import BaseModel

logger = logging.getLogger(__name__)

_POSITION_PRIORS = {
    "QB":  {"targets": 0.0,  "carries": 3.5,  "pass_attempts": 32.0},
    "RB":  {"targets": 3.5,  "carries": 12.0, "pass_attempts": 0.0},
    "WR":  {"targets": 5.5,  "carries": 0.2,  "pass_attempts": 0.0},
    "TE":  {"targets": 4.0,  "carries": 0.0,  "pass_attempts": 0.0},
    "FB":  {"targets": 1.5,  "carries": 2.0,  "pass_attempts": 0.0},
}
_DEFAULT_PRIOR = {"targets": 2.0, "carries": 1.0, "pass_attempts": 0.0}

FEATURES = [
    "availability", "depth_team",
    "target_share_l4", "carry_share_l4", "air_yards_share_l4",
    "route_rate_l4", "redzone_target_share_l4", "redzone_carry_share_l4",
    "opp_def_pass_epa_l4", "opp_def_rush_epa_l4",
    "team_off_pass_rate_l4", "team_implied_total",
    "is_home",
]


class PlayerUsageModel(BaseModel):
    """XGBoost player usage model (per-position)."""
    name = "player_usage"
    version = "v1.0"

    def __init__(self) -> None:
        self._regs: dict[str, dict[str, XGBRegressor]] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Train separate regressors per position × target combination."""
        feat_cols = [c for c in FEATURES if c in X.columns]
        positions = X["position"].unique() if "position" in X.columns else ["WR"]

        target_map = {
            "targets":        "target_targets",
            "carries":        "target_carries",
            "pass_attempts":  "target_pass_att",
        }

        for pos in positions:
            mask_pos = X["position"] == pos
            if mask_pos.sum() < 20:
                continue
            Xp = X[mask_pos][feat_cols].fillna(0)
            self._regs[pos] = {}
            for out_key, tgt_col in target_map.items():
                if tgt_col not in y.columns:
                    continue
                yp = y[tgt_col][mask_pos]
                mask_valid = yp.notna()
                if mask_valid.sum() < 20:
                    continue
                reg = XGBRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                )
                reg.fit(Xp[mask_valid], yp[mask_valid])
                self._regs[pos][out_key] = reg

        self._fitted = True
        logger.info("PlayerUsageModel fitted for positions: %s", list(self._regs.keys()))

    def predict(self, X: pd.DataFrame) -> dict:
        pos = str(X["position"].iloc[0]) if "position" in X.columns else "WR"
        availability = _col(X, "availability", 1.0)

        if availability == 0.0:
            return {"targets_projected": 0.0, "carries_projected": 0.0,
                    "pass_attempts_projected": 0.0, "availability": 0.0}

        if self._fitted and pos in self._regs and self._regs[pos]:
            return self._predict_trained(X, pos, availability)
        return self._predict_stub(X, pos, availability)

    def _predict_trained(self, X: pd.DataFrame, pos: str, availability: float) -> dict:
        feat_cols = [c for c in FEATURES if c in X.columns]
        Xm = X[feat_cols].fillna(0)
        regs = self._regs[pos]

        def pred(key, default):
            if key not in regs:
                return default
            return max(0.0, float(regs[key].predict(Xm)[0]))

        priors = _POSITION_PRIORS.get(pos, _DEFAULT_PRIOR)
        return {
            "targets_projected":       pred("targets",       priors["targets"]),
            "carries_projected":       pred("carries",       priors["carries"]),
            "pass_attempts_projected": pred("pass_attempts", priors["pass_attempts"]),
            "availability": availability,
        }

    def _predict_stub(self, X: pd.DataFrame, pos: str, availability: float) -> dict:
        priors = _POSITION_PRIORS.get(pos, _DEFAULT_PRIOR)
        team_total = _col(X, "team_implied_total", 23.0)
        volume_scale = team_total / 23.0
        target_share = _col(X, "target_share_l4", None)
        carry_share = _col(X, "carry_share_l4", None)

        targets = (target_share * 32.0 * volume_scale) if target_share else (priors["targets"] * volume_scale)
        carries = (carry_share * 25.0 * volume_scale) if carry_share else (priors["carries"] * volume_scale)
        pass_att = priors.get("pass_attempts", 0.0) * volume_scale

        return {
            "targets_projected":       max(0.0, float(targets)),
            "carries_projected":       max(0.0, float(carries)),
            "pass_attempts_projected": max(0.0, float(pass_att)),
            "availability": availability,
        }


def _col(X: pd.DataFrame, col: str, default):
    if col not in X.columns:
        return default
    val = X[col].iloc[0] if not X.empty else None
    if pd.isna(val):
        return default
    return float(val)
