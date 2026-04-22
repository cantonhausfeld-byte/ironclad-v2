"""Player usage model: projected targets, carries, snap share."""
from __future__ import annotations

import pandas as pd

from ironclad.models.base import BaseModel

# Position-level priors (league average volume per game)
_POSITION_PRIORS = {
    "QB":  {"targets": 0.0,  "carries": 3.5,  "pass_attempts": 32.0},
    "RB":  {"targets": 3.5,  "carries": 12.0, "pass_attempts": 0.0},
    "WR":  {"targets": 5.5,  "carries": 0.2,  "pass_attempts": 0.0},
    "TE":  {"targets": 4.0,  "carries": 0.0,  "pass_attempts": 0.0},
    "FB":  {"targets": 1.5,  "carries": 2.0,  "pass_attempts": 0.0},
}
_DEFAULT_PRIOR = {"targets": 2.0, "carries": 1.0, "pass_attempts": 0.0}


class PlayerUsageModel(BaseModel):
    """Stub: scales position priors by team pass rate and implied total."""
    name = "player_usage"
    version = "stub_v1"

    def predict(self, X: pd.DataFrame) -> dict:
        pos = str(X["position"].iloc[0]) if "position" in X.columns else "WR"
        priors = _POSITION_PRIORS.get(pos, _DEFAULT_PRIOR)
        availability = _col(X, "availability", 1.0)

        # Scale by team implied total (more points = more volume)
        team_total = _col(X, "team_implied_total", 23.0)
        volume_scale = team_total / 23.0

        # Override with rolling L4 shares when available
        target_share = _col(X, "target_share_l4", None)
        carry_share = _col(X, "carry_share_l4", None)
        team_pass_att = 32.0  # baseline

        if target_share is not None and target_share > 0:
            targets = target_share * team_pass_att * volume_scale
        else:
            targets = priors["targets"] * volume_scale

        if carry_share is not None and carry_share > 0:
            carries = carry_share * 25.0 * volume_scale  # 25 rush att baseline
        else:
            carries = priors["carries"] * volume_scale

        pass_attempts = priors.get("pass_attempts", 0.0) * volume_scale

        # Availability gate: zero out if player is out
        if availability == 0.0:
            targets = carries = pass_attempts = 0.0

        return {
            "targets_projected": max(0.0, float(targets)),
            "carries_projected": max(0.0, float(carries)),
            "pass_attempts_projected": max(0.0, float(pass_attempts)),
            "availability": float(availability),
        }


def _col(X: pd.DataFrame, col: str, default):
    if col not in X.columns:
        return default
    val = X[col].iloc[0] if not X.empty else None
    if pd.isna(val):
        return default
    return float(val)
