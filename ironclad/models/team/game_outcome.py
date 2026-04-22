"""Game outcome model: win probability, spread, total.

Phase 1 uses a calibrated stub based on odds-implied priors and
historical distributions. XGBoost training is added in Phase 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ironclad.config import (
    LEAGUE_HOME_WIN_PROB,
    LEAGUE_AVG_TOTAL,
    LEAGUE_AVG_HOME_MARGIN,
)
from ironclad.models.base import BaseModel


# Historical std devs derived from 2016–2023 NFL data
_MARGIN_STD = 14.1
_TOTAL_STD = 10.2


class GameOutcomeModel(BaseModel):
    """Stub implementation using odds-implied priors."""
    name = "game_outcome"
    version = "stub_v1"

    def predict(self, X: pd.DataFrame) -> dict:
        """Predict game outcome parameters.

        X must have columns: home_win_prob_from_odds, implied_total_from_odds,
        spread_from_odds (home perspective). Falls back to league averages when missing.
        """
        home_win_prob = _col(X, "home_win_prob_from_odds", LEAGUE_HOME_WIN_PROB)
        total_mean = _col(X, "implied_total_from_odds", LEAGUE_AVG_TOTAL)
        spread_mean = _col(X, "spread_from_odds", -LEAGUE_AVG_HOME_MARGIN)

        # Derive margin from spread (home_spread = home_score - away_score)
        home_margin_mean = -float(spread_mean)  # spread is from home perspective

        return {
            "home_win_prob": float(home_win_prob),
            "away_win_prob": 1.0 - float(home_win_prob),
            "home_margin_mean": home_margin_mean,
            "home_margin_std": _MARGIN_STD,
            "total_mean": float(total_mean),
            "total_std": _TOTAL_STD,
        }


def _col(X: pd.DataFrame, col: str, default: float) -> float:
    if col in X.columns and not X[col].isna().all():
        val = X[col].iloc[0]
        if pd.notna(val):
            return float(val)
    return default
