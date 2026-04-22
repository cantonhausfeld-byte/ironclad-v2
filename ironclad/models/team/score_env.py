"""Score environment model: pace, pass rate, scoring context."""
from __future__ import annotations

import pandas as pd

from ironclad.config import LEAGUE_PRIORS
from ironclad.models.base import BaseModel


class ScoreEnvironmentModel(BaseModel):
    """Stub: returns rolling averages for team game shape."""
    name = "score_env"
    version = "stub_v1"

    def predict(self, X: pd.DataFrame) -> dict:
        pass_rate = _col(X, "off_pass_rate_l4", LEAGUE_PRIORS["pass_rate"])
        total_plays = _col(X, None, LEAGUE_PRIORS["total_plays"])
        sack_rate = _col(X, "def_sack_rate_l4", LEAGUE_PRIORS["sack_rate"])
        team_total = _col(X, "implied_total_from_odds", LEAGUE_PRIORS["points_per_game"])
        if team_total is None:
            team_total = LEAGUE_PRIORS["points_per_game"]
        # Team implied total is half of game total (rough split)
        team_total = float(team_total) / 2.0

        return {
            "pass_rate_projected": float(pass_rate),
            "total_plays_projected": float(total_plays),
            "sack_rate_projected": float(sack_rate),
            "team_total_projected": float(team_total),
        }


def _col(X: pd.DataFrame | None, col: str | None, default: float) -> float:
    if X is None or col is None or col not in X.columns:
        return default
    val = X[col].iloc[0] if not X.empty else None
    return float(val) if pd.notna(val) else default
