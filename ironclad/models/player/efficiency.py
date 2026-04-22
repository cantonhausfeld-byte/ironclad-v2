"""Player efficiency model: catch rate, yards per opportunity, TD rate."""
from __future__ import annotations

import pandas as pd

from ironclad.models.base import BaseModel

# Position-level efficiency priors
_EFFICIENCY_PRIORS = {
    "QB":  {"catch_rate": None, "yds_per_tgt": None, "yds_per_carry": 4.5, "td_rate_tgt": None, "td_rate_carry": 0.025},
    "RB":  {"catch_rate": 0.80, "yds_per_tgt": 7.0,  "yds_per_carry": 4.2, "td_rate_tgt": 0.03, "td_rate_carry": 0.045},
    "WR":  {"catch_rate": 0.64, "yds_per_tgt": 8.5,  "yds_per_carry": 5.0, "td_rate_tgt": 0.06, "td_rate_carry": 0.05},
    "TE":  {"catch_rate": 0.70, "yds_per_tgt": 7.8,  "yds_per_carry": 4.0, "td_rate_tgt": 0.07, "td_rate_carry": 0.04},
    "FB":  {"catch_rate": 0.70, "yds_per_tgt": 6.0,  "yds_per_carry": 3.5, "td_rate_tgt": 0.04, "td_rate_carry": 0.05},
}
_DEFAULT_EFF = {"catch_rate": 0.65, "yds_per_tgt": 8.0, "yds_per_carry": 4.2, "td_rate_tgt": 0.05, "td_rate_carry": 0.04}

# Defensive EPA adjusts efficiency: each 0.1 EPA difference from 0 shifts by ~2%
_DEF_EPA_SCALE = 0.02


class PlayerEfficiencyModel(BaseModel):
    """Stub: uses rolling L4 efficiency with opponent-defense adjustment."""
    name = "player_efficiency"
    version = "stub_v1"

    def predict(self, X: pd.DataFrame) -> dict:
        pos = str(X["position"].iloc[0]) if "position" in X.columns else "WR"
        priors = _EFFICIENCY_PRIORS.get(pos, _DEFAULT_EFF)

        catch_rate = _col(X, "catch_rate_l4") or priors["catch_rate"]
        yds_per_tgt = _col(X, "yards_per_target_l4") or priors["yds_per_tgt"]
        yds_per_carry = _col(X, "yards_per_carry_l4") or priors["yds_per_carry"]
        td_rate_tgt = _col(X, "td_rate_per_target_l4") or priors["td_rate_tgt"]
        td_rate_carry = _col(X, "td_rate_per_carry_l4") or priors["td_rate_carry"]

        # Adjust for opponent defensive quality
        opp_def_epa = _col(X, "opp_def_pass_epa_l4") or 0.0
        eff_adj = 1.0 - (opp_def_epa * _DEF_EPA_SCALE)
        eff_adj = max(0.6, min(1.4, eff_adj))

        if catch_rate is not None:
            catch_rate = max(0.1, min(1.0, catch_rate * eff_adj))
        if yds_per_tgt is not None:
            yds_per_tgt = max(2.0, yds_per_tgt * eff_adj)
        if yds_per_carry is not None:
            opp_rush_epa = _col(X, "opp_def_rush_epa_l4") or 0.0
            rush_adj = 1.0 - (opp_rush_epa * _DEF_EPA_SCALE)
            rush_adj = max(0.6, min(1.4, rush_adj))
            yds_per_carry = max(1.0, yds_per_carry * rush_adj)

        return {
            "catch_rate": catch_rate,
            "yards_per_target": yds_per_tgt,
            "yards_per_carry": yds_per_carry,
            "td_rate_per_target": td_rate_tgt,
            "td_rate_per_carry": td_rate_carry,
            "yards_per_target_std": (yds_per_tgt or 8.0) * 0.7,
            "yards_per_carry_std": (yds_per_carry or 4.2) * 0.9,
        }


def _col(X: pd.DataFrame, col: str):
    if col not in X.columns:
        return None
    val = X[col].iloc[0] if not X.empty else None
    return float(val) if pd.notna(val) else None
