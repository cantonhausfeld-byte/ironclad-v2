"""Player usage model: projected targets, carries, snap share."""
from __future__ import annotations

import logging

import pandas as pd
from xgboost import XGBRegressor

from ironclad.models.base import BaseModel
from ironclad.models.team.game_outcome import _sample_weights

logger = logging.getLogger(__name__)

_POSITION_PRIORS = {
    "QB":  {"targets": 0.0,  "carries": 6.0,  "pass_attempts": 32.0},
    "RB":  {"targets": 3.5,  "carries": 18.0, "pass_attempts": 0.0},
    "WR":  {"targets": 5.5,  "carries": 0.2,  "pass_attempts": 0.0},
    "TE":  {"targets": 4.0,  "carries": 0.0,  "pass_attempts": 0.0},
    "FB":  {"targets": 1.5,  "carries": 2.0,  "pass_attempts": 0.0},
}
_DEFAULT_PRIOR = {"targets": 2.0, "carries": 1.0, "pass_attempts": 0.0}


def _to_xgb(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to float for XGBoost (handles nullable int / object dtype)."""
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


FEATURES = [
    "availability", "depth_team",
    "target_share_l4", "carry_share_l4", "air_yards_share_l4",
    # route_rate_l4 omitted: FTN charting data not consistently available (always NULL)
    "redzone_target_share_l4", "redzone_carry_share_l4",
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
        w_all = _sample_weights(X, season_col="season")

        target_map = {
            "targets":        "target_targets",
            "carries":        "target_carries",
            "pass_attempts":  "target_pass_att",
        }

        for pos in positions:
            mask_pos = X["position"] == pos
            if mask_pos.sum() < 20:
                continue
            Xp = _to_xgb(X[mask_pos][feat_cols])
            w_pos = w_all[mask_pos] if w_all is not None else None
            self._regs[pos] = {}
            for out_key, tgt_col in target_map.items():
                if tgt_col not in y.columns:
                    continue
                yp = y[tgt_col][mask_pos]
                mask_valid = yp.notna()
                if mask_valid.sum() < 20:
                    continue
                # Poisson objective for count targets (targets, carries): enforces
                # non-negative predictions and models the log of expected count.
                obj = "count:poisson" if out_key in ("targets", "carries") else "reg:squarederror"
                w_valid = w_pos[mask_valid] if w_pos is not None else None
                reg = XGBRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                    objective=obj,
                )
                reg.fit(Xp[mask_valid], yp[mask_valid].clip(lower=0), sample_weight=w_valid)
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
        regs = self._regs[pos]

        def pred(key, default):
            if key not in regs:
                return default
            train_cols = list(regs[key].feature_names_in_)
            Xm = _to_xgb(X.reindex(columns=train_cols, fill_value=0.0))
            return max(0.0, float(regs[key].predict(Xm)[0]))

        priors = _POSITION_PRIORS.get(pos, _DEFAULT_PRIOR)
        targets  = pred("targets",       priors["targets"])
        carries  = pred("carries",       priors["carries"])
        pass_att = pred("pass_attempts", priors["pass_attempts"])

        # Zero out fringe roster players: no historical share AND no known depth slot.
        # Without this guard the model's learned intercept gives non-zero volume to
        # every supplemental roster player, diluting starters via the reconciler.
        carry_share  = _col(X, "carry_share_l4",  None)
        target_share = _col(X, "target_share_l4", None)
        depth = _col(X, "depth_team", None)
        has_history   = (carry_share  is not None and carry_share  > 0.005) or \
                        (target_share is not None and target_share > 0.005)
        starter_proxy = (depth is not None and depth <= 2) or _infer_starter(
            depth, carry_share, target_share
        )
        if not has_history and not starter_proxy:
            carries = 0.0
            targets = 0.0

        return {
            "targets_projected":       targets,
            "carries_projected":       carries,
            "pass_attempts_projected": pass_att,
            "availability": availability,
        }

    def _predict_stub(self, X: pd.DataFrame, pos: str, availability: float) -> dict:
        priors = _POSITION_PRIORS.get(pos, _DEFAULT_PRIOR)
        # team_implied_total stores the full game over/under (~46 avg), not the
        # per-team total. Normalise against 46 so average games yield scale ≈ 1.0.
        team_total = _col(X, "team_implied_total", 46.0)
        volume_scale = team_total / 46.0
        target_share = _col(X, "target_share_l4", None)
        carry_share = _col(X, "carry_share_l4", None)

        # Only grant position-prior volume to players with a known starter slot.
        # When depth chart data is unavailable (depth_team=None), infer starter
        # status from historical share — covers seasons where nfl_data_py lags.
        depth = _col(X, "depth_team", None)
        starter_proxy = (depth is not None and depth <= 2) or _infer_starter(
            depth, carry_share, target_share
        )

        targets = (target_share * 32.0 * volume_scale) if target_share else (priors["targets"] * volume_scale if starter_proxy else 0.0)
        carries = (carry_share * 27.6 * volume_scale) if carry_share else (priors["carries"] * volume_scale if starter_proxy else 0.0)
        pass_att = priors.get("pass_attempts", 0.0) * volume_scale

        return {
            "targets_projected":       max(0.0, float(targets)),
            "carries_projected":       max(0.0, float(carries)),
            "pass_attempts_projected": max(0.0, float(pass_att)),
            "availability": availability,
        }


def _infer_starter(
    depth: float | None,
    carry_share: float | None,
    target_share: float | None,
) -> bool:
    """Infer starter status from historical share when depth_team is unavailable.

    When nfl_data_py depth chart data lags (e.g. 2025 early season), treat
    any player with genuine starter-level historical usage as depth_team ≤ 2.
    Thresholds: WR/TE starter ≈ 10%+ target share; RB starter ≈ 15%+ carry share.
    """
    if depth is not None:
        return False  # Real depth data takes precedence — no inference needed
    return (target_share is not None and target_share > 0.08) or \
           (carry_share  is not None and carry_share  > 0.12)


def _col(X: pd.DataFrame, col: str, default):
    if col not in X.columns:
        return default
    val = X[col].iloc[0] if not X.empty else None
    if pd.isna(val):
        return default
    return float(val)
