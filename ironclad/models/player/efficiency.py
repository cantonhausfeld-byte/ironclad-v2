"""Player efficiency model: catch rate, yards/opportunity, TD rate."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ironclad.models.base import BaseModel
from ironclad.models.calibration import IsotonicCalibrator
from ironclad.models.team.game_outcome import _sample_weights

logger = logging.getLogger(__name__)

_EFFICIENCY_PRIORS = {
    "QB":  {"catch_rate": 0.64, "yds_per_tgt": 8.5,  "yds_per_carry": 4.5, "td_rate_tgt": 0.04, "td_rate_carry": 0.025},
    "RB":  {"catch_rate": 0.80, "yds_per_tgt": 7.0,  "yds_per_carry": 4.2, "td_rate_tgt": 0.03, "td_rate_carry": 0.045},
    "WR":  {"catch_rate": 0.64, "yds_per_tgt": 8.5,  "yds_per_carry": 5.0, "td_rate_tgt": 0.06, "td_rate_carry": 0.050},
    "TE":  {"catch_rate": 0.70, "yds_per_tgt": 7.8,  "yds_per_carry": 4.0, "td_rate_tgt": 0.07, "td_rate_carry": 0.040},
    "FB":  {"catch_rate": 0.70, "yds_per_tgt": 6.0,  "yds_per_carry": 3.5, "td_rate_tgt": 0.04, "td_rate_carry": 0.050},
}
_DEFAULT_EFF = {"catch_rate": 0.65, "yds_per_tgt": 8.0, "yds_per_carry": 4.2, "td_rate_tgt": 0.05, "td_rate_carry": 0.04}
_DEF_EPA_SCALE = 0.02
_QB_MIN_ROWS = 50  # minimum QB training rows required


def _to_xgb(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


FEATURES = [
    "catch_rate_l4", "yards_per_target_l4", "yards_per_carry_l4",
    "yac_per_rec_l4", "td_rate_per_target_l4", "td_rate_per_carry_l4",
    "opp_def_pass_epa_l4", "opp_def_rush_epa_l4", "opp_def_sack_rate_l4",
    "is_dome", "temp_f", "wind_mph",
    "team_off_epa_l4",
]

# QB-specific feature set: uses QB passing stats + NGS tracking signals.
# Separate from FEATURES because the standard features include receiving-only
# columns (yac_per_rec, td_rate_per_carry) that are zero/meaningless for passers.
QB_FEATURES = [
    "catch_rate_l4",           # historical completion %
    "yards_per_target_l4",     # historical YPA
    "td_rate_per_target_l4",   # historical TD rate per attempt
    "adot_l4",                 # avg depth of target (deep vs. checkdown tendency)
    "cpoe_l4",                 # completion % above expectation from NGS
    "aggressiveness_l4",       # deep-ball rate from NGS passing
    "opp_def_pass_epa_l4",     # opponent pass defense quality
    "opp_def_sack_rate_l4",    # opponent pressure rate
    "team_off_pass_rate_l4",   # team pass rate environment
    "wind_mph", "temp_f", "is_dome",
]


class PlayerEfficiencyModel(BaseModel):
    """XGBoost player efficiency model (per-position)."""
    name = "player_efficiency"
    version = "v1.0"

    def __init__(self) -> None:
        self._regs: dict[str, dict[str, XGBRegressor]] = {}
        self._td_calibrators: dict[str, IsotonicCalibrator] = {}
        self._qb_regs: dict[str, XGBRegressor] | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        feat_cols = [c for c in FEATURES if c in X.columns]
        positions = X["position"].unique() if "position" in X.columns else ["WR"]
        w_all = _sample_weights(X, season_col="season")

        target_map = {
            "catch_rate":     "target_catch_rate",
            "yds_per_target": "target_yds_per_target",
            "yds_per_carry":  "target_yds_per_carry",
            "td_rate_tgt":    "target_td_rate_tgt",
            "td_rate_carry":  "target_td_rate_carry",
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
                mask_valid = yp.notna() & (yp >= 0)
                if mask_valid.sum() < 20:
                    continue
                w_valid = w_pos[mask_valid] if w_pos is not None else None
                reg = XGBRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                )
                reg.fit(Xp[mask_valid], yp[mask_valid], sample_weight=w_valid)
                self._regs[pos][out_key] = reg

        # QB sub-model: train on QB-specific features so QBs use real data,
        # not the receiving-based regressors that produce meaningless outputs.
        if "position" in X.columns:
            qb_mask = X["position"] == "QB"
            if qb_mask.sum() >= _QB_MIN_ROWS:
                w_qb = w_all[qb_mask] if w_all is not None else None
                self._fit_qb_regressors(X[qb_mask], y[qb_mask], w_qb)

        self._fitted = True
        logger.info("PlayerEfficiencyModel fitted for positions: %s", list(self._regs.keys()))

    def predict(self, X: pd.DataFrame) -> dict:
        pos = str(X["position"].iloc[0]) if "position" in X.columns else "WR"
        qb_regs = getattr(self, "_qb_regs", None)
        if pos == "QB" and qb_regs:
            return self._predict_qb(X)
        if self._fitted and pos in self._regs and self._regs[pos]:
            return self._predict_trained(X, pos)
        return self._predict_stub(X, pos)

    def _fit_qb_regressors(
        self, X_qb: pd.DataFrame, y_qb: pd.DataFrame, weights: np.ndarray | None
    ) -> None:
        qb_feat_cols = [c for c in QB_FEATURES if c in X_qb.columns]
        Xm = _to_xgb(X_qb[qb_feat_cols])
        self._qb_regs = {}
        qb_target_map = {
            "catch_rate":        "target_catch_rate",
            "yards_per_target":  "target_yds_per_target",
            "td_rate_per_target": "target_td_rate_tgt",
        }
        for out_key, tgt_col in qb_target_map.items():
            if tgt_col not in y_qb.columns:
                continue
            yq = y_qb[tgt_col]
            mask_valid = yq.notna() & (yq >= 0)
            if mask_valid.sum() < 20:
                continue
            w_valid = weights[mask_valid] if weights is not None else None
            reg = XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            )
            reg.fit(Xm[mask_valid], yq[mask_valid], sample_weight=w_valid)
            self._qb_regs[out_key] = reg
        logger.info(
            "QB efficiency sub-model fitted (%d rows, regressors: %s)",
            len(X_qb), list(self._qb_regs.keys()),
        )

    def _predict_qb(self, X: pd.DataFrame) -> dict:
        priors = _EFFICIENCY_PRIORS["QB"]
        qb_regs = getattr(self, "_qb_regs", None) or {}

        def pred(key, lo, hi, default):
            if key not in qb_regs:
                return default
            reg = qb_regs[key]
            train_cols = list(reg.feature_names_in_)
            Xm = _to_xgb(X.reindex(columns=train_cols, fill_value=0.0))
            val = float(reg.predict(Xm)[0])
            return max(lo, min(hi, val))

        completion_pct = pred("catch_rate",        0.50, 0.75, priors["catch_rate"])
        ypa            = pred("yards_per_target",   4.0, 13.0, priors["yds_per_tgt"])
        td_rate        = pred("td_rate_per_target", 0.0,  0.15, priors["td_rate_tgt"])

        return {
            "catch_rate":           completion_pct,
            "yards_per_target":     ypa,
            "yards_per_carry":      priors["yds_per_carry"],
            "td_rate_per_target":   td_rate,
            "td_rate_per_carry":    priors["td_rate_carry"],
            "yards_per_target_std": max(0.5, ypa * 0.7),
            "yards_per_carry_std":  priors["yds_per_carry"] * 2.1,
        }

    def _predict_trained(self, X: pd.DataFrame, pos: str) -> dict:
        regs = self._regs[pos]
        priors = _EFFICIENCY_PRIORS.get(pos, _DEFAULT_EFF)

        def pred(key, default, lo=0.0, hi=None):
            if key not in regs:
                return default
            train_cols = list(regs[key].feature_names_in_)
            Xm = _to_xgb(X.reindex(columns=train_cols, fill_value=0.0))
            val = float(regs[key].predict(Xm)[0])
            val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
            return val

        catch_rate = pred("catch_rate", priors["catch_rate"], 0.1, 0.99)
        yds_per_tgt = pred("yds_per_target", priors["yds_per_tgt"], 1.0)
        yds_per_carry = pred("yds_per_carry", priors["yds_per_carry"], 0.5)
        td_rate_tgt = pred("td_rate_tgt", priors["td_rate_tgt"], 0.0, 0.5)
        td_rate_carry = pred("td_rate_carry", priors["td_rate_carry"], 0.0, 0.5)

        return {
            "catch_rate": catch_rate,
            "yards_per_target": yds_per_tgt,
            "yards_per_carry": yds_per_carry,
            "td_rate_per_target": td_rate_tgt,
            "td_rate_per_carry": td_rate_carry,
            "yards_per_target_std": max(0.5, yds_per_tgt * 0.7),
            "yards_per_carry_std": max(0.5, yds_per_carry * 2.1),
        }

    def _predict_stub(self, X: pd.DataFrame, pos: str) -> dict:
        priors = _EFFICIENCY_PRIORS.get(pos, _DEFAULT_EFF)

        def col(c):
            if c not in X.columns:
                return None
            val = X[c].iloc[0] if not X.empty else None
            return float(val) if pd.notna(val) else None

        opp_def_epa = col("opp_def_pass_epa_l4") or 0.0
        eff_adj = max(0.6, min(1.4, 1.0 - opp_def_epa * _DEF_EPA_SCALE))

        catch_rate = col("catch_rate_l4") or priors["catch_rate"]
        yds_per_tgt = col("yards_per_target_l4") or priors["yds_per_tgt"]
        yds_per_carry = col("yards_per_carry_l4") or priors["yds_per_carry"]
        # QBs: td_rate_per_target_l4 is a receiving stat (meaningless for passers);
        # always use the QB prior so small-sample outliers don't skew TD distribution.
        if pos == "QB":
            td_rate_tgt = priors["td_rate_tgt"]
        else:
            td_rate_tgt = col("td_rate_per_target_l4") or priors["td_rate_tgt"]
        td_rate_carry = min(col("td_rate_per_carry_l4") or priors["td_rate_carry"], 0.10)

        if catch_rate is not None:
            catch_rate = max(0.1, min(1.0, catch_rate * eff_adj))
        if yds_per_tgt is not None:
            yds_per_tgt = max(2.0, yds_per_tgt * eff_adj)
        if yds_per_carry is not None:
            yds_per_carry = max(1.0, yds_per_carry)

        return {
            "catch_rate": catch_rate,
            "yards_per_target": yds_per_tgt,
            "yards_per_carry": yds_per_carry,
            "td_rate_per_target": td_rate_tgt,
            "td_rate_per_carry": td_rate_carry,
            "yards_per_target_std": (yds_per_tgt or 8.0) * 0.7,
            "yards_per_carry_std": (yds_per_carry or 4.2) * 2.1,
        }
