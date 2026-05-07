"""LightGBM game outcome model with Optuna hyperparameter tuning."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

import optuna

from ironclad.config import LEAGUE_HOME_WIN_PROB, LEAGUE_AVG_TOTAL, LEAGUE_AVG_HOME_MARGIN
from ironclad.models.base import BaseModel
from ironclad.models.calibration import PlattCalibrator
from ironclad.models.team.game_outcome import (
    CLF_FEATURES, TEAM_FEATURES, _build_diff_features, _col,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

_MARGIN_STD = 14.1
_TOTAL_STD = 10.2
_N_TRIALS = 50
_N_CV_SPLITS = 3


def _lgbm_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "num_leaves":         trial.suggest_int("num_leaves", 20, 200),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples":  trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators":       trial.suggest_int("n_estimators", 100, 500),
    }
    tscv = TimeSeriesSplit(n_splits=_N_CV_SPLITS)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        clf = LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_val)[:, 1]
        scores.append(log_loss(y_val, probs))
    return float(np.mean(scores))


class GameOutcomeLGBM(BaseModel):
    """LightGBM game outcome model with Optuna-tuned hyperparameters."""
    name = "game_outcome_lgbm"
    version = "v1.0"

    def __init__(self) -> None:
        self._clf: LGBMClassifier | None = None
        self._reg_margin: LGBMRegressor | None = None
        self._reg_total: LGBMRegressor | None = None
        self._calibrator = PlattCalibrator()
        self._margin_std = _MARGIN_STD
        self._total_std = _TOTAL_STD
        self._best_params: dict = {}
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

        # Tune on classifier (log-loss objective) with time-series CV.
        # Never shuffle — NFL seasons are a time series; shuffled CV leaks future data.
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            lambda trial: _lgbm_objective(trial, Xm_clf, y_win),
            n_trials=_N_TRIALS,
            show_progress_bar=False,
        )
        self._best_params = study.best_params
        logger.info(
            "Optuna: best log-loss=%.4f  params=%s",
            study.best_value, self._best_params,
        )

        base_kw = {**self._best_params, "random_state": 42, "n_jobs": -1, "verbose": -1}

        self._clf = LGBMClassifier(**base_kw)
        self._clf.fit(Xm_clf, y_win)

        self._reg_margin = LGBMRegressor(**base_kw)
        self._reg_margin.fit(Xm_reg, y_margin)

        self._reg_total = LGBMRegressor(**base_kw)
        self._reg_total.fit(Xm_reg, y_total)

        margin_pred = self._reg_margin.predict(Xm_reg)
        total_pred = self._reg_total.predict(Xm_reg)
        self._margin_std = float(np.std(y_margin.values - margin_pred))
        self._total_std = float(np.std(y_total.values - total_pred))

        self._fitted = True
        logger.info(
            "GameOutcomeLGBM fitted on %d rows (clf=%d feats, reg=%d feats)",
            len(Xm_clf), len(clf_cols), len(reg_cols),
        )

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

        vegas_win_prob = _col(X, "home_win_prob_from_odds", None)
        if vegas_win_prob is not None:
            home_win_prob = float(np.clip(0.5 * platt_prob + 0.5 * float(vegas_win_prob), 0.02, 0.98))
        else:
            home_win_prob = platt_prob

        home_margin_mean = float(self._reg_margin.predict(Xm_reg)[0])
        total_mean_model = float(self._reg_total.predict(Xm_reg)[0])
        vegas_total = _col(X, "implied_total_from_odds", total_mean_model)
        total_mean = 0.4 * total_mean_model + 0.6 * vegas_total

        return {
            "home_win_prob":      home_win_prob,
            "away_win_prob":      1.0 - home_win_prob,
            "home_margin_mean":   home_margin_mean,
            "home_margin_std":    self._margin_std,
            "total_mean":         max(20.0, total_mean),
            "total_std":          self._total_std,
        }

    def _predict_stub(self, X: pd.DataFrame) -> dict:
        home_win_prob = _col(X, "home_win_prob_from_odds", LEAGUE_HOME_WIN_PROB)
        total_mean = _col(X, "implied_total_from_odds", LEAGUE_AVG_TOTAL)
        spread = _col(X, "spread_from_odds", -LEAGUE_AVG_HOME_MARGIN)
        return {
            "home_win_prob":    float(home_win_prob),
            "away_win_prob":    1.0 - float(home_win_prob),
            "home_margin_mean": -float(spread),
            "home_margin_std":  _MARGIN_STD,
            "total_mean":       float(total_mean),
            "total_std":        _TOTAL_STD,
        }
