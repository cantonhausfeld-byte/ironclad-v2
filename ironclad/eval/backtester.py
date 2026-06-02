"""Walk-forward backtest engine for GameOutcomeModel."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ironclad.models.team.game_outcome import (
    CLF_FEATURES,
    TEAM_FEATURES,
    GameOutcomeModel,
    _build_diff_features,
)
from ironclad.models.trainer import ModelTrainer
from ironclad.store.connection import get_connection

logger = logging.getLogger(__name__)


class Backtester:
    """
    Expanding-window walk-forward backtester.

    For each test season T in test_seasons, trains GameOutcomeModel on
    [train_start .. T-1], predicts all completed games in season T, and
    persists results to gold.backtest_predictions.
    """

    def __init__(self, conn=None, model_class=None) -> None:
        self._conn = conn or get_connection()
        self._model_class = model_class if model_class is not None else GameOutcomeModel

    def run(
        self,
        train_start: int,
        test_seasons: list[int],
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Execute the walk-forward backtest.

        Returns a DataFrame of all predictions (same schema as
        gold.backtest_predictions).
        """
        from ironclad.store.schema import create_all_tables
        create_all_tables(self._conn)
        self._conn.execute("CHECKPOINT")

        if run_id is None:
            run_id = f"bt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        logger.info("Backtest run %s | train_start=%d | test=%s", run_id, train_start, test_seasons)

        all_rows: list[dict] = []

        for T in sorted(test_seasons):
            train_seasons = list(range(train_start, T))
            if not train_seasons:
                logger.warning("No training seasons before %d — skipping fold", T)
                continue

            rows = self._run_fold(run_id, train_seasons, T)
            if rows:
                all_rows.extend(rows)
                logger.info("Fold %d: %d games predicted", T, len(rows))
            else:
                logger.warning("Fold %d produced no predictions", T)

        if not all_rows:
            logger.warning("Backtest produced no results")
            return pd.DataFrame()

        preds_df = pd.DataFrame(all_rows)
        self._save(preds_df, run_id)
        return preds_df

    def load_latest(self) -> pd.DataFrame:
        """Return predictions from the most recent completed backtest run."""
        try:
            row = self._conn.execute("""
                SELECT backtest_run_id FROM gold.backtest_predictions
                ORDER BY predicted_at DESC LIMIT 1
            """).fetchone()
            if not row:
                return pd.DataFrame()
            return self._conn.execute("""
                SELECT * FROM gold.backtest_predictions
                WHERE backtest_run_id = ?
                ORDER BY season, week
            """, [row[0]]).df()
        except Exception as e:
            logger.warning("Could not load backtest predictions: %s", e)
            return pd.DataFrame()

    def load_run(self, run_id: str) -> pd.DataFrame:
        """Return predictions for a specific run_id."""
        try:
            return self._conn.execute("""
                SELECT * FROM gold.backtest_predictions
                WHERE backtest_run_id = ?
                ORDER BY season, week
            """, [run_id]).df()
        except Exception as e:
            logger.warning("Could not load run %s: %s", run_id, e)
            return pd.DataFrame()

    def list_runs(self) -> pd.DataFrame:
        """Return a summary of all backtest runs."""
        try:
            return self._conn.execute("""
                SELECT
                    backtest_run_id,
                    MIN(season)   AS first_season,
                    MAX(season)   AS last_season,
                    COUNT(*)      AS n_games,
                    MAX(predicted_at) AS run_at
                FROM gold.backtest_predictions
                GROUP BY backtest_run_id
                ORDER BY run_at DESC
            """).df()
        except Exception as e:
            logger.warning("Could not list runs: %s", e)
            return pd.DataFrame()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_fold(
        self,
        run_id: str,
        train_seasons: list[int],
        test_season: int,
    ) -> list[dict]:
        trainer = ModelTrainer(conn=self._conn)

        # ── Training ──
        X_train, y_train = trainer._load_team_data(train_seasons)
        if X_train.empty:
            logger.warning("No training data for %s", train_seasons)
            return []

        model = self._model_class()
        model.fit(X_train, y_train)

        # ── Calibration ──
        # Use the full last training season (~267 games) for Platt scaling.
        # Platt scaling (logistic regression on logits) is robust with ~267 samples;
        # isotonic overfits on small calibration sets and produces step-function behavior.
        last_season = max(train_seasons)
        X_cal, y_cal = trainer._load_team_data([last_season])
        if not X_cal.empty and len(X_cal) >= 30:
            Xf_cal = _build_diff_features(X_cal)
            feat_cols_cal = [c for c in CLF_FEATURES if c in Xf_cal.columns]
            if feat_cols_cal and model._clf is not None:
                Xm_cal = Xf_cal[feat_cols_cal].fillna(0.0)
                raw_cal_probs = model._clf.predict_proba(Xm_cal)[:, 1]
                y_cal_arr = y_cal["home_win"].astype(int).to_numpy()
                try:
                    model.calibrate_from_probs(raw_cal_probs, y_cal_arr)
                    logger.info("Fold %d: Platt calibration fitted on %d games", test_season, len(raw_cal_probs))
                except Exception as e:
                    logger.warning("Platt calibration failed for fold %d: %s", test_season, e)

        # ── Test data ──
        X_test, y_test = trainer._load_team_data([test_season])
        if X_test.empty:
            return []

        # ── Batch prediction ──
        Xf = _build_diff_features(X_test)
        clf_cols = [c for c in CLF_FEATURES if c in Xf.columns]
        reg_cols = [c for c in TEAM_FEATURES if c in Xf.columns]
        if not clf_cols:
            logger.warning("No CLF_FEATURES found in fold %d", test_season)
            return []
        Xm_clf = Xf[clf_cols].fillna(0.0)
        Xm_reg = Xf[reg_cols].fillna(0.0)

        if model._clf is None:
            return []

        raw_probs = model._clf.predict_proba(Xm_clf)[:, 1]
        platt_probs = np.clip(model._calibrator.transform(raw_probs), 0.02, 0.98)
        margin_preds = model._reg_margin.predict(Xm_reg)
        total_preds = model._reg_total.predict(Xm_reg)

        now_ts = datetime.now(timezone.utc).isoformat()
        rows: list[dict] = []

        for i in range(len(X_test)):
            xrow = X_test.iloc[i]
            yrow = y_test.iloc[i]

            rows.append({
                "backtest_run_id":    run_id,
                "game_id":            str(xrow["game_id"]),
                "season":             test_season,
                "week":               int(xrow.get("home_week", 0)),
                "home_team":          str(xrow.get("home_team", "")),
                "away_team":          str(xrow.get("away_team", "")),
                "model_name":         model.name,
                "model_version":      f"fold_{test_season}",
                "predicted_at":       now_ts,
                "home_win_prob":      round(float(platt_probs[i]), 4),
                "home_win_prob_vegas": _fval(xrow, "home_home_win_prob_from_odds"),
                "home_margin_pred":   round(float(margin_preds[i]), 2),
                "total_pred":         round(float(total_preds[i]), 2),
                "home_win_actual":    bool(yrow["home_win"]) if "home_win" in yrow.index else None,
                "home_margin_actual": _ival(yrow, "home_margin"),
                "total_actual":       _ival(yrow, "total_score"),
                "vegas_spread":       _fval(xrow, "home_spread_from_odds"),
                "vegas_total":        _fval(xrow, "home_implied_total_from_odds"),
            })

        return rows

    def _save(self, df: pd.DataFrame, run_id: str) -> None:
        try:
            self._conn.execute(
                "DELETE FROM gold.backtest_predictions WHERE backtest_run_id = ?",
                [run_id],
            )
            self._conn.register("_bt_tmp", df)
            self._conn.execute("""
                INSERT INTO gold.backtest_predictions (
                    backtest_run_id, game_id, season, week, home_team, away_team,
                    model_name, model_version, predicted_at,
                    home_win_prob, home_win_prob_vegas,
                    home_margin_pred, total_pred,
                    home_win_actual, home_margin_actual, total_actual,
                    vegas_spread, vegas_total
                )
                SELECT
                    backtest_run_id, game_id, season, week, home_team, away_team,
                    model_name, model_version, predicted_at::TIMESTAMPTZ,
                    home_win_prob, home_win_prob_vegas,
                    home_margin_pred, total_pred,
                    home_win_actual, home_margin_actual, total_actual,
                    vegas_spread, vegas_total
                FROM _bt_tmp
            """)
            logger.info("Saved %d backtest rows for run %s", len(df), run_id)
        except Exception as e:
            logger.error("Failed to save backtest predictions: %s", e)
        finally:
            try:
                self._conn.unregister("_bt_tmp")
            except Exception:
                pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fval(row, col: str) -> float | None:
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ival(row, col: str) -> int | None:
    v = _fval(row, col)
    return int(round(v)) if v is not None else None
