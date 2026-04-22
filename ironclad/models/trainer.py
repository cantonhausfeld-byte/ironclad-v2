"""Model training orchestrator with walk-forward cross-validation."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error

from ironclad.models.registry import ModelRegistry
from ironclad.models.team.game_outcome import GameOutcomeModel
from ironclad.models.team.score_env import ScoreEnvironmentModel
from ironclad.models.player.usage import PlayerUsageModel
from ironclad.models.player.efficiency import PlayerEfficiencyModel
from ironclad.store.connection import get_connection

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, conn=None) -> None:
        self._conn = conn or get_connection()
        self._registry = ModelRegistry()

    def train_all(self, train_seasons: list[int], val_seasons: list[int] | None = None) -> dict[str, dict]:
        metrics: dict[str, dict] = {}
        metrics["game_outcome"] = self.train_game_outcome(train_seasons, val_seasons)
        metrics["score_env"] = self.train_score_env(train_seasons)
        metrics["player_usage"] = self.train_player_usage(train_seasons)
        metrics["player_efficiency"] = self.train_player_efficiency(train_seasons)
        return metrics

    # ── Game outcome ──────────────────────────────────────────────────────────

    def train_game_outcome(
        self,
        train_seasons: list[int],
        val_seasons: list[int] | None = None,
    ) -> dict[str, Any]:
        logger.info("Training GameOutcomeModel on seasons %s", train_seasons)
        X_train, y_train = self._load_team_data(train_seasons)
        if X_train.empty:
            logger.warning("No team feature data found for %s", train_seasons)
            return {}

        model = GameOutcomeModel()
        model.fit(X_train, y_train)

        metrics: dict[str, Any] = {"train_rows": len(X_train)}

        if val_seasons:
            X_val, y_val = self._load_team_data(val_seasons)
            if not X_val.empty:
                metrics.update(self._eval_game_outcome(model, X_val, y_val))
                model.calibrate(X_val, y_val)
                logger.info("Calibrated on validation seasons %s", val_seasons)

        self._registry.save(model, metrics)
        logger.info("GameOutcomeModel saved. Metrics: %s", metrics)
        return metrics

    def train_score_env(self, train_seasons: list[int]) -> dict[str, Any]:
        logger.info("Training ScoreEnvironmentModel on seasons %s", train_seasons)
        X_train, y_train = self._load_team_data(train_seasons, for_score_env=True)
        if X_train.empty:
            return {}

        model = ScoreEnvironmentModel()
        model.fit(X_train, y_train)

        metrics: dict[str, Any] = {"train_rows": len(X_train)}
        self._registry.save(model, metrics)
        logger.info("ScoreEnvironmentModel saved.")
        return metrics

    # ── Player models ─────────────────────────────────────────────────────────

    def train_player_usage(self, train_seasons: list[int]) -> dict[str, Any]:
        logger.info("Training PlayerUsageModel on seasons %s", train_seasons)
        X, y = self._load_player_data(train_seasons)
        if X.empty:
            return {}

        model = PlayerUsageModel()
        model.fit(X, y)

        metrics: dict[str, Any] = {"train_rows": len(X)}
        if "target_targets" in y.columns:
            mask = y["target_targets"].notna()
            if mask.sum() > 0:
                feat_cols = [c for c in model._regs.get("WR", {}).keys()]
                metrics["mae_targets"] = _safe_mae(y["target_targets"][mask], 4.5)  # baseline
        self._registry.save(model, metrics)
        logger.info("PlayerUsageModel saved.")
        return metrics

    def train_player_efficiency(self, train_seasons: list[int]) -> dict[str, Any]:
        logger.info("Training PlayerEfficiencyModel on seasons %s", train_seasons)
        X, y = self._load_player_data(train_seasons, for_efficiency=True)
        if X.empty:
            return {}

        model = PlayerEfficiencyModel()
        model.fit(X, y)

        metrics: dict[str, Any] = {"train_rows": len(X)}
        self._registry.save(model, metrics)
        logger.info("PlayerEfficiencyModel saved.")
        return metrics

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_team_data(
        self,
        seasons: list[int],
        for_score_env: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        s = ", ".join(str(s) for s in seasons)

        # Load home-side and away-side features, join into one row per game
        home_df = self._conn.execute(f"""
            SELECT * FROM gold.team_game_features
            WHERE season IN ({s}) AND is_home = true
              AND target_points_scored IS NOT NULL
        """).df()

        away_df = self._conn.execute(f"""
            SELECT * FROM gold.team_game_features
            WHERE season IN ({s}) AND is_home = false
              AND target_points_scored IS NOT NULL
        """).df()

        if home_df.empty or away_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Pivot: one row per game with home_ and away_ prefix
        home_df = home_df.add_prefix("home_").rename(columns={"home_game_id": "game_id"})
        away_df = away_df.add_prefix("away_").rename(columns={"away_game_id": "game_id"})

        merged = home_df.merge(away_df, on="game_id", how="inner")

        # Add opponent DEF EPA to home row
        merged["home_opp_def_pass_epa_l4"] = merged.get("away_def_epa_per_play_l4", 0.0)
        merged["away_opp_def_pass_epa_l4"] = merged.get("home_def_epa_per_play_l4", 0.0)

        # Targets from silver.games
        games = self._conn.execute(f"""
            SELECT game_id, home_score, away_score,
                   home_score - away_score AS home_margin,
                   home_score + away_score AS total_score,
                   CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS home_win
            FROM silver.games WHERE season IN ({s})
              AND home_score IS NOT NULL
        """).df()

        X = merged.merge(games, on="game_id", how="inner")

        y_cols = ["home_win", "home_margin", "total_score"]
        if for_score_env:
            y_cols += ["home_target_points_scored", "home_target_yards_total", "home_target_pass_rate"]
            # Flatten score-env targets
            X["target_pass_rate"] = X.get("home_target_pass_rate")
            X["target_points_scored"] = X.get("home_target_points_scored")
            # Estimate plays from PBP (not directly in gold — approximate)
            X["target_plays_total"] = None
            X["target_sack_rate"] = None

        y_df = X[y_cols + [c for c in ["target_pass_rate", "target_points_scored",
                                        "target_plays_total", "target_sack_rate"] if c in X.columns]].copy()
        return X, y_df

    def _load_player_data(
        self,
        seasons: list[int],
        for_efficiency: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        s = ", ".join(str(s) for s in seasons)
        X = self._conn.execute(f"""
            SELECT p.*,
                   t.def_epa_per_play_l4 AS opp_def_pass_epa_l4_from_team,
                   t.def_sack_rate_l4    AS opp_def_sack_rate_l4_from_team
            FROM gold.player_game_features p
            LEFT JOIN gold.team_game_features t
              ON p.game_id = t.game_id AND p.opponent = t.team
            WHERE p.season IN ({s})
              AND p.target_targets IS NOT NULL
              AND p.position IN ('QB','RB','WR','TE','FB')
        """).df()

        if X.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Fill opp defensive features from joined team row
        if "opp_def_pass_epa_l4" not in X.columns or X["opp_def_pass_epa_l4"].isna().all():
            X["opp_def_pass_epa_l4"] = X.get("opp_def_pass_epa_l4_from_team", 0.0)
        if "opp_def_sack_rate_l4" not in X.columns or X["opp_def_sack_rate_l4"].isna().all():
            X["opp_def_sack_rate_l4"] = X.get("opp_def_sack_rate_l4_from_team", 0.06)

        if for_efficiency:
            # Compute per-opportunity rates as targets
            eps = 1e-6
            X["target_catch_rate"] = X["target_receptions"] / (X["target_targets"] + eps)
            X["target_yds_per_target"] = X["target_rec_yards"] / (X["target_targets"] + eps)
            X["target_yds_per_carry"] = X["target_rush_yards"] / (X["target_carries"] + eps)
            X["target_td_rate_tgt"] = (
                X["target_total_tds"] / (X["target_targets"] + X["target_carries"] + eps)
            )
            X["target_td_rate_carry"] = X["target_total_tds"] / (X["target_carries"] + eps)
            # Clip unreasonable rates
            X["target_catch_rate"] = X["target_catch_rate"].clip(0.0, 1.0)
            X["target_yds_per_target"] = X["target_yds_per_target"].clip(0.0, 30.0)
            X["target_yds_per_carry"] = X["target_yds_per_carry"].clip(0.0, 20.0)
            # Only use rows with enough volume to compute stable rates
            X = X[(X["target_targets"] >= 2) | (X["target_carries"] >= 3)]

        y_cols = [c for c in [
            "target_targets", "target_carries", "target_receptions",
            "target_rec_yards", "target_rush_yards", "target_total_tds",
            "target_catch_rate", "target_yds_per_target", "target_yds_per_carry",
            "target_td_rate_tgt", "target_td_rate_carry",
        ] if c in X.columns]

        # Rename for usage model compatibility
        y = X[y_cols].rename(columns={
            "target_targets": "target_targets",
            "target_carries": "target_carries",
        })
        # Add pass_att target for QB rows
        if "pass_attempts" in X.columns:
            y["target_pass_att"] = X.get("pass_attempts")

        return X, y

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _eval_game_outcome(
        self,
        model: GameOutcomeModel,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> dict:
        from ironclad.models.team.game_outcome import _build_diff_features, TEAM_FEATURES
        Xf = _build_diff_features(X_val)
        feat_cols = [c for c in TEAM_FEATURES if c in Xf.columns]
        Xm = Xf[feat_cols].fillna(0)

        if model._clf is None:
            return {}

        probs = model._clf.predict_proba(Xm)[:, 1]
        labels = y_val["home_win"].astype(int).values

        margin_pred = model._reg_margin.predict(Xm)
        total_pred = model._reg_total.predict(Xm)

        return {
            "val_log_loss": round(log_loss(labels, probs), 4),
            "val_brier": round(brier_score_loss(labels, probs), 4),
            "val_margin_mae": round(mean_absolute_error(y_val["home_margin"].values, margin_pred), 2),
            "val_total_mae": round(mean_absolute_error(y_val["total_score"].values, total_pred), 2),
        }

    def evaluate(self, val_seasons: list[int]) -> dict[str, Any]:
        """Evaluate all trained models on held-out seasons."""
        X_val, y_val = self._load_team_data(val_seasons)
        if X_val.empty:
            return {}

        try:
            model = self._registry.load("game_outcome")
            metrics = self._eval_game_outcome(model, X_val, y_val)
            logger.info("Evaluation metrics: %s", metrics)
            return metrics
        except FileNotFoundError:
            logger.warning("No trained game_outcome model found")
            return {}


def _safe_mae(actual: pd.Series, baseline_pred: float) -> float:
    mask = actual.notna()
    if not mask.any():
        return float("nan")
    return float(mean_absolute_error(actual[mask], np.full(mask.sum(), baseline_pred)))
