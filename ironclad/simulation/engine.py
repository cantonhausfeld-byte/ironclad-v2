"""Monte Carlo simulation engine: orchestrates N game draws."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ironclad.config import DEFAULT_N_DRAWS
from ironclad.simulation.game_draw import GameDraw
from ironclad.simulation.player_draw import PlayerDraw, PlayerContext, PlayerDrawResult
from ironclad.simulation.reconciler import Reconciler
from ironclad.simulation.results import SimulationResult, DrawRecord
from ironclad.models.team.game_outcome import GameOutcomeModel
from ironclad.models.team.score_env import ScoreEnvironmentModel
from ironclad.models.player.usage import PlayerUsageModel
from ironclad.models.player.efficiency import PlayerEfficiencyModel
from ironclad.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    def __init__(self, n_draws: int = DEFAULT_N_DRAWS, seed: int = 42) -> None:
        self.n_draws = n_draws
        self.seed = seed
        self._game_draw = GameDraw()
        self._player_draw = PlayerDraw()
        self._reconciler = Reconciler()
        # Load trained models if available, fall back to stubs
        registry = ModelRegistry()
        self._outcome_model = _try_load(registry, "game_outcome", GameOutcomeModel)
        self._env_model = _try_load(registry, "score_env", ScoreEnvironmentModel)
        self._usage_model = _try_load(registry, "player_usage", PlayerUsageModel)
        self._eff_model = _try_load(registry, "player_efficiency", PlayerEfficiencyModel)

    def run(
        self,
        home_team: str,
        away_team: str,
        home_features: pd.DataFrame,
        away_features: pd.DataFrame,
        home_player_features: pd.DataFrame,
        away_player_features: pd.DataFrame,
    ) -> SimulationResult:
        logger.info("Running %d Monte Carlo draws: %s vs %s", self.n_draws, home_team, away_team)

        # ── Model inference (deterministic, before simulation) ────────────────
        home_outcome = self._outcome_model.predict(home_features)
        home_env = self._env_model.predict(home_features)
        away_env = self._env_model.predict(away_features)

        home_contexts = self._build_player_contexts(home_player_features, home_team, is_home=True)
        away_contexts = self._build_player_contexts(away_player_features, away_team, is_home=False)

        rng = np.random.default_rng(self.seed)
        draws: list[DrawRecord] = []

        for _ in range(self.n_draws):
            # Team-level draw
            gd = self._game_draw.draw(rng, home_outcome, home_env, away_env)

            # Player-level draws
            home_player_draws = [
                self._player_draw.draw(rng, ctx, gd.home_pass_att, gd.home_rush_att)
                for ctx in home_contexts
            ]
            away_player_draws = [
                self._player_draw.draw(rng, ctx, gd.away_pass_att, gd.away_rush_att)
                for ctx in away_contexts
            ]

            # Reconcile
            home_player_draws, away_player_draws = self._reconciler.reconcile(
                gd, home_player_draws, away_player_draws,
                home_contexts, away_contexts, rng,
            )

            # Collect stats
            player_stats = [
                _player_to_dict(p) for p in home_player_draws + away_player_draws
            ]

            draws.append(DrawRecord(
                home_score=gd.home_score,
                away_score=gd.away_score,
                home_pass_yards=gd.home_pass_yards,
                away_pass_yards=gd.away_pass_yards,
                home_rush_yards=gd.home_rush_yards,
                away_rush_yards=gd.away_rush_yards,
                home_pass_att=gd.home_pass_att,
                away_pass_att=gd.away_pass_att,
                player_stats=player_stats,
            ))

        logger.info("Simulation complete: %d draws", len(draws))
        return SimulationResult(home_team, away_team, draws)

    def _build_player_contexts(
        self,
        features_df: pd.DataFrame,
        team: str,
        is_home: bool,
    ) -> list[PlayerContext]:
        contexts = []
        for _, row in features_df.iterrows():
            usage = self._usage_model.predict(pd.DataFrame([row]))
            eff = self._eff_model.predict(pd.DataFrame([row]))

            ctx = PlayerContext(
                player_id=str(row.get("player_id", "")),
                player_name=str(row.get("player_name", "")),
                team=team,
                position=str(row.get("position", "UNK")),
                is_home=is_home,
                availability=float(row.get("availability", 1.0)),
                targets_projected=usage["targets_projected"],
                carries_projected=usage["carries_projected"],
                pass_attempts_projected=usage["pass_attempts_projected"],
                catch_rate=eff["catch_rate"] or 0.65,
                yards_per_target=eff["yards_per_target"] or 8.0,
                yards_per_carry=eff["yards_per_carry"] or 4.2,
                td_rate_per_target=eff["td_rate_per_target"] or 0.05,
                td_rate_per_carry=eff["td_rate_per_carry"] or 0.04,
                yards_per_target_std=eff["yards_per_target_std"],
                yards_per_carry_std=eff["yards_per_carry_std"],
            )
            contexts.append(ctx)

        contexts = _allocate_qb_pass_volume(contexts)
        return contexts


def _allocate_qb_pass_volume(contexts: list[PlayerContext]) -> list[PlayerContext]:
    """Give all pass and rush volume to the primary QB; backups get zero."""
    qbs = [ctx for ctx in contexts if ctx.position == "QB"]
    if len(qbs) <= 1:
        return contexts
    primary = max(qbs, key=lambda c: c.pass_attempts_projected + c.carries_projected)
    for ctx in qbs:
        if ctx is not primary:
            ctx.pass_attempts_projected = 0.0
            ctx.carries_projected = 0.0
    return contexts


def _try_load(registry: ModelRegistry, name: str, fallback_cls):
    try:
        return registry.load(name)
    except FileNotFoundError:
        logger.debug("No trained %s found; using stub", name)
        return fallback_cls()


def _player_to_dict(p: PlayerDrawResult) -> dict:
    return {
        "player_id": p.player_id,
        "player_name": p.player_name,
        "team": p.team,
        "position": p.position,
        "is_home": p.is_home,
        "played": p.played,
        "targets": p.targets,
        "receptions": p.receptions,
        "rec_yards": p.rec_yards,
        "carries": p.carries,
        "rush_yards": p.rush_yards,
        "pass_attempts": p.pass_attempts,
        "completions": p.completions,
        "pass_yards": p.pass_yards,
        "tds": p.tds,
    }
