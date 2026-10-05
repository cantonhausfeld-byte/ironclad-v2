"""Monte Carlo simulation engine: orchestrates N game draws."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ironclad.config import DEFAULT_N_DRAWS
from ironclad.models.bias_corrector import TeamBiasCorrector
from ironclad.models.player.efficiency import PlayerEfficiencyModel
from ironclad.models.player.usage import PlayerUsageModel
from ironclad.models.registry import ModelRegistry
from ironclad.models.team.game_outcome import GameOutcomeModel, pivot_game_rows
from ironclad.models.team.score_env import ScoreEnvironmentModel
from ironclad.simulation.game_draw import GameDraw
from ironclad.simulation.player_draw import PlayerContext, PlayerDraw, PlayerDrawResult
from ironclad.simulation.reconciler import Reconciler
from ironclad.simulation.results import DrawRecord, SimulationResult

logger = logging.getLogger(__name__)


class MonteCarloEngine:
    def __init__(
        self,
        n_draws: int = DEFAULT_N_DRAWS,
        seed: int = 42,
        use_drive_sim: bool = False,
    ) -> None:
        self.n_draws = n_draws
        self.seed = seed
        if use_drive_sim:
            from ironclad.simulation.drive_sim import DriveSimulator
            self._game_draw = DriveSimulator()
        else:
            self._game_draw = GameDraw()
        self._player_draw = PlayerDraw()
        self._reconciler = Reconciler()
        # Load trained models if available, fall back to stubs
        registry = ModelRegistry()
        self._outcome_model = _try_load(registry, "game_outcome", GameOutcomeModel)
        self._env_model = _try_load(registry, "score_env", ScoreEnvironmentModel)
        self._usage_model = _try_load(registry, "player_usage", PlayerUsageModel)
        self._eff_model = _try_load(registry, "player_efficiency", PlayerEfficiencyModel)
        self._bias_corrector: TeamBiasCorrector | None = _try_load_optional(registry, "team_bias")

    def _predict_outcome(
        self,
        home_team: str,
        away_team: str,
        home_features: pd.DataFrame,
        away_features: pd.DataFrame,
    ) -> dict:
        """Game outcome from the same one-row-per-game shape the model trains on."""
        game_X = pivot_game_rows(home_features, away_features)
        outcome = self._outcome_model.predict(game_X)
        # Apply per-team bias correction to margin if corrector is available.
        if self._bias_corrector is not None:
            corrected = self._bias_corrector.correct(
                home_team, away_team, outcome["home_margin_mean"]
            )
            outcome = {**outcome, "home_margin_mean": corrected}
        return outcome

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
        home_outcome = self._predict_outcome(home_team, away_team, home_features, away_features)
        home_env = self._env_model.predict(home_features)
        away_env = self._env_model.predict(away_features)

        home_contexts = self._build_player_contexts(home_player_features, home_team, is_home=True)
        away_contexts = self._build_player_contexts(away_player_features, away_team, is_home=False)

        rng = np.random.default_rng(self.seed)
        draws: list[DrawRecord] = []

        for _ in range(self.n_draws):
            # Team-level draw
            gd = self._game_draw.draw(rng, home_outcome, home_env, away_env)

            # Player-level draws — pass QB game-quality factor so all
            # skill-position players on the same team share a correlated
            # efficiency adjustment within this draw.
            home_player_draws = [
                self._player_draw.draw(
                    rng, ctx, gd.home_pass_att, gd.home_rush_att,
                    game_quality_factor=gd.home_game_quality_factor,
                )
                for ctx in home_contexts
            ]
            away_player_draws = [
                self._player_draw.draw(
                    rng, ctx, gd.away_pass_att, gd.away_rush_att,
                    game_quality_factor=gd.away_game_quality_factor,
                )
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
                # Net of sacks, matching the QB lines
                home_pass_yards=max(0.0, gd.home_pass_yards - gd.home_sack_yards),
                away_pass_yards=max(0.0, gd.away_pass_yards - gd.away_sack_yards),
                home_rush_yards=gd.home_rush_yards,
                away_rush_yards=gd.away_rush_yards,
                home_pass_att=gd.home_pass_att,
                away_pass_att=gd.away_pass_att,
                home_turnovers=gd.home_turnovers,
                away_turnovers=gd.away_turnovers,
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
                depth_team=_int_or_none(row.get("depth_team")),
            )
            contexts.append(ctx)

        contexts = _allocate_qb_pass_volume(contexts)
        contexts = _allocate_rb_rush_volume(contexts)
        contexts = _allocate_wr_target_volume(contexts)
        _normalize_team_volume(contexts)
        return contexts


# PlayerDraw scales projections by team_pass_att / 32 and team_rush_att / 25,
# so a team's projections should add up to its real volume at that baseline:
# 0.903 targets per pass attempt and 1.0 carries per rush attempt (2016-2024).
_TEAM_TARGETS = 0.903 * 32.0
_TEAM_CARRIES = 1.0 * 25.0
# Shares are sharpened (projection ** gamma, then renormalized): the usage
# model's intercept gives every rostered player 1-2 targets/carries, which
# spread ~30% of yards to players who don't record a stat. Calibrated against
# the real top-3 receiver share and the prop backtest (docs/metrics).
_USAGE_GAMMA = 2.0


def _normalize_team_volume(contexts: list[PlayerContext], gamma: float | None = None) -> None:
    """Rescale a team's projected targets and carries to real team volume.

    Targets are sharpened within each position group (WR, TE, RB/FB) so each
    group keeps its share of the team's targets. QB carries (scrambles,
    designed runs) are kept as projected; only the other ball carriers share
    the remaining team carries.
    """
    g = _USAGE_GAMMA if gamma is None else gamma

    def expected(cs, attr):
        return sum(getattr(c, attr) * c.availability for c in cs)

    def sharpen(cs, attr, total):
        weights = [getattr(c, attr) ** g for c in cs]
        exp = sum(w * c.availability for w, c in zip(weights, cs))
        if exp > 0:
            for w, c in zip(weights, cs):
                setattr(c, attr, total * w / exp)

    receivers = [c for c in contexts if c.targets_projected > 0 and c.position != "QB"]
    all_targets = expected(receivers, "targets_projected")
    if all_targets > 0:
        for group in (("WR",), ("TE",), ("RB", "FB")):
            members = [c for c in receivers if c.position in group]
            share = expected(members, "targets_projected") / all_targets
            sharpen(members, "targets_projected", _TEAM_TARGETS * share)

    qb_carries = expected([c for c in contexts if c.position == "QB"], "carries_projected")
    rushers = [c for c in contexts if c.carries_projected > 0 and c.position != "QB"]
    sharpen(rushers, "carries_projected", max(0.0, _TEAM_CARRIES - qb_carries))


def _allocate_rb_rush_volume(contexts: list[PlayerContext]) -> list[PlayerContext]:
    """Zero carries for RBs ranked 5th or lower by projected carries.

    Supplemental roster RBs with historical carry share from prior roles
    dilute starter yards through the reconciler scale factor.
    """
    rbs = [ctx for ctx in contexts if ctx.position in ("RB", "FB")]
    if len(rbs) <= 4:
        return contexts
    rbs.sort(key=lambda c: c.carries_projected, reverse=True)
    for ctx in rbs[4:]:
        ctx.carries_projected = 0.0
    return contexts


def _allocate_wr_target_volume(contexts: list[PlayerContext]) -> list[PlayerContext]:
    """Zero targets for WRs ranked 5th or lower by projected targets.

    With 6+ WRs simulated, the reconciler scale factor drops below 1.0 and
    each WR gets fewer yards than their true expected share. Concentrating
    volume on the top 4 WRs matches real NFL distributions.
    """
    wrs = [ctx for ctx in contexts if ctx.position == "WR"]
    if len(wrs) <= 4:
        return contexts
    wrs.sort(key=lambda c: c.targets_projected, reverse=True)
    for ctx in wrs[4:]:
        ctx.targets_projected = 0.0
    return contexts


def _allocate_qb_pass_volume(contexts: list[PlayerContext]) -> list[PlayerContext]:
    """Give the team's QB volume to the starter; backups get zero.

    The starter is the depth-chart QB1 who isn't ruled out (next on the chart
    otherwise), falling back to projected volume without depth data. Picking by
    recent volume chose the previous starter after every QB change (a rookie
    taking over, a veteran back from injury). The starter inherits the team's
    passing volume, which his own short history understates; rushing stays his.
    """
    qbs = [ctx for ctx in contexts if ctx.position == "QB"]
    if len(qbs) <= 1:
        return contexts
    team_pass_att = max(c.pass_attempts_projected for c in qbs)
    charted = sorted(
        (c for c in qbs if c.depth_team is not None and c.availability > 0),
        key=lambda c: c.depth_team,
    )
    if charted:
        primary = charted[0]
    else:
        primary = max(qbs, key=lambda c: c.pass_attempts_projected + c.carries_projected)
    for ctx in qbs:
        if ctx is not primary:
            ctx.pass_attempts_projected = 0.0
            ctx.carries_projected = 0.0
    primary.pass_attempts_projected = team_pass_att
    return contexts


def _int_or_none(v) -> int | None:
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


def _try_load(registry: ModelRegistry, name: str, fallback_cls):
    try:
        return registry.load(name)
    except FileNotFoundError:
        logger.debug("No trained %s found; using stub", name)
        return fallback_cls()


def _try_load_optional(registry: ModelRegistry, name: str):
    """Load a model that has no stub fallback; return None if not found."""
    try:
        return registry.load(name)
    except FileNotFoundError:
        logger.debug("No trained %s found; skipping", name)
        return None


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
