"""Single player Monte Carlo draw within a game realization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ironclad.simulation.distributions import (
    bernoulli,
    poisson_draw,
    truncated_normal,
)


@dataclass
class PlayerContext:
    player_id: str
    player_name: str
    team: str
    position: str
    is_home: bool
    availability: float
    # Model outputs
    targets_projected: float
    carries_projected: float
    pass_attempts_projected: float
    catch_rate: float
    yards_per_target: float
    yards_per_carry: float
    td_rate_per_target: float
    td_rate_per_carry: float
    yards_per_target_std: float = 5.0
    yards_per_carry_std: float = 3.5


@dataclass
class PlayerDrawResult:
    player_id: str
    player_name: str
    team: str
    position: str
    is_home: bool
    # Raw draw results
    targets: int = 0
    receptions: int = 0
    rec_yards: float = 0.0
    carries: int = 0
    rush_yards: float = 0.0
    pass_attempts: int = 0
    completions: int = 0
    pass_yards: float = 0.0
    tds: int = 0
    played: bool = True


class PlayerDraw:
    """Draws one player's stat line within a game realization."""

    def draw(
        self,
        rng: np.random.Generator,
        ctx: PlayerContext,
        team_pass_att: int,
        team_rush_att: int,
        game_quality_factor: float = 0.0,
    ) -> PlayerDrawResult:
        result = PlayerDrawResult(
            player_id=ctx.player_id,
            player_name=ctx.player_name,
            team=ctx.team,
            position=ctx.position,
            is_home=ctx.is_home,
        )

        # Availability Bernoulli gate
        if not bernoulli(rng, ctx.availability):
            result.played = False
            return result

        # Scale projected volume to actual team volume in this draw
        pass_scale = team_pass_att / 32.0 if team_pass_att > 0 else 0.0
        rush_scale = team_rush_att / 25.0 if team_rush_att > 0 else 0.0

        # ── Receiving ─────────────────────────────────────────────────────────
        if ctx.position in ("WR", "TE", "RB", "FB") and ctx.targets_projected > 0:
            # Correlate receiver efficiency with QB quality for this draw.
            # gqf=+0.15 (1 std) → +4.5% catch rate, +3.75% yards/target.
            adj_catch_rate = min(0.99, (ctx.catch_rate or 0.65) * (1.0 + game_quality_factor * 0.30))
            adj_yds_per_tgt = max(1.0, (ctx.yards_per_target or 8.0) * (1.0 + game_quality_factor * 0.25))

            lam = max(0.0, ctx.targets_projected * pass_scale)
            # Negative-binomial for targets: receiver target counts have
            # more game-script variance than Poisson allows.
            # r=15 → std ≈ 2.9 at lam=6 (vs Poisson std=2.45), targeting 80% coverage.
            if lam > 0:
                r_tgt = 15
                targets = int(rng.negative_binomial(r_tgt, r_tgt / (r_tgt + lam)))
            else:
                targets = 0
            receptions = 0
            rec_yards = 0.0
            for _ in range(targets):
                if bernoulli(rng, adj_catch_rate):
                    receptions += 1
                    yds = float(truncated_normal(
                        rng,
                        mean=adj_yds_per_tgt,
                        std=ctx.yards_per_target_std,
                        low=0.0, high=80.0,
                    )[0])
                    rec_yards += yds
            result.targets = targets
            result.receptions = receptions
            result.rec_yards = rec_yards

        # ── Rushing ───────────────────────────────────────────────────────────
        if ctx.carries_projected > 0:
            lam = max(0.0, ctx.carries_projected * rush_scale)
            if lam > 0:
                if ctx.position == "QB":
                    # QB carries are lower-volume and more predictable (scrambles);
                    # NB overdispersion at lam<5 blows coverage to ~100%. Poisson
                    # is appropriate here — game-script variance is captured by lam.
                    carries = poisson_draw(rng, lam)
                else:
                    # NB for RB/FB: carry counts have high game-script variance
                    # (std ≈ 5-7 at mean 12); Poisson is too narrow.
                    # r=4 → var = lam*(1+lam/4), std ≈ 5.5 at lam=10.
                    r_car = 4
                    carries = int(rng.negative_binomial(r_car, r_car / (r_car + lam)))
            else:
                carries = 0
            rush_yards = 0.0
            for _ in range(carries):
                yds = float(truncated_normal(
                    rng,
                    mean=ctx.yards_per_carry or 4.2,
                    std=ctx.yards_per_carry_std,
                    low=0.0, high=80.0,
                )[0])
                rush_yards += yds
            result.carries = carries
            result.rush_yards = rush_yards

        # ── Passing (QB) ──────────────────────────────────────────────────────
        if ctx.position == "QB" and ctx.pass_attempts_projected > 0:
            lam = max(0.0, ctx.pass_attempts_projected * pass_scale)
            # Negative-binomial instead of Poisson: real NFL QB pass attempts
            # have std ≈ 9-10 (game-script, blowout, weather variance); Poisson
            # gives std ≈ 6 which is too narrow and causes under-coverage.
            # r=12 → var = lam*(1 + lam/12), std ≈ 9 at lam=32.
            r_att = 12
            p_att = r_att / (r_att + lam) if lam > 0 else 0.5
            pass_att = int(rng.negative_binomial(r_att, p_att))
            # Clamp to realistic NFL completion rate range; ctx.catch_rate may
            # reflect a receiving catch rate (meaningless for QBs) rather than
            # pass completion rate. Apply game-quality factor to QB accuracy.
            base_comp_rate = max(0.50, min(0.75, ctx.catch_rate or 0.64))
            comp_rate = max(0.45, min(0.80, base_comp_rate * (1.0 + game_quality_factor * 0.20)))
            completions = int(rng.binomial(pass_att, comp_rate))
            adj_ypc = (ctx.yards_per_target or 8.5) * (1.0 + game_quality_factor * 0.20)
            pass_yards = max(0.0, float(rng.normal(
                completions * adj_ypc,
                completions * 3.5,
            ))) if completions > 0 else 0.0
            result.pass_attempts = pass_att
            result.completions = completions
            result.pass_yards = pass_yards

        # ── TDs (assigned during reconciliation; set 0 here) ──────────────────
        result.tds = 0
        return result
