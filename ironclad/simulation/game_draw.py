"""Single game-level Monte Carlo draw."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ironclad.simulation.distributions import truncated_normal, beta_from_mean_std


@dataclass
class GameDrawResult:
    home_score: int
    away_score: int
    home_pass_att: int
    away_pass_att: int
    home_rush_att: int
    away_rush_att: int
    home_pass_yards: float
    away_pass_yards: float
    home_rush_yards: float
    away_rush_yards: float
    total_plays: int = 0
    home_effective_pass_rate: float = 0.0
    away_effective_pass_rate: float = 0.0


def _apply_game_script(
    rng: np.random.Generator,
    base_rate: float,
    final_margin: float,
) -> float:
    # Halftime margin estimated from final margin with substantial noise —
    # final-margin blowouts correlate with halftime leads, but not perfectly.
    # Trailing teams (>7 down at half) pass more; leading teams run more.
    # Effective rate blends first-half (55% weight, base rate) and second-half
    # (45% weight, script-adjusted).
    halftime_est = float(rng.normal(final_margin * 0.45, 7.0))
    urgency = max(0.0, (-halftime_est - 7.0) / 14.0)
    comfort = max(0.0, (halftime_est - 7.0) / 14.0)

    if urgency > 0:
        late_rate = min(0.85, base_rate + urgency * 0.12)
    elif comfort > 0:
        late_rate = max(0.35, base_rate - comfort * 0.10)
    else:
        late_rate = base_rate

    return float(np.clip(0.55 * base_rate + 0.45 * late_rate, 0.2, 0.85))


class GameDraw:
    """Draws one team-level game realization."""

    def draw(
        self,
        rng: np.random.Generator,
        home_outcome: dict,
        home_env: dict,
        away_env: dict,
    ) -> GameDrawResult:
        # ── Scores ────────────────────────────────────────────────────────────
        total = float(truncated_normal(
            rng,
            mean=home_outcome["total_mean"],
            std=max(home_outcome["total_std"], home_outcome["total_mean"] * 0.12),
            low=0.0, high=90.0,
        )[0])
        margin = float(rng.normal(home_outcome["home_margin_mean"], home_outcome["home_margin_std"]))
        home_raw = (total + margin) / 2.0
        away_raw = total - home_raw
        home_score = max(0, round(home_raw))
        away_score = max(0, round(away_raw))
        final_margin = float(home_score - away_score)

        # ── Home team volume ──────────────────────────────────────────────────
        home_plays = max(30, int(rng.normal(home_env["total_plays_projected"], 6)))
        home_base_rate = float(np.clip(
            rng.beta(*beta_from_mean_std(home_env["pass_rate_projected"], 0.06)),
            0.2, 0.85,
        ))
        home_effective_rate = _apply_game_script(rng, home_base_rate, final_margin)
        home_pass_att = round(home_effective_rate * home_plays)
        home_rush_att = home_plays - home_pass_att

        # ── Away team volume ──────────────────────────────────────────────────
        away_plays = max(30, int(rng.normal(away_env["total_plays_projected"], 6)))
        away_base_rate = float(np.clip(
            rng.beta(*beta_from_mean_std(away_env["pass_rate_projected"], 0.06)),
            0.2, 0.85,
        ))
        away_effective_rate = _apply_game_script(rng, away_base_rate, -final_margin)
        away_pass_att = round(away_effective_rate * away_plays)
        away_rush_att = away_plays - away_pass_att

        # ── Yardage ───────────────────────────────────────────────────────────
        # Pass yards from attempts × NFL avg yards/attempt (7.2); far better
        # than score × 6 which underestimates low-scoring games and has no
        # relationship to actual passing volume drawn above.
        home_pass_yards = max(0.0, float(rng.normal(home_pass_att * 7.2, home_pass_att * 2.5)))
        away_pass_yards = max(0.0, float(rng.normal(away_pass_att * 7.2, away_pass_att * 2.5)))
        home_rush_yards = max(0.0, float(rng.normal(home_rush_att * 4.3, 25.0)))
        away_rush_yards = max(0.0, float(rng.normal(away_rush_att * 4.3, 25.0)))

        return GameDrawResult(
            home_score=home_score,
            away_score=away_score,
            home_pass_att=max(0, home_pass_att),
            away_pass_att=max(0, away_pass_att),
            home_rush_att=max(0, home_rush_att),
            away_rush_att=max(0, away_rush_att),
            home_pass_yards=home_pass_yards,
            away_pass_yards=away_pass_yards,
            home_rush_yards=home_rush_yards,
            away_rush_yards=away_rush_yards,
            total_plays=home_plays + away_plays,
            home_effective_pass_rate=home_effective_rate,
            away_effective_pass_rate=away_effective_rate,
        )
