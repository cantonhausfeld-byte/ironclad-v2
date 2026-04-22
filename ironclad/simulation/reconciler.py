"""Reconcile player stat totals to team-level constraints."""
from __future__ import annotations

import numpy as np

from ironclad.simulation.game_draw import GameDrawResult
from ironclad.simulation.player_draw import PlayerDrawResult, PlayerContext


class Reconciler:
    """Scales player yards to match team totals and distributes TDs."""

    def reconcile(
        self,
        game: GameDrawResult,
        home_players: list[PlayerDrawResult],
        away_players: list[PlayerDrawResult],
        home_contexts: list[PlayerContext],
        away_contexts: list[PlayerContext],
        rng: np.random.Generator,
    ) -> tuple[list[PlayerDrawResult], list[PlayerDrawResult]]:
        home = self._reconcile_team(
            game.home_score, game.home_pass_yards, game.home_rush_yards,
            home_players, home_contexts, rng,
        )
        away = self._reconcile_team(
            game.away_score, game.away_pass_yards, game.away_rush_yards,
            away_players, away_contexts, rng,
        )
        return home, away

    def _reconcile_team(
        self,
        team_score: int,
        team_pass_yards: float,
        team_rush_yards: float,
        players: list[PlayerDrawResult],
        contexts: list[PlayerContext],
        rng: np.random.Generator,
    ) -> list[PlayerDrawResult]:
        players = [p for p in players]  # copy list

        # ── Yard scaling ──────────────────────────────────────────────────────
        total_rec = sum(p.rec_yards for p in players)
        if total_rec > 0:
            scale = team_pass_yards / total_rec
            scale = max(0.3, min(3.0, scale))
            for p in players:
                p.rec_yards = max(0.0, p.rec_yards * scale)

        total_rush = sum(p.rush_yards for p in players)
        if total_rush > 0:
            scale = team_rush_yards / total_rush
            scale = max(0.3, min(3.0, scale))
            for p in players:
                p.rush_yards = max(0.0, p.rush_yards * scale)

        # ── TD distribution ───────────────────────────────────────────────────
        # Estimate team TDs from score (rough: subtract 3 FGs worth, rest are TDs)
        team_tds = max(0, round((team_score - 3) / 7)) if team_score >= 7 else 0

        ctx_by_id = {c.player_id: c for c in contexts}
        eligible = [p for p in players if p.played and p.player_id in ctx_by_id]

        if team_tds > 0 and eligible:
            # Weight by usage: pass players use td_rate_per_target, rush use td_rate_per_carry
            weights = []
            for p in eligible:
                ctx = ctx_by_id[p.player_id]
                w = 0.0
                if p.targets > 0 and ctx.td_rate_per_target:
                    w += p.targets * ctx.td_rate_per_target
                if p.carries > 0 and ctx.td_rate_per_carry:
                    w += p.carries * ctx.td_rate_per_carry
                if p.pass_attempts > 0:  # QB passing TDs
                    w += p.completions * 0.05
                weights.append(max(w, 0.01))

            total_w = sum(weights)
            probs = [w / total_w for w in weights]

            # Assign TDs by sampling
            td_counts = {p.player_id: 0 for p in eligible}
            td_indices = rng.choice(len(eligible), size=team_tds, p=probs, replace=True)
            for idx in td_indices:
                td_counts[eligible[idx].player_id] += 1

            for p in players:
                p.tds = td_counts.get(p.player_id, 0)

        return players
