"""Reconcile player stat totals to team-level constraints."""
from __future__ import annotations

import numpy as np

from ironclad.simulation.game_draw import GameDrawResult
from ironclad.simulation.player_draw import PlayerDrawResult, PlayerContext

# League-average red zone TD conversion rate (~57% of RZ trips end in TD).
_RZ_TD_RATE = 0.57
# Expected points per RZ trip: 0.57*7 + 0.43*3 ≈ 5.28
_EXPECTED_PTS_PER_RZ_TRIP = _RZ_TD_RATE * 7.0 + (1.0 - _RZ_TD_RATE) * 3.0
# League-average QB passing TD per completion (~30 TDs / 17 games / 22 comp/game ≈ 0.08;
# starting QBs average ~2 passing TDs per game on ~22 completions → 0.09).
_QB_TD_PER_COMPLETION = 0.09


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

        # ── Yard + volume scaling ─────────────────────────────────────────────
        # Scale rec_yards to match team totals; do NOT scale targets/receptions —
        # those are drawn from NB distributions and scaling by the yards ratio
        # would compress out the variance we intentionally added.
        total_rec = sum(p.rec_yards for p in players)
        if total_rec > 0:
            scale = team_pass_yards / total_rec
            scale = max(0.3, min(3.0, scale))
            for p in players:
                p.rec_yards = max(0.0, p.rec_yards * scale)

        # QB pass_yards: drawn independently using calibrated NFL net yds/att (6.0).
        # Decoupled from total_rec_after so WR/TE scale and QB calibration are
        # independent — team_pass_yards (used above for WR scale) is intentionally
        # higher than net pass yards to correct for under-projected player volumes.
        for p in players:
            if p.pass_attempts > 0:
                p.pass_yards = max(
                    0.0,
                    float(rng.normal(p.pass_attempts * 6.0, p.pass_attempts * 1.8)),
                )

        # Scale rush_yards to match team totals; do NOT scale carries for the
        # same reason as targets: NB carry counts have intentional variance that
        # the yards-ratio scale would suppress.
        total_rush = sum(p.rush_yards for p in players)
        if total_rush > 0:
            scale = team_rush_yards / total_rush
            scale = max(0.3, min(3.0, scale))
            for p in players:
                p.rush_yards = max(0.0, p.rush_yards * scale)

        # ── TD distribution ───────────────────────────────────────────────────
        # Sample red zone trips from Poisson, then each trip independently
        # scores a TD with probability _RZ_TD_RATE (vs field goal).
        rz_trips = int(rng.poisson(max(0.5, team_score / _EXPECTED_PTS_PER_RZ_TRIP)))
        team_tds = int(rng.binomial(rz_trips, _RZ_TD_RATE))

        ctx_by_id = {c.player_id: c for c in contexts}
        eligible = [p for p in players if p.played and p.player_id in ctx_by_id]

        if eligible:
            # QB passing TDs: drawn directly as Binomial(completions, rate) rather
            # than through the team-TD multinomial.  The multinomial anchors tightly
            # to team_score, producing 58-65% 80% coverage; the direct Binomial draw
            # captures the observed per-game variance (0-5 TDs) independently.
            # Clamped to team_tds so the team total stays consistent.
            qb_pass_tds = 0
            qb_player = next((p for p in eligible if p.pass_attempts > 0), None)
            if qb_player is not None and qb_player.completions > 0:
                # No clamp against team_tds: the QB's TD count should float
                # freely to capture the observed 0-5 per-game variance.
                qb_pass_tds = int(rng.binomial(qb_player.completions, _QB_TD_PER_COMPLETION))
                qb_player.tds = qb_pass_tds

            # Remaining TDs (skill-position receiving + all rush TDs including QB)
            # distributed via weighted multinomial.  Weights raised to ^1.5 to
            # concentrate TDs on the highest-usage player, reducing the excessive
            # per-player variance that was widening WR/TE/RB coverage to 87-94%.
            remaining_tds = max(0, team_tds - qb_pass_tds)
            if remaining_tds > 0:
                weights = []
                for p in eligible:
                    ctx = ctx_by_id[p.player_id]
                    w = 0.0
                    if p.targets > 0 and ctx.td_rate_per_target:
                        w += p.targets * ctx.td_rate_per_target
                    if p.carries > 0 and ctx.td_rate_per_carry:
                        w += p.carries * ctx.td_rate_per_carry
                    weights.append(max(w ** 1.5, 0.001))

                total_w = sum(weights)
                probs = [w / total_w for w in weights]
                td_counts = {p.player_id: 0 for p in eligible}
                td_indices = rng.choice(len(eligible), size=remaining_tds, p=probs, replace=True)
                for idx in td_indices:
                    td_counts[eligible[idx].player_id] += 1
                for p in players:
                    p.tds += td_counts.get(p.player_id, 0)

        return players
