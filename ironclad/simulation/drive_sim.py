"""Drive-level Markov chain game simulator.

Produces the same GameDrawResult interface as GameDraw but derives scores from
simulated possession sequences rather than sampling from score distributions.
Game-script effects (trailing teams passing more, leading teams running) emerge
naturally from the per-drive pass-rate adjustment.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from ironclad.simulation.game_draw import GameDrawResult

LEAGUE_AVG_PTS = 23.5
_HALF_SECS = 1800

# Drive base rates (quality=1.0, neutral field position)
# Calibrated: 11 drives × (0.25×7 + 0.18×3) pts ≈ 24 pts/team ≈ NFL average
_BASE_TD = 0.25
_BASE_FG = 0.18
_BASE_TO = 0.10

# Per-outcome drive statistics (mean values; 30% CV applied via normal draw)
_DRIVE_PLAYS = {"TD": 9, "FG": 7, "PUNT": 6, "TURNOVER": 4}
_DRIVE_YARDS = {"TD": 65.0, "FG": 40.0, "PUNT": 25.0, "TURNOVER": 20.0}
_DRIVE_SECS  = {"TD": 210, "FG": 175, "PUNT": 150, "TURNOVER": 110}

# FG make probability by max kick distance (yards)
_FG_DIST_PROB = [(30, 0.97), (40, 0.92), (50, 0.82), (60, 0.65)]
_FG_MAX_ATTEMPT = 60  # beyond this → convert to punt


class DriveOutcome(Enum):
    TD       = "TD"
    FG       = "FG"
    PUNT     = "PUNT"
    TURNOVER = "TURNOVER"


class DriveSimulator:
    """Drop-in replacement for GameDraw using drive-level Markov chain."""

    def draw(
        self,
        rng: np.random.Generator,
        home_outcome: dict,  # noqa: ARG002  accepted for interface compatibility; unused
        home_env: dict,
        away_env: dict,
    ) -> GameDrawResult:
        home_quality = home_env["team_total_projected"] / LEAGUE_AVG_PTS
        away_quality = away_env["team_total_projected"] / LEAGUE_AVG_PTS

        home_pass_rate = float(np.clip(home_env["pass_rate_projected"], 0.20, 0.85))
        away_pass_rate = float(np.clip(away_env["pass_rate_projected"], 0.20, 0.85))

        # QB game-quality factors for player-level stat correlation (same as GameDraw)
        home_gqf = float(rng.normal(0.0, 0.15))
        away_gqf = float(rng.normal(0.0, 0.15))

        s = self._simulate_game(rng, home_quality, away_quality, home_pass_rate, away_pass_rate)

        home_pass_att = max(0, s["home_pass"])
        away_pass_att = max(0, s["away_pass"])
        home_rush_att = max(0, s["home_rush"])
        away_rush_att = max(0, s["away_rush"])

        # Yardage sampled from attempts (mirrors GameDraw approach)
        home_pass_yards = max(0.0, float(rng.normal(home_pass_att * 7.2, home_pass_att * 2.5 + 1)))
        away_pass_yards = max(0.0, float(rng.normal(away_pass_att * 7.2, away_pass_att * 2.5 + 1)))
        home_rush_yards = max(0.0, float(rng.normal(home_rush_att * 4.3, 25.0)))
        away_rush_yards = max(0.0, float(rng.normal(away_rush_att * 4.3, 25.0)))

        total_plays = home_pass_att + home_rush_att + away_pass_att + away_rush_att
        home_eff = home_pass_att / max(1, home_pass_att + home_rush_att)
        away_eff = away_pass_att / max(1, away_pass_att + away_rush_att)

        return GameDrawResult(
            home_score=s["home_score"],
            away_score=s["away_score"],
            home_pass_att=home_pass_att,
            away_pass_att=away_pass_att,
            home_rush_att=home_rush_att,
            away_rush_att=away_rush_att,
            home_pass_yards=home_pass_yards,
            away_pass_yards=away_pass_yards,
            home_rush_yards=home_rush_yards,
            away_rush_yards=away_rush_yards,
            total_plays=total_plays,
            home_effective_pass_rate=home_eff,
            away_effective_pass_rate=away_eff,
            home_game_quality_factor=home_gqf,
            away_game_quality_factor=away_gqf,
            home_turnovers=s["home_tos"],
            away_turnovers=s["away_tos"],
        )

    def _simulate_game(
        self,
        rng: np.random.Generator,
        home_quality: float,
        away_quality: float,
        home_pass_rate: float,
        away_pass_rate: float,
    ) -> dict:
        home_score = away_score = 0
        home_pass = away_pass = home_rush = away_rush = 0
        home_tos = away_tos = 0

        # Away team gets first-half possession; home receives to start second half
        for half_first_poss in ("away", "home"):
            clock = _HALF_SECS
            possession = half_first_poss
            yard_line = 25  # yards from own end zone after kickoff (0=own goal, 100=opp goal)

            while clock > 60:
                is_home = possession == "home"
                quality     = home_quality  if is_home else away_quality
                base_rate   = home_pass_rate if is_home else away_pass_rate
                score_diff  = (home_score - away_score) if is_home else (away_score - home_score)
                two_min     = clock < 120

                # ── Pass rate adjusted for game script ───────────────────────
                adj_rate = self._script_pass_rate(base_rate, score_diff, two_min)

                # ── Drive outcome ─────────────────────────────────────────────
                outcome = self._drive_outcome(rng, yard_line, quality, score_diff, clock)

                # ── Plays and clock consumption ───────────────────────────────
                mean_plays = _DRIVE_PLAYS[outcome.name]
                plays = max(1, int(rng.normal(mean_plays, mean_plays * 0.30)))
                if two_min:
                    # Two-minute drill: fast plays (20 s/play) to conserve clock
                    secs = min(clock, plays * 20)
                else:
                    mean_secs = _DRIVE_SECS[outcome.name]
                    secs = min(clock, int(rng.normal(mean_secs, mean_secs * 0.20)))
                clock -= secs

                # ── Pass / rush split ─────────────────────────────────────────
                pass_plays = max(0, round(adj_rate * plays))
                rush_plays = max(0, plays - pass_plays)

                if is_home:
                    home_pass += pass_plays
                    home_rush += rush_plays
                else:
                    away_pass += pass_plays
                    away_rush += rush_plays

                # ── Resolve outcome ───────────────────────────────────────────
                if outcome == DriveOutcome.TD:
                    if is_home:
                        home_score += 7
                    else:
                        away_score += 7
                    yard_line = 25
                    possession = "away" if is_home else "home"

                elif outcome == DriveOutcome.FG:
                    distance = (100 - yard_line) + 17
                    if distance > _FG_MAX_ATTEMPT:
                        # Too far — punt instead
                        net = self._punt_net(rng)
                        yard_line = max(5, 100 - (yard_line + net))
                    else:
                        if rng.random() < self._fg_make_prob(yard_line):
                            if is_home:
                                home_score += 3
                            else:
                                away_score += 3
                        # Regardless of make/miss, kick changes possession;
                        # opponent starts at own 25 (touchback on miss over endline)
                        yard_line = 25
                    possession = "away" if is_home else "home"

                elif outcome == DriveOutcome.PUNT:
                    net = self._punt_net(rng)
                    yard_line = max(5, 100 - (yard_line + net))
                    possession = "away" if is_home else "home"

                else:  # TURNOVER
                    if is_home:
                        home_tos += 1
                    else:
                        away_tos += 1
                    # Opponent takes over near where the turnover happened;
                    # approximate with yards gained before the stop
                    gain = max(0, int(rng.normal(10, 8)))
                    yard_line = max(5, 100 - (yard_line + gain))
                    possession = "away" if is_home else "home"

        return {
            "home_score": home_score,
            "away_score": away_score,
            "home_pass": home_pass,
            "away_pass": away_pass,
            "home_rush": home_rush,
            "away_rush": away_rush,
            "home_tos": home_tos,
            "away_tos": away_tos,
        }

    def _drive_outcome(
        self,
        rng: np.random.Generator,
        yard_line: int,
        quality: float,
        score_diff: float,
        clock: float,
    ) -> DriveOutcome:
        # field_pos_factor: scales scoring probability by field position.
        # yard_line = 25 (own 25, baseline after kickoff) → factor = 1.0
        # yard_line = 100 (opponent goal) → factor = 1.5
        # yard_line = 0 (own goal) → factor = 0.83
        fp = float(np.clip(1.0 + 0.5 * ((yard_line - 25) / 75), 0.5, 2.0))

        td_prob = _BASE_TD * quality * fp
        fg_prob = _BASE_FG * quality * (fp ** 0.5)
        to_prob = _BASE_TO
        # Increase TO rate when trailing late (desperation plays)
        if score_diff < -14 and clock < 120:
            to_prob *= 2.0
        punt_prob = max(0.05, 1.0 - td_prob - fg_prob - to_prob)

        # Normalize
        total = td_prob + fg_prob + to_prob + punt_prob
        probs = [td_prob / total, fg_prob / total, to_prob / total, punt_prob / total]
        idx = int(rng.choice(4, p=probs))
        return [DriveOutcome.TD, DriveOutcome.FG, DriveOutcome.TURNOVER, DriveOutcome.PUNT][idx]

    def _fg_make_prob(self, yard_line: int) -> float:
        # distance = yards from goal posts (end zone depth 10 + snap 7)
        distance = (100 - yard_line) + 17
        for max_dist, make_prob in _FG_DIST_PROB:
            if distance <= max_dist:
                return make_prob
        return 0.50  # beyond 60 yards

    @staticmethod
    def _punt_net(rng: np.random.Generator) -> int:
        return max(10, min(65, int(rng.normal(40, 12))))

    @staticmethod
    def _script_pass_rate(base_rate: float, score_diff: float, two_min: bool) -> float:
        urgency = max(0.0, (-score_diff - 7.0) / 14.0)
        comfort = max(0.0, (score_diff - 7.0) / 14.0)

        if two_min and score_diff < 0:
            rate = min(0.85, base_rate + 0.15 + urgency * 0.12)
        elif urgency > 0:
            rate = min(0.85, base_rate + urgency * 0.12)
        elif comfort > 0:
            rate = max(0.35, base_rate - comfort * 0.10)
        else:
            rate = base_rate

        return float(np.clip(rate, 0.20, 0.85))
