"""Elo team ratings, computed from completed games visible at the knowledge cutoff.

Ported from the PR #1 branch (bebcb61) with three changes:
- Computed from the FeatureSnapshot's game history instead of a separate
  gold.elo_ratings table, so upcoming games get a real pre-game rating and the
  cutoff invariant holds by construction.
- Home-field advantage in the expected score (otherwise every home win inflates
  the home team's rating).
- FiveThirtyEight's margin-of-victory autocorrelation correction, so favourites
  winning big don't over-inflate.
"""
from __future__ import annotations

import math

import pandas as pd

INITIAL_ELO = 1500.0
K = 20.0
HOME_FIELD_ELO = 48.0
REGRESS_FRAC = 1 / 3  # fraction of the way back to 1500 at each new season


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _mov_mult(margin: float, winner_elo_diff: float) -> float:
    """ln(|MOV|+1) scaled down when the higher-rated team wins (538 formula)."""
    return math.log(max(abs(margin), 1) + 1) * 2.2 / (winner_elo_diff * 0.001 + 2.2)


def _regress(ratings: dict[str, float]) -> None:
    for team in ratings:
        ratings[team] += REGRESS_FRAC * (INITIAL_ELO - ratings[team])


def compute_ratings(games: pd.DataFrame, as_of_season: int) -> dict[str, float]:
    """Team ratings after replaying *games* (completed, pre-cutoff) in order.

    games needs: season, gameday, game_id, home_team, away_team, home_score,
    away_score. Ratings are regressed at every season boundary, including the
    one into *as_of_season* if no games from it have been played yet.
    """
    ratings: dict[str, float] = {}
    done = games.dropna(subset=["home_score", "away_score"])
    done = done.sort_values(["gameday", "game_id"])

    prev_season: int | None = None
    for g in done.itertuples(index=False):
        season = int(g.season)
        if prev_season is not None and season != prev_season:
            _regress(ratings)
        prev_season = season

        r_home = ratings.get(g.home_team, INITIAL_ELO)
        r_away = ratings.get(g.away_team, INITIAL_ELO)
        margin = float(g.home_score) - float(g.away_score)

        diff = r_home + HOME_FIELD_ELO - r_away
        exp_home = _expected(r_home + HOME_FIELD_ELO, r_away)
        actual = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        winner_diff = diff if margin >= 0 else -diff

        delta = K * _mov_mult(margin, winner_diff) * (actual - exp_home)
        ratings[g.home_team] = r_home + delta
        ratings[g.away_team] = r_away - delta

    if prev_season is not None and as_of_season > prev_season:
        _regress(ratings)
    return ratings
