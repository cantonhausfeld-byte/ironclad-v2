"""Elo team rating computer for NFL games.

Ratings are computed chronologically through all completed games and stored in
gold.elo_ratings as (game_id, team, elo_pre_game, elo_post_game).

Design decisions:
- K=20 (standard NFL Elo; FiveThirtyEight uses K=20 with margin adjustment)
- Margin adjustment: multiply K by log(|margin|+1) so blowouts move ratings more
- Season carry-over: regress 1/3 of the way toward 1500 at week 1 of each season
- Initial rating: 1500 for every team on their first appearance
"""
from __future__ import annotations

import logging
import math

import duckdb
import pandas as pd

from ironclad.store.writer import GoldWriter

logger = logging.getLogger(__name__)

_INITIAL_ELO = 1500.0
_K = 20.0
_REGRESS_FRAC = 1 / 3  # move this fraction toward 1500 each new season


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _margin_mult(margin: int) -> float:
    """Scale K by log(|margin|+1) — blowouts carry more information."""
    return math.log(abs(margin) + 1)


class EloComputer:
    def __init__(self, conn: duckdb.DuckDBPyConnection | None = None) -> None:
        from ironclad.store.connection import get_connection
        self._conn = conn or get_connection()
        self._writer = GoldWriter(self._conn)

    def compute_and_write(self) -> int:
        """Recompute Elo for all completed games and write to gold.elo_ratings."""
        rows = self._compute_all()
        if rows.empty:
            return 0
        # Wipe and rewrite — ratings are globally interdependent
        self._conn.execute("DELETE FROM gold.elo_ratings")
        n = self._writer.write_elo_ratings(rows)
        logger.info("Wrote %d Elo rating rows", n)
        return n

    def _compute_all(self) -> pd.DataFrame:
        games = self._conn.execute("""
            SELECT game_id, season, week, gameday,
                   home_team, away_team,
                   home_score, away_score
            FROM silver.games
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY gameday, game_id
        """).df()

        if games.empty:
            return pd.DataFrame()

        ratings: dict[str, float] = {}
        prev_season: int | None = None
        records: list[dict] = []

        for _, g in games.iterrows():
            season = int(g["season"])
            week   = int(g["week"])
            home   = str(g["home_team"])
            away   = str(g["away_team"])
            home_score = int(g["home_score"])
            away_score = int(g["away_score"])
            margin = home_score - away_score

            # Season start: regress all ratings toward 1500
            if season != prev_season:
                for team in list(ratings):
                    ratings[team] = ratings[team] + _REGRESS_FRAC * (_INITIAL_ELO - ratings[team])
                prev_season = season

            r_home = ratings.get(home, _INITIAL_ELO)
            r_away = ratings.get(away, _INITIAL_ELO)

            # Expected win probability for home team
            exp_home = _expected(r_home, r_away)
            actual_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)

            k_adj = _K * _margin_mult(margin)
            delta = k_adj * (actual_home - exp_home)

            new_home = r_home + delta
            new_away = r_away - delta

            records.append({"game_id": g["game_id"], "team": home, "season": season,
                            "week": week, "elo_pre_game": r_home, "elo_post_game": new_home})
            records.append({"game_id": g["game_id"], "team": away, "season": season,
                            "week": week, "elo_pre_game": r_away, "elo_post_game": new_away})

            ratings[home] = new_home
            ratings[away] = new_away

        top5 = sorted(ratings.items(), key=lambda x: -x[1])[:5]
        logger.info("Top 5 Elo after latest season: %s", top5)
        return pd.DataFrame(records)
