"""Fill in gold target columns with actual game results (post-game backfill)."""
from __future__ import annotations

import logging

from ironclad.store.connection import get_connection

logger = logging.getLogger(__name__)


class TargetBackfiller:
    """Updates target_* columns in gold tables with actual outcomes."""

    def __init__(self, conn=None) -> None:
        self._conn = conn or get_connection()

    def run(self, seasons: list[int] | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        counts["team_targets"] = self._fill_team_targets(seasons)
        counts["player_targets"] = self._fill_player_targets(seasons)
        return counts

    # ── Team targets ──────────────────────────────────────────────────────────

    def _fill_team_targets(self, seasons: list[int] | None) -> int:
        where = self._season_filter("t.season", seasons)
        rows = self._conn.execute(f"""
            UPDATE gold.team_game_features AS t
            SET
                target_points_scored = s.points_scored,
                target_yards_total   = s.total_yards,
                target_pass_rate     = s.pass_rate,
                target_home_win      = g.home_win,
                target_home_margin   = g.home_margin,
                target_total_score   = g.total_score
            FROM silver.team_game_stats s
            JOIN silver.games g ON s.game_id = g.game_id
            WHERE t.game_id = s.game_id
              AND t.team    = s.team
              AND t.target_points_scored IS NULL
              AND s.points_scored IS NOT NULL
            {where.replace('WHERE', 'AND')}
            RETURNING t.game_id
        """).df()
        updated = len(rows)
        logger.info("Filled %d team target rows", updated)
        return updated

    # ── Player targets ────────────────────────────────────────────────────────

    def _fill_player_targets(self, seasons: list[int] | None) -> int:
        where = self._season_filter("p.season", seasons)
        rows = self._conn.execute(f"""
            UPDATE gold.player_game_features AS p
            SET
                target_targets    = s.targets,
                target_carries    = s.carries,
                target_receptions = s.receptions,
                target_rec_yards  = s.rec_yards,
                target_rush_yards = s.rush_yards,
                target_total_tds  = s.total_tds
            FROM silver.player_game_stats s
            WHERE p.game_id    = s.game_id
              AND p.player_id  = s.player_id
              AND p.target_targets IS NULL
              AND s.targets IS NOT NULL
            {where.replace('WHERE', 'AND')}
            RETURNING p.game_id
        """).df()
        updated = len(rows)
        logger.info("Filled %d player target rows", updated)
        return updated

    @staticmethod
    def _season_filter(col: str, seasons: list[int] | None) -> str:
        if not seasons:
            return ""
        s = ", ".join(str(s) for s in seasons)
        return f"WHERE {col} IN ({s})"
