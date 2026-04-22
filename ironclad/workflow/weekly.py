"""Weekly refresh workflow: ingest current week + build silver."""
from __future__ import annotations

import logging

from ironclad.ingest.pipeline import IngestPipeline
from ironclad.store.silver import SilverTransformer
from ironclad.store.connection import get_connection
from ironclad.store.schema import create_all_tables

logger = logging.getLogger(__name__)


class WeeklyWorkflow:
    def run(self, season: int, week: int) -> None:
        logger.info("Starting weekly refresh: season=%d week=%d", season, week)
        conn = get_connection()
        create_all_tables(conn)

        pipeline = IngestPipeline()
        counts = pipeline.run([season], include_pbp=True)
        logger.info("Ingest counts: %s", counts)

        silver = SilverTransformer(conn)
        silver_counts = silver.run([season])
        logger.info("Silver counts: %s", silver_counts)

        # Print upcoming games for the week
        games = conn.execute("""
            SELECT game_id, gameday, away_team, home_team
            FROM silver.games
            WHERE season = ? AND week = ?
            ORDER BY gameday
        """, [season, week]).df()

        if games.empty:
            logger.info("No games found for week %d", week)
        else:
            logger.info("Week %d games:", week)
            for _, g in games.iterrows():
                logger.info("  %s  %s @ %s", g["gameday"], g["away_team"], g["home_team"])

        logger.info("Weekly refresh complete")
