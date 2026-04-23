"""Weekly refresh workflow: ingest → silver → gold features → target backfill."""
from __future__ import annotations

import logging
from datetime import timedelta

from ironclad.config import KNOWLEDGE_CUTOFF_MARGIN_MINUTES
from ironclad.features.player_features import PlayerFeatureBuilder
from ironclad.features.team_features import TeamFeatureBuilder, _parse_kickoff
from ironclad.ingest.pipeline import IngestPipeline
from ironclad.store.connection import get_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.silver import SilverTransformer
from ironclad.store.targets import TargetBackfiller

logger = logging.getLogger(__name__)


class WeeklyWorkflow:
    def run(self, season: int, week: int) -> dict:
        """Full weekly cycle for a given season/week.

        Steps:
          1. Ingest latest data (schedules, PBP, rosters, snap counts, etc.)
          2. Rebuild silver tables for this season
          3. Build gold team + player features for every game in this week
          4. Backfill gold target columns for completed games
          5. Print a summary of week's matchups
        """
        logger.info("=== Weekly refresh: season=%d week=%d ===", season, week)
        conn = get_connection()
        create_all_tables(conn)

        # ── 1. Ingest ─────────────────────────────────────────────────────────
        logger.info("Ingesting latest data for season %d...", season)
        pipeline = IngestPipeline()
        counts = pipeline.run([season], include_pbp=True)
        logger.info("Ingest counts: %s", counts)

        # ── 2. Silver ─────────────────────────────────────────────────────────
        logger.info("Rebuilding silver tables for season %d...", season)
        silver = SilverTransformer(conn)
        silver_counts = silver.run([season])
        logger.info("Silver counts: %s", silver_counts)

        # ── 3. Gold features for this week's games ────────────────────────────
        games = conn.execute("""
            SELECT game_id, gameday, gametime_local, home_team, away_team
            FROM silver.games
            WHERE season = ? AND week = ?
            ORDER BY gameday
        """, [season, week]).df()

        team_builder = TeamFeatureBuilder(conn)
        player_builder = PlayerFeatureBuilder(conn)
        feature_count = 0

        for _, g in games.iterrows():
            try:
                cutoff = (
                    _parse_kickoff(g["gameday"], g.get("gametime_local"))
                    - timedelta(minutes=KNOWLEDGE_CUTOFF_MARGIN_MINUTES)
                )
                team_builder.build_for_game(g["game_id"], cutoff)
                player_builder.build_for_game(g["game_id"], cutoff)
                feature_count += 1
            except Exception as exc:
                logger.warning("Feature build failed for %s: %s", g["game_id"], exc)

        logger.info("Built features for %d games in week %d", feature_count, week)

        # ── 4. Target backfill for completed games ────────────────────────────
        logger.info("Backfilling targets for completed games in season %d...", season)
        backfiller = TargetBackfiller(conn)
        target_counts = backfiller.run([season])
        logger.info("Target backfill counts: %s", target_counts)

        # ── 5. Summary ────────────────────────────────────────────────────────
        if games.empty:
            logger.info("No games found for week %d season %d", week, season)
        else:
            logger.info("Week %d matchups:", week)
            for _, g in games.iterrows():
                logger.info("  %s  %s @ %s", g["gameday"], g.get("away_team", "?"), g.get("home_team", "?"))

        logger.info("=== Weekly refresh complete: season=%d week=%d ===", season, week)
        return {
            "ingest": counts,
            "silver": silver_counts,
            "features_built": feature_count,
            "targets": target_counts,
        }
