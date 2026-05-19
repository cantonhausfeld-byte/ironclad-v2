"""Backfill historical data: ingest + silver transform + gold features + targets."""
from __future__ import annotations

import logging

from ironclad.ingest.pipeline import IngestPipeline
from ironclad.store.silver import SilverTransformer
from ironclad.store.targets import TargetBackfiller
from ironclad.features.team_features import TeamFeatureBuilder
from ironclad.features.player_features import PlayerFeatureBuilder
from ironclad.store.connection import get_connection
from ironclad.store.schema import create_all_tables

logger = logging.getLogger(__name__)


class BackfillWorkflow:
    def run(
        self,
        seasons: list[int],
        include_pbp: bool = True,
        build_features: bool = True,
    ) -> dict:
        logger.info("Starting backfill for seasons: %s", seasons)
        conn = get_connection()
        create_all_tables(conn)

        pipeline = IngestPipeline()
        counts = pipeline.run(seasons, include_pbp=include_pbp)
        logger.info("Ingest counts: %s", counts)

        logger.info("Building silver tables...")
        silver = SilverTransformer(conn)
        silver_counts = silver.run(seasons)
        logger.info("Silver counts: %s", silver_counts)

        features_built = 0
        features_failed = 0

        if build_features and include_pbp:
            logger.info("Building gold team features...")
            team_builder = TeamFeatureBuilder(conn)
            for season in seasons:
                n = team_builder.build_for_season(season)
                logger.info("Season %d: built features for %d games", season, n)

            logger.info("Building gold player features...")
            player_builder = PlayerFeatureBuilder(conn)
            games = conn.execute(
                "SELECT game_id, gameday, gametime_local, season FROM silver.games WHERE season IN ({})".format(
                    ", ".join(str(s) for s in seasons)
                )
            ).df()
            from ironclad.features.team_features import _parse_kickoff
            from datetime import timedelta
            from ironclad.config import KNOWLEDGE_CUTOFF_MARGIN_MINUTES
            for _, g in games.iterrows():
                try:
                    cutoff = _parse_kickoff(g["gameday"], g.get("gametime_local")) - timedelta(minutes=KNOWLEDGE_CUTOFF_MARGIN_MINUTES)
                    player_builder.build_for_game(g["game_id"], cutoff)
                    features_built += 1
                except Exception as exc:
                    features_failed += 1
                    logger.warning("Player features failed for %s: %s", g["game_id"], exc)

            if features_failed > 0:
                logger.warning(
                    "Player feature build: %d succeeded, %d failed",
                    features_built, features_failed,
                )

            logger.info("Backfilling gold target columns...")
            backfiller = TargetBackfiller(conn)
            target_counts = backfiller.run(seasons)
            logger.info("Target backfill counts: %s", target_counts)

        logger.info("Backfill complete for seasons %s", seasons)
        return {
            "ingest": counts,
            "silver": silver_counts,
            "features_built": features_built,
            "features_failed": features_failed,
        }
