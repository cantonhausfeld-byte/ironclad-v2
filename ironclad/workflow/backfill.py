"""Backfill historical data: ingest + silver transform."""
from __future__ import annotations

import logging

from ironclad.ingest.pipeline import IngestPipeline
from ironclad.store.silver import SilverTransformer
from ironclad.store.connection import get_connection
from ironclad.store.schema import create_all_tables

logger = logging.getLogger(__name__)


class BackfillWorkflow:
    def run(self, seasons: list[int], include_pbp: bool = True) -> None:
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

        logger.info("Backfill complete for seasons %s", seasons)
