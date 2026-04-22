"""Orchestrates all ingestors for a full ingest run."""
from __future__ import annotations

import logging

from ironclad.ingest.schedules import ScheduleIngestor
from ironclad.ingest.play_by_play import PBPIngestor
from ironclad.ingest.rosters import RosterIngestor
from ironclad.ingest.injuries import InjuryIngestor
from ironclad.ingest.depth_charts import DepthChartIngestor
from ironclad.ingest.stadiums import StadiumIngestor
from ironclad.ingest.weather import WeatherIngestor
from ironclad.store.connection import get_connection
from ironclad.store.schema import create_all_tables
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)


class IngestPipeline:
    def __init__(self) -> None:
        conn = get_connection()
        create_all_tables(conn)
        writer = BronzeWriter(conn)
        self._schedules = ScheduleIngestor(writer)
        self._pbp = PBPIngestor(writer)
        self._rosters = RosterIngestor(writer)
        self._injuries = InjuryIngestor(writer)
        self._depth = DepthChartIngestor(writer)
        self._stadiums = StadiumIngestor(writer)
        self._weather = WeatherIngestor(writer)
        self._conn = conn

    def run(
        self,
        seasons: list[int],
        include_pbp: bool = True,
        include_weather: bool = True,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}

        logger.info("=== Ingesting stadiums ===")
        counts["stadiums"] = self._stadiums.ingest()

        logger.info("=== Ingesting schedules for %s ===", seasons)
        counts["schedules"] = self._schedules.ingest(seasons)

        if include_pbp:
            logger.info("=== Ingesting play-by-play ===")
            counts["play_by_play"] = self._pbp.ingest(seasons)

        logger.info("=== Ingesting rosters ===")
        counts["rosters"] = self._rosters.ingest(seasons)

        logger.info("=== Ingesting injuries ===")
        counts["injuries"] = self._injuries.ingest(seasons)

        logger.info("=== Ingesting depth charts ===")
        counts["depth_charts"] = self._depth.ingest(seasons)

        if include_weather:
            logger.info("=== Ingesting weather ===")
            counts["weather"] = self._ingest_weather()

        logger.info("Ingest complete: %s", counts)
        return counts

    def _ingest_weather(self) -> int:
        # Pull games with stadium coordinates to fetch weather
        try:
            df = self._conn.execute("""
                SELECT g.game_id, g.gameday, s.lat, s.lon
                FROM bronze.schedules g
                LEFT JOIN bronze.stadiums s ON g.home_team = s.team
                WHERE s.lat IS NOT NULL AND s.is_dome = false
                  AND g.gameday IS NOT NULL
            """).df()
            if df.empty:
                return 0
            return self._weather.ingest(df)
        except Exception as exc:
            logger.warning("Weather ingest failed: %s", exc)
            return 0
