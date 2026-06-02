"""Orchestrates all ingestors for a full ingest run."""
from __future__ import annotations

import logging

from ironclad.ingest.depth_charts import DepthChartIngestor
from ironclad.ingest.ftn_charting import FTNChartingIngestor
from ironclad.ingest.injuries import InjuryIngestor
from ironclad.ingest.ngs_stats import NGSPassingIngestor, NGSReceivingIngestor, NGSRushingIngestor
from ironclad.ingest.pfr_pressure import PFRPressureIngestor
from ironclad.ingest.play_by_play import PBPIngestor
from ironclad.ingest.player_ids import PlayerIDIngestor
from ironclad.ingest.player_stats import PlayerStatsIngestor
from ironclad.ingest.rosters import RosterIngestor
from ironclad.ingest.schedules import ScheduleIngestor
from ironclad.ingest.snap_counts import SnapCountIngestor
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
        self._snap_counts = SnapCountIngestor(writer)
        self._player_stats = PlayerStatsIngestor(writer)
        self._ngs_receiving = NGSReceivingIngestor(writer)
        self._ngs_passing = NGSPassingIngestor(writer)
        self._ngs_rushing = NGSRushingIngestor(writer)
        self._player_ids = PlayerIDIngestor(writer)
        self._pfr_pressure = PFRPressureIngestor(writer)
        self._ftn_charting = FTNChartingIngestor(writer)
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

        logger.info("=== Ingesting player ID crosswalk ===")
        try:
            counts["player_ids"] = self._player_ids.ingest()
        except Exception as exc:
            logger.warning("Player ID mapping ingest failed (non-fatal): %s", exc)
            counts["player_ids"] = 0

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

        logger.info("=== Ingesting snap counts (nflverse) ===")
        try:
            counts["snap_counts"] = self._snap_counts.ingest(seasons)
        except Exception as exc:
            logger.warning("Snap count ingest failed (non-fatal): %s", exc)
            counts["snap_counts"] = 0

        logger.info("=== Ingesting player stats (nflverse) ===")
        try:
            counts["player_stats"] = self._player_stats.ingest(seasons)
        except Exception as exc:
            logger.warning("Player stats ingest failed (non-fatal): %s", exc)
            counts["player_stats"] = 0

        logger.info("=== Ingesting NGS receiving stats ===")
        try:
            counts["ngs_receiving"] = self._ngs_receiving.ingest(seasons)
        except Exception as exc:
            logger.warning("NGS receiving ingest failed (non-fatal): %s", exc)
            counts["ngs_receiving"] = 0

        logger.info("=== Ingesting NGS passing stats ===")
        try:
            counts["ngs_passing"] = self._ngs_passing.ingest(seasons)
        except Exception as exc:
            logger.warning("NGS passing ingest failed (non-fatal): %s", exc)
            counts["ngs_passing"] = 0

        logger.info("=== Ingesting NGS rushing stats ===")
        try:
            counts["ngs_rushing"] = self._ngs_rushing.ingest(seasons)
        except Exception as exc:
            logger.warning("NGS rushing ingest failed (non-fatal): %s", exc)
            counts["ngs_rushing"] = 0

        logger.info("=== Ingesting PFR pressure stats ===")
        try:
            counts["pfr_pressure"] = self._pfr_pressure.ingest(seasons)
        except Exception as exc:
            logger.warning("PFR pressure ingest failed (non-fatal): %s", exc)
            counts["pfr_pressure"] = 0

        logger.info("=== Ingesting FTN charting data ===")
        try:
            counts["ftn_charting"] = self._ftn_charting.ingest(seasons)
        except Exception as exc:
            logger.warning("FTN charting ingest failed (non-fatal): %s", exc)
            counts["ftn_charting"] = 0

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
