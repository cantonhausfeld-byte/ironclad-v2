"""Ingest NFL Next Gen Stats (NGS) receiving and passing data via nflreadpy."""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_RECEIVING_KEEP = [
    "season", "week", "player_id", "player_name", "position", "team",
    "season_type", "avg_separation", "avg_cushion", "avg_intended_air_yards",
    "avg_yac", "avg_yac_above_expectation", "targets", "receptions",
]

_PASSING_KEEP = [
    "season", "week", "player_id", "player_name", "team", "season_type",
    "avg_time_to_throw", "avg_intended_air_yards", "aggressiveness",
    "completion_percentage_above_expectation", "attempts", "completions",
]

_RECEIVING_RENAMES = {
    "player_gsis_id": "player_id",
    "player_display_name": "player_name",
    "player_position": "position",
    "team_abbr": "team",
}

_PASSING_RENAMES = {
    "player_gsis_id": "player_id",
    "player_display_name": "player_name",
    "team_abbr": "team",
}


class NGSReceivingIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        import nflreadpy as nfl
        logger.info("Fetching NGS receiving stats for seasons %s", seasons)
        total = 0
        for season in seasons:
            try:
                raw = nfl.load_nextgen_stats(seasons=season, stat_type="receiving").to_pandas()
                df = _clean_receiving(raw)
                if not df.empty:
                    total += self._writer.write_ngs_receiving(df)
                    logger.info("NGS receiving season %d: wrote %d rows", season, len(df))
            except Exception as exc:
                logger.warning("NGS receiving failed for season %d: %s", season, exc)
        return total


class NGSPassingIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        import nflreadpy as nfl
        logger.info("Fetching NGS passing stats for seasons %s", seasons)
        total = 0
        for season in seasons:
            try:
                raw = nfl.load_nextgen_stats(seasons=season, stat_type="passing").to_pandas()
                df = _clean_passing(raw)
                if not df.empty:
                    total += self._writer.write_ngs_passing(df)
                    logger.info("NGS passing season %d: wrote %d rows", season, len(df))
            except Exception as exc:
                logger.warning("NGS passing failed for season %d: %s", season, exc)
        return total


def _clean_receiving(raw: pd.DataFrame) -> pd.DataFrame:
    for src, dst in _RECEIVING_RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"] == "REG"]
    df = _safe_select(raw, _RECEIVING_KEEP)
    df = df.dropna(subset=["player_id", "season", "week"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    int_cols = ["targets", "receptions"]
    float_cols = [
        "avg_separation", "avg_cushion", "avg_intended_air_yards",
        "avg_yac", "avg_yac_above_expectation",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clean_passing(raw: pd.DataFrame) -> pd.DataFrame:
    for src, dst in _PASSING_RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"] == "REG"]
    df = _safe_select(raw, _PASSING_KEEP)
    df = df.dropna(subset=["player_id", "season", "week"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    int_cols = ["attempts", "completions"]
    float_cols = [
        "avg_time_to_throw", "avg_intended_air_yards", "aggressiveness",
        "completion_percentage_above_expectation",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
