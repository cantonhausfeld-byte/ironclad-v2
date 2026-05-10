"""Ingest Next Gen Stats (NGS) tracking data via nfl_data_py."""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_STAT_TYPES = ["passing", "rushing", "receiving"]

# Columns we want per stat type, plus the stat_type discriminator
_PASSING_COLS = [
    "player_id", "player_display_name", "season", "week",
    "avg_time_to_throw", "avg_completed_air_yards", "avg_intended_air_yards",
    "completion_percentage_above_expectation",
]
_RUSHING_COLS = [
    "player_id", "player_display_name", "season", "week",
    "efficiency", "rush_yards_over_expected",
]
_RECEIVING_COLS = [
    "player_id", "player_display_name", "season", "week",
    "avg_separation", "catch_percentage_above_expectation",
    "avg_intended_air_yards",
]

_TYPE_COLS: dict[str, list[str]] = {
    "passing":   _PASSING_COLS,
    "rushing":   _RUSHING_COLS,
    "receiving": _RECEIVING_COLS,
}

# Output schema columns (matching bronze.ngs_data)
_OUT_COLS = [
    "player_id", "player_name", "season", "week", "stat_type",
    "avg_time_to_throw", "avg_completed_air_yards", "avg_intended_air_yards",
    "completion_percentage_above_expectation",
    "efficiency", "rush_yards_over_expected",
    "avg_separation", "catch_percentage_above_expectation",
]


class NGSDataIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        import nfl_data_py as nfl
        total = 0
        for stat_type in _STAT_TYPES:
            try:
                raw = nfl.import_ngs_data(stat_type=stat_type, years=seasons)
                df = _clean(raw, stat_type)
                if not df.empty:
                    n = self._writer.write_ngs_data(df)
                    total += n
                    logger.info("NGS %s: wrote %d rows for seasons %s", stat_type, n, seasons)
            except Exception as exc:
                logger.warning("NGS ingest failed for stat_type=%s: %s", stat_type, exc)
        return total


def _clean(raw: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        return pd.DataFrame(columns=_OUT_COLS)

    # Normalize player_id column name variations
    if "player_gsis_id" in raw.columns and "player_id" not in raw.columns:
        raw = raw.rename(columns={"player_gsis_id": "player_id"})

    # Normalize display name
    if "player_display_name" in raw.columns and "player_name" not in raw.columns:
        raw = raw.rename(columns={"player_display_name": "player_name"})
    elif "player_name" not in raw.columns:
        raw["player_name"] = None

    # Filter to regular season only if column exists
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"].str.upper() == "REG"]

    raw = raw.copy()
    raw["stat_type"] = stat_type

    df = _safe_select(raw, _OUT_COLS)
    df = df.dropna(subset=["player_id", "season", "week"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season", "week"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    for col in _OUT_COLS:
        if col not in ("player_id", "player_name", "season", "week", "stat_type"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
