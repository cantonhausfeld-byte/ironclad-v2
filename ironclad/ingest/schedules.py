"""Ingest NFL schedules via nfl_data_py."""
from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

from ironclad.ingest.base import BaseIngestor
from ironclad.store.normalization import normalize_teams
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_KEEP = [
    "game_id", "season", "game_type", "week", "gameday", "gametime",
    "away_team", "home_team", "away_score", "home_score",
    "stadium_id", "roof", "surface", "temp", "wind",
    "spread_line", "total_line", "away_moneyline", "home_moneyline",
    "div_game", "overtime",
]


class ScheduleIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        logger.info("Fetching schedules for seasons %s", seasons)
        raw = nfl.import_schedules(seasons)
        df = _clean(raw)
        n = self._writer.write_schedules(df)
        logger.info("Wrote %d schedule rows", n)
        return n


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _KEEP if c in raw.columns]
    df = raw[cols].copy()
    df = df.rename(columns={"game_type": "season_type"})
    df["season_type"] = df["season_type"].fillna("REG")
    df["gameday"] = pd.to_datetime(df["gameday"]).dt.date
    df["game_id"] = df["game_id"].astype(str)
    df = df.dropna(subset=["game_id", "away_team", "home_team"])
    df["home_team"] = normalize_teams(df["home_team"])
    df["away_team"] = normalize_teams(df["away_team"])
    # Ensure required columns exist
    for col in ["season_type", "week", "gameday"]:
        if col not in df.columns:
            df[col] = None
    return df
