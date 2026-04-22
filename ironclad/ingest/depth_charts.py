"""Ingest depth charts via nfl_data_py."""
from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

from ironclad.ingest.base import BaseIngestor
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_KEEP = [
    "season", "club_code", "week", "game_type", "depth_team",
    "position", "player_id", "full_name",
]


class DepthChartIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        logger.info("Fetching depth charts for seasons %s", seasons)
        raw = nfl.import_depth_charts(seasons)
        df = _clean(raw)
        n = self._writer.write_depth_charts(df)
        logger.info("Wrote %d depth chart rows", n)
        return n


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _KEEP if c in raw.columns]
    df = raw[cols].copy()
    df = df.rename(columns={"club_code": "team", "full_name": "player_name"})
    df = df.dropna(subset=["player_id", "team", "position"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].fillna(0).astype(int)
    df["depth_team"] = pd.to_numeric(df["depth_team"], errors="coerce").fillna(99).astype(int)
    if "player_name" not in df.columns:
        df["player_name"] = ""
    return df
