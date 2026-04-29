"""Ingest depth charts via nfl_data_py."""
from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

# nfl_data_py uses club_code; normalise to team before selecting
_RAW_RENAMES = {"club_code": "team", "gsis_id": "player_id", "full_name": "player_name", "player_display_name": "player_name"}

_KEEP = [
    "season", "team", "week", "depth_team",
    "position", "player_id", "player_name",
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
    # Apply renames only when the source column exists and target doesn't yet
    for src, dst in _RAW_RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})
    df = _safe_select(raw, _KEEP)
    df = df.dropna(subset=["player_id", "team", "position"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].fillna(0).astype(int)
    df["depth_team"] = pd.to_numeric(df["depth_team"], errors="coerce").fillna(99).astype(int)
    if df["player_name"].isna().all():
        df["player_name"] = ""
    else:
        df["player_name"] = df["player_name"].fillna("")
    return df
