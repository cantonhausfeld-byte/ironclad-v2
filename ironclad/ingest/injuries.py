"""Ingest injury reports via nfl_data_py."""
from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_RAW_RENAMES = {
    "gsis_id": "player_id",
    "full_name": "player_name",
    "primary_injury": "injury_type",
    "report_primary_injury": "injury_type",
    "date_modified": "report_date",
}

_KEEP = [
    "season", "week", "player_id", "player_name", "team", "position",
    "report_status", "practice_status", "injury_type", "report_date",
]


class InjuryIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        logger.info("Fetching injuries for seasons %s", seasons)
        raw = nfl.import_injuries(seasons)
        df = _clean(raw)
        n = self._writer.write_injuries(df)
        logger.info("Wrote %d injury rows", n)
        return n


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    for src, dst in _RAW_RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})
    df = _safe_select(raw, _KEEP)
    df = df.dropna(subset=["player_id", "team"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].fillna(0).astype(int)
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce").dt.date
    return df
