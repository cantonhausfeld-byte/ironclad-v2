"""Ingest nfl_data_py player ID crosswalk (gsis_id ↔ pfr_id, espn_id, etc.)."""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_KEEP = [
    "gsis_id", "pfr_id", "nfl_id", "espn_id",
    "player_name", "position", "team",
]

_RENAMES = {
    "name":     "player_name",
    "gsis":     "gsis_id",
    "pfr":      "pfr_id",
    "nfl":      "nfl_id",
    "espn":     "espn_id",
    "position": "position",
    "team":     "team",
}


class PlayerIDIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self) -> int:
        import nfl_data_py as nfl
        logger.info("Fetching player ID crosswalk via nfl_data_py.import_ids()")
        raw = nfl.import_ids()
        df = _clean_player_ids(raw)
        if df.empty:
            logger.warning("Player ID crosswalk returned empty DataFrame")
            return 0
        n = self._writer.write_player_id_mapping(df)
        logger.info("Player ID mapping: wrote %d rows", n)
        return n

    def ingest(self) -> int:  # override — no seasons arg
        return self._ingest()


def _clean_player_ids(raw: pd.DataFrame) -> pd.DataFrame:
    # Apply renames for alternative column naming conventions
    for src, dst in _RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})
    df = _safe_select(raw, _KEEP)
    # Must have a valid gsis_id to be useful
    df = df.dropna(subset=["gsis_id"])
    df["gsis_id"] = df["gsis_id"].astype(str).str.strip()
    df = df[df["gsis_id"] != ""]
    # Deduplicate on primary key (keep latest / longest pfr_id)
    df = df.drop_duplicates(subset=["gsis_id"], keep="last")
    return df.reset_index(drop=True)
