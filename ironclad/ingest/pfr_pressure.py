"""Ingest PFR advanced passing stats (pressure / blitz rates) via nflreadpy."""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_KEEP = [
    "season", "week", "game_id", "team",
    "times_pressured", "times_pressured_pct",
    "times_blitzed", "times_hurried", "times_hit", "times_sacked",
]

_RENAMES = {
    "pfr_game_id": "game_id",
}


class PFRPressureIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        import nflreadpy as nfl
        logger.info("Fetching PFR pressure stats for seasons %s", seasons)
        total = 0
        for season in seasons:
            try:
                raw = nfl.load_pfr_advstats(seasons=[season], stat_type="pass")
                if hasattr(raw, "to_pandas"):
                    raw = raw.to_pandas()
                df = _clean_pfr_pressure(raw)
                if not df.empty:
                    total += self._writer.write_pfr_pressure_weekly(df)
                    logger.info("PFR pressure season %d: wrote %d rows", season, len(df))
            except Exception as exc:
                logger.warning("PFR pressure failed for season %d: %s", season, exc)
        return total


def _clean_pfr_pressure(raw: pd.DataFrame) -> pd.DataFrame:
    for src, dst in _RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})

    # Filter to regular season only
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"] == "REG"]

    df = _safe_select(raw, _KEEP)
    df = df.dropna(subset=["season", "week", "team"])
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)

    int_cols = ["times_pressured", "times_blitzed", "times_hurried", "times_hit", "times_sacked"]
    float_cols = ["times_pressured_pct"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)
