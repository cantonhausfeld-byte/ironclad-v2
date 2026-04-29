"""Ingest weekly rosters via nfl_data_py."""
from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_KEEP = [
    "season", "week", "player_id", "player_name", "team", "position",
    "depth_chart_pos", "jersey_number", "status",
    "height", "weight", "years_exp",
]


class RosterIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        logger.info("Fetching weekly rosters for seasons %s", seasons)
        frames = []
        for season in seasons:
            try:
                raw = nfl.import_weekly_rosters([season])
                frames.append(_clean(raw))
            except Exception as exc:
                logger.warning("Roster fetch failed for %d: %s", season, exc)
        if not frames:
            return 0
        df = pd.concat(frames, ignore_index=True)
        n = self._writer.write_rosters(df)
        logger.info("Wrote %d roster rows", n)
        return n


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    # Normalise depth_chart_position → depth_chart_pos regardless of source name
    if "depth_chart_position" in raw.columns and "depth_chart_pos" not in raw.columns:
        raw = raw.rename(columns={"depth_chart_position": "depth_chart_pos"})
    df = _safe_select(raw, _KEEP)
    df = df.dropna(subset=["player_id", "team", "position"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].fillna(0).astype(int)
    return df
