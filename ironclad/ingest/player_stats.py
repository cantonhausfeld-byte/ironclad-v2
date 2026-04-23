"""Ingest pre-aggregated weekly player stats via nflreadpy (nflverse)."""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_RAW_RENAMES = {
    "passing_interceptions": "interceptions",
}

_KEEP = [
    "season", "week", "player_id", "player_name", "team", "position",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share", "wopr",
]


class PlayerStatsIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        import nflreadpy as nfl
        logger.info("Fetching weekly player stats for seasons %s (nflverse)", seasons)
        total = 0
        for season in seasons:
            try:
                raw = nfl.load_player_stats(season, summary_level="week").to_pandas()
                df = _clean(raw)
                if not df.empty:
                    total += self._writer.write_player_stats_weekly(df)
                    logger.info("Season %d: wrote %d player stat rows", season, len(df))
            except Exception as exc:
                logger.warning("Player stats failed for season %d: %s", season, exc)
        return total


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    # Apply column renames before selecting
    for src, dst in _RAW_RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})
    # Only keep regular-season weeks (filter out post-season if present)
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"] == "REG"]
    df = _safe_select(raw, _KEEP)
    df = df.dropna(subset=["player_id", "season", "week"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    int_cols = ["completions", "attempts", "passing_tds", "interceptions",
                "carries", "rushing_tds", "receptions", "targets", "receiving_tds"]
    float_cols = ["passing_yards", "rushing_yards", "receiving_yards",
                  "target_share", "air_yards_share", "wopr"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df
