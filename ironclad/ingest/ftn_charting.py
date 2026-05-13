"""Ingest FTN play-level charting data (blitz count, play-action, motion, box count).

Available 2022+ only. Gold features will be NULL before 2022 if not backfilled.
FTN data is play-level; we aggregate per team per game during silver enrichment.
"""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_KEEP = [
    "game_id", "season", "week", "team",
    "n_blitzers", "n_pass_rushers", "n_defense_box",
    "is_play_action", "is_motion",
]

_RENAMES = {
    "nflverse_game_id": "game_id",
}


class FTNChartingIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        import nflreadpy as nfl
        logger.info("Fetching FTN charting data for seasons %s", seasons)
        total = 0
        for season in seasons:
            if season < 2022:
                logger.debug("FTN charting not available before 2022, skipping %d", season)
                continue
            try:
                raw = nfl.load_ftn_charting(seasons=[season])
                if hasattr(raw, "to_pandas"):
                    raw = raw.to_pandas()
                df = _clean_ftn_charting(raw)
                if not df.empty:
                    total += self._writer.write_ftn_charting(df)
                    logger.info("FTN charting season %d: wrote %d rows", season, len(df))
            except Exception as exc:
                logger.warning("FTN charting failed for season %d: %s", season, exc)
        return total


def _clean_ftn_charting(raw: pd.DataFrame) -> pd.DataFrame:
    for src, dst in _RENAMES.items():
        if src in raw.columns and dst not in raw.columns:
            raw = raw.rename(columns={src: dst})

    # Filter to regular season rows when season_type is present
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"].isin(["REG", "reg", None]) |
                  raw["season_type"].isna()]

    df = _safe_select(raw, _KEEP)
    df = df.dropna(subset=["game_id", "season"])

    # Derive team from game_id if not present (FTN may not have explicit team col)
    # We keep a team col as NULL if not available; silver enrichment joins by game_id
    if "team" not in df.columns:
        df["team"] = None

    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)

    bool_cols = ["is_play_action", "is_motion"]
    int_cols = ["n_blitzers", "n_pass_rushers", "n_defense_box"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)
