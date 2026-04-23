"""Ingest weekly snap counts via nflreadpy (nflverse)."""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.ingest.base import BaseIngestor, _safe_select
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

_KEEP = [
    "season", "week", "game_id", "player_id", "player_name",
    "team", "position",
    "offense_snaps", "offense_pct",
    "defense_snaps", "defense_pct",
    "st_snaps", "st_pct",
]


class SnapCountIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()
        self._id_map: pd.DataFrame | None = None

    def _ingest(self, seasons: list[int]) -> int:
        import nflreadpy as nfl
        logger.info("Fetching snap counts for seasons %s (nflverse)", seasons)

        # Build pfr_id → gsis_id crosswalk once per ingest run
        if self._id_map is None:
            self._id_map = _build_id_map(nfl)

        total = 0
        for season in seasons:
            try:
                raw = nfl.load_snap_counts(season).to_pandas()
                df = _clean(raw, self._id_map)
                if not df.empty:
                    total += self._writer.write_snap_counts(df)
                    logger.info("Season %d: wrote %d snap count rows", season, len(df))
            except Exception as exc:
                logger.warning("Snap counts failed for season %d: %s", season, exc)
        return total


def _build_id_map(nfl) -> pd.DataFrame:
    """Return DataFrame with pfr_id → gsis_id mapping."""
    try:
        players = nfl.load_players().to_pandas()
        return players[["gsis_id", "pfr_id"]].dropna(subset=["pfr_id", "gsis_id"])
    except Exception as exc:
        logger.warning("Could not load player ID crosswalk: %s", exc)
        return pd.DataFrame(columns=["gsis_id", "pfr_id"])


def _clean(raw: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    # nflverse snap_counts uses 'player' for name and 'pfr_player_id' for ID
    if "pfr_player_id" in raw.columns and "player_id" not in raw.columns:
        raw = raw.rename(columns={"pfr_player_id": "pfr_id"})
        # Resolve to gsis_id via crosswalk
        if not id_map.empty:
            raw = raw.merge(id_map, on="pfr_id", how="left")
            raw["player_id"] = raw["gsis_id"].fillna(raw["pfr_id"])
        else:
            raw["player_id"] = raw["pfr_id"]
    if "player" in raw.columns and "player_name" not in raw.columns:
        raw = raw.rename(columns={"player": "player_name"})

    df = _safe_select(raw, _KEEP)
    df = df.dropna(subset=["player_id", "team", "season", "week"])
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    for col in ["offense_snaps", "defense_snaps", "st_snaps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ["offense_pct", "defense_pct", "st_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
