"""Ingest NFL play-by-play data via nfl_data_py."""
from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

from ironclad.ingest.base import BaseIngestor
from ironclad.store.writer import BronzeWriter
from ironclad.store.normalization import normalize_teams

logger = logging.getLogger(__name__)

_KEEP = [
    "play_id", "game_id", "season", "week", "home_team", "away_team",
    "posteam", "defteam", "play_type", "yards_gained",
    "pass_attempt", "complete_pass", "rush_attempt",
    "sack", "touchdown", "interception", "fumble_lost", "penalty",
    "field_goal_attempt", "field_goal_result", "kick_distance",
    "ep", "epa", "wp", "down", "ydstogo", "yardline_100",
    "game_seconds_remaining", "qb_dropback", "qb_scramble",
    "air_yards", "yards_after_catch",
    "passer_player_id", "passer_player_name",
    "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name",
    # Phase 2A additions — situational features
    "cpoe",                  # Completion % over expectation (QB quality signal)
    "xpass",                 # Expected pass probability (coach tendency vs situation)
    "score_differential",    # Current score margin (needed for neutral-script EPA filter)
    "qb_hit",                # QB was hit on the play (pressure proxy)
    "third_down_converted",  # Converted 3rd down
    "third_down_failed",     # Failed 3rd down
    "fourth_down_converted", # Converted 4th down go-for-it
    "fourth_down_failed",    # Failed 4th down go-for-it
]


class PBPIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, seasons: list[int]) -> int:
        total = 0
        for season in seasons:
            logger.info("Fetching PBP for season %d", season)
            raw = nfl.import_pbp_data([season], downcast=True)
            df = _clean(raw)
            n = self._writer.write_play_by_play(df)
            logger.info("Season %d: wrote %d PBP rows", season, n)
            total += n
        return total


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _KEEP if c in raw.columns]
    df = raw[cols].copy()
    df["play_id"] = df["play_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    # Fill integer flags with 0
    int_flags = ["pass_attempt", "complete_pass", "rush_attempt", "sack",
                 "touchdown", "interception", "fumble_lost", "penalty",
                 "field_goal_attempt", "qb_dropback", "qb_scramble"]
    for col in int_flags:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    for col in ["home_team", "away_team", "posteam", "defteam"]:
        if col in df.columns:
            df[col] = normalize_teams(df[col])
    return df
