"""Ingest depth charts via nflreadpy (nflverse)."""
from __future__ import annotations

import logging
from datetime import date

import nflreadpy as nfl
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
        # One season at a time: 2016–2024 and 2025+ use different schemas, and a
        # mixed load leaves the old rows with an empty `team` column.
        total = 0
        for season in seasons:
            logger.info("Fetching depth charts for season %d", season)
            raw = nfl.load_depth_charts([season]).to_pandas()
            if _is_daily_format(raw):
                schedule = nfl.load_schedules([season]).to_pandas()
                df = _clean_daily(raw, season, _week_start_dates(schedule))
            else:
                df = _clean(raw)
            n = self._writer.write_depth_charts(df) if not df.empty else 0
            logger.info("Season %d: wrote %d depth chart rows", season, n)
            total += n
        return total


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


# ── 2025+ format: daily ESPN depth chart snapshots ────────────────────────────
#
# Columns: dt (snapshot UTC timestamp), team, player_name, gsis_id, pos_grp,
# pos_abb, pos_slot, pos_rank. No season/week/depth_team. pos_rank counts
# across every slot of a position (WR3 starter is rank 3), so depth is re-ranked
# within (pos_abb, pos_slot) to match the old per-slot depth_team.

_POSITION_MAP = {
    "PK": "K", "LT": "T", "RT": "T", "LG": "G", "RG": "G",
    "LDE": "DE", "RDE": "DE", "LDT": "DT", "RDT": "DT",
    "WLB": "LB", "SLB": "LB", "MLB": "LB", "LILB": "LB", "RILB": "LB",
    "LCB": "CB", "RCB": "CB", "NB": "CB",
}
_SPECIAL_TEAMS_KEEP = {"K", "P"}


def _is_daily_format(raw: pd.DataFrame) -> bool:
    return {"dt", "pos_abb", "pos_rank"} <= set(raw.columns)


def _week_start_dates(schedule: pd.DataFrame) -> dict[int, date]:
    """First game date of each week in the season (regular season + playoffs)."""
    days = pd.to_datetime(schedule["gameday"]).dt.date
    return days.groupby(schedule["week"].astype(int)).min().to_dict()


def _clean_daily(raw: pd.DataFrame, season: int, week_starts: dict[int, date]) -> pd.DataFrame:
    """Latest snapshot strictly before each week's first game day, per team."""
    empty = pd.DataFrame(columns=_KEEP)
    if raw.empty or not week_starts:
        return empty

    df = raw.dropna(subset=["gsis_id", "team", "pos_abb", "pos_rank"]).copy()
    snap_day = pd.to_datetime(df["dt"], utc=True).dt.date

    weeks = sorted(week_starts)
    starts = [week_starts[w] for w in weeks]

    def _week_for(day: date) -> int | None:
        for w, start in zip(weeks, starts):
            if day < start:
                return w
        return None  # after the season's last game

    df["week"] = snap_day.map(_week_for)
    df = df.dropna(subset=["week"])
    if df.empty:
        return empty
    df["week"] = df["week"].astype(int)

    # Latest snapshot per (team, week)
    latest = df.groupby(["team", "week"])["dt"].transform("max")
    df = df[df["dt"] == latest].copy()

    df["position"] = df["pos_abb"].map(lambda p: _POSITION_MAP.get(p, p))
    is_st = df["pos_grp"].eq("Special Teams") if "pos_grp" in df.columns else False
    df = df[~is_st | df["position"].isin(_SPECIAL_TEAMS_KEEP)].copy()

    slot = df["pos_slot"] if "pos_slot" in df.columns else 0
    df["depth_team"] = (
        df.assign(_slot=slot)
        .groupby(["team", "week", "pos_abb", "_slot"])["pos_rank"]
        .rank(method="first")
        .astype(int)
    )
    # One row per player per week: their best (lowest) depth
    df = df.sort_values("depth_team").drop_duplicates(["team", "week", "gsis_id"], keep="first")

    df = df.rename(columns={"gsis_id": "player_id"})
    df["season"] = season
    df["player_id"] = df["player_id"].astype(str)
    df["player_name"] = df["player_name"].fillna("")
    return df[_KEEP].sort_values(["week", "team", "position", "depth_team"]).reset_index(drop=True)
