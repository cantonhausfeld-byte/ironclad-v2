"""Tests for play-by-play ingestor and silver REG filter."""
from __future__ import annotations

import pandas as pd

from ironclad.ingest.play_by_play import _clean


def _make_raw_pbp(season_types: list[str]) -> pd.DataFrame:
    """Minimal raw PBP DataFrame with the given season_type values."""
    n = len(season_types)
    return pd.DataFrame({
        "play_id":       [str(i) for i in range(n)],
        "game_id":       [f"2024_01_KC_BUF"] * n,
        "season":        [2024] * n,
        "week":          [1] * n,
        "season_type":   season_types,
        "posteam":       ["KC"] * n,
        "defteam":       ["BUF"] * n,
        "home_team":     ["KC"] * n,
        "away_team":     ["BUF"] * n,
        "play_type":     ["pass"] * n,
        "yards_gained":  [5] * n,
        "pass_attempt":  [1] * n,
        "complete_pass": [1] * n,
        "rush_attempt":  [0] * n,
        "epa":           [0.5] * n,
        "down":          [1] * n,
        "ydstogo":       [10] * n,
        "yardline_100":  [50] * n,
    })


def test_clean_filters_reg_only():
    """_clean() keeps only REG season_type rows."""
    raw = _make_raw_pbp(["REG", "POST", "REG", "PRE"])
    df = _clean(raw)
    assert len(df) == 2
    assert all(df["season"].notna())


def test_clean_no_season_type_column():
    """_clean() works when season_type column is absent (older data)."""
    raw = _make_raw_pbp(["REG"])
    raw = raw.drop(columns=["season_type"])
    df = _clean(raw)
    assert len(df) == 1


def test_clean_all_post_returns_empty():
    """_clean() returns empty DataFrame when all rows are POST."""
    raw = _make_raw_pbp(["POST", "POST"])
    df = _clean(raw)
    assert df.empty
