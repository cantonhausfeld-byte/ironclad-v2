"""Tests for hardened ingestor column handling and _safe_select."""
from __future__ import annotations

import pandas as pd
import pytest

from ironclad.ingest.base import _safe_select
from ironclad.ingest.rosters import _clean as roster_clean
from ironclad.ingest.depth_charts import _clean as depth_clean
from ironclad.ingest.injuries import _clean as injury_clean


# ── _safe_select ──────────────────────────────────────────────────────────────

def test_safe_select_fills_missing_cols():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = _safe_select(df, ["a", "b", "c"])
    assert "c" in result.columns
    assert result["c"].isna().all()


def test_safe_select_preserves_order():
    df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
    result = _safe_select(df, ["a", "m", "z"])
    assert list(result.columns) == ["a", "m", "z"]


def test_safe_select_only_requested_cols():
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [99]})
    result = _safe_select(df, ["a", "b"])
    assert "extra" not in result.columns


def test_safe_select_empty_df():
    df = pd.DataFrame()
    result = _safe_select(df, ["a", "b"])
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 0


# ── RosterIngestor._clean ─────────────────────────────────────────────────────

def _roster_raw(**extra):
    row = {
        "season": 2023, "week": 1, "player_id": "ABC123",
        "player_name": "Joe Smith", "team": "KC", "position": "WR",
        "jersey_number": 11, "status": "ACT",
        "height": "6-1", "weight": 200, "years_exp": 3,
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_roster_clean_with_depth_chart_position():
    raw = _roster_raw(depth_chart_position="WR")
    df = roster_clean(raw)
    assert "depth_chart_pos" in df.columns
    assert df.iloc[0]["depth_chart_pos"] == "WR"


def test_roster_clean_without_any_depth_col():
    raw = _roster_raw()
    df = roster_clean(raw)
    assert "depth_chart_pos" in df.columns  # filled with None via _safe_select


def test_roster_clean_drops_null_player_id():
    raw = _roster_raw()
    raw.loc[0, "player_id"] = None
    df = roster_clean(raw)
    assert df.empty


# ── DepthChartIngestor._clean ─────────────────────────────────────────────────

def _depth_raw(**extra):
    row = {
        "season": 2023, "team": "KC", "week": 1, "depth_team": 1,
        "position": "WR", "player_id": "ABC123", "full_name": "Joe Smith",
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_depth_clean_full_name_becomes_player_name():
    raw = _depth_raw()
    df = depth_clean(raw)
    assert df.iloc[0]["player_name"] == "Joe Smith"


def test_depth_clean_missing_player_name_falls_back_to_empty():
    raw = _depth_raw()
    raw = raw.drop(columns=["full_name"])
    df = depth_clean(raw)
    assert "player_name" in df.columns
    assert df.iloc[0]["player_name"] == ""


def test_depth_clean_club_code_to_team():
    raw = pd.DataFrame([{
        "season": 2023, "club_code": "KC", "week": 1, "depth_team": 1,
        "position": "WR", "player_id": "ABC123", "full_name": "Joe Smith",
    }])
    df = depth_clean(raw)
    assert df.iloc[0]["team"] == "KC"


# ── InjuryIngestor._clean ─────────────────────────────────────────────────────

def _injury_raw(**extra):
    row = {
        "season": 2023, "week": 1, "player_id": "ABC123",
        "full_name": "Joe Smith", "team": "KC", "position": "WR",
        "report_status": "Questionable", "practice_status": "Limited",
        "primary_injury": "Hamstring", "date_modified": "2023-09-06",
    }
    row.update(extra)
    return pd.DataFrame([row])


def test_injury_clean_primary_injury_renamed():
    raw = _injury_raw()
    df = injury_clean(raw)
    assert "injury_type" in df.columns
    assert df.iloc[0]["injury_type"] == "Hamstring"


def test_injury_clean_missing_primary_injury():
    raw = _injury_raw()
    raw = raw.drop(columns=["primary_injury"])
    df = injury_clean(raw)
    assert "injury_type" in df.columns
    assert pd.isna(df.iloc[0]["injury_type"])


def test_injury_clean_full_name_renamed():
    raw = _injury_raw()
    df = injury_clean(raw)
    assert df.iloc[0]["player_name"] == "Joe Smith"
