"""Tests for team name normalization."""
import pandas as pd
import pytest

from ironclad.store.normalization import normalize_team, normalize_teams, TEAM_MAP


# ── normalize_team ─────────────────────────────────────────────────────────────

def test_oak_maps_to_lv():
    assert normalize_team("OAK") == "LV"


def test_sd_maps_to_lac():
    assert normalize_team("SD") == "LAC"


def test_stl_maps_to_lar():
    assert normalize_team("STL") == "LAR"


def test_la_maps_to_lar():
    assert normalize_team("LA") == "LAR"


def test_jac_maps_to_jax():
    assert normalize_team("JAC") == "JAX"


def test_current_team_unchanged():
    for team in ["KC", "BAL", "SF", "NE", "DAL", "LAR", "LAC", "LV"]:
        assert normalize_team(team) == team


def test_lowercase_input_normalized():
    assert normalize_team("oak") == "LV"
    assert normalize_team("stl") == "LAR"


def test_none_input_returns_none():
    assert normalize_team(None) is None


def test_empty_string_returns_empty():
    assert normalize_team("") is None or normalize_team("") == ""


def test_unknown_code_returned_uppercased():
    assert normalize_team("xyz") == "XYZ"


def test_whitespace_stripped():
    assert normalize_team(" OAK ") == "LV"


# ── normalize_teams (vectorized) ──────────────────────────────────────────────

def test_normalize_teams_series():
    s = pd.Series(["OAK", "SD", "STL", "KC", None])
    result = normalize_teams(s)
    assert result.iloc[0] == "LV"
    assert result.iloc[1] == "LAC"
    assert result.iloc[2] == "LAR"
    assert result.iloc[3] == "KC"
    assert pd.isna(result.iloc[4])


def test_normalize_teams_all_current():
    teams = ["KC", "BAL", "SF", "NE", "DAL"]
    s = pd.Series(teams)
    result = normalize_teams(s)
    assert list(result) == teams


def test_normalize_teams_empty_series():
    result = normalize_teams(pd.Series([], dtype=str))
    assert len(result) == 0


# ── TEAM_MAP completeness ─────────────────────────────────────────────────────

def test_team_map_values_are_current():
    """All mapped-to values should not themselves be in TEAM_MAP keys."""
    for src, dst in TEAM_MAP.items():
        assert dst not in TEAM_MAP, (
            f"TEAM_MAP maps {src}→{dst} but {dst} is also a key — "
            "normalization would not be idempotent"
        )


def test_normalize_is_idempotent():
    """Applying normalize_team twice yields the same result as once."""
    for src in list(TEAM_MAP.keys()) + ["KC", "BAL", "SF"]:
        once = normalize_team(src)
        twice = normalize_team(once)
        assert once == twice, f"normalize_team not idempotent for {src!r}"
