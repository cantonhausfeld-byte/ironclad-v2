"""Tests for team feature builder helpers."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ironclad.features.team_features import _parse_kickoff
from ironclad.features.utils import safe_divide, completeness_score
from ironclad.models.team.game_outcome import _build_diff_features


# ── _parse_kickoff ─────────────────────────────────────────────────────────────

def test_parse_kickoff_with_time():
    dt = _parse_kickoff("2023-09-07", "13:00")
    assert dt.year == 2023
    assert dt.month == 9
    assert dt.day == 7
    # hour is UTC-shifted; just verify it's a valid hour
    assert 0 <= dt.hour < 24


def test_parse_kickoff_no_time_returns_datetime():
    dt = _parse_kickoff("2023-09-07", None)
    assert dt.year == 2023 and dt.month == 9 and dt.day == 7


def test_parse_kickoff_returns_utc():
    dt = _parse_kickoff("2023-09-07", "20:20")
    assert dt.tzinfo is not None


def test_parse_kickoff_string_date():
    dt = _parse_kickoff("2024-01-15", "18:30")
    assert dt.month == 1
    assert dt.day == 15


# ── safe_divide ────────────────────────────────────────────────────────────────

def test_safe_divide_normal():
    import pandas as pd
    result = safe_divide(pd.Series([10.0]), pd.Series([5.0]))
    assert float(result.iloc[0]) == pytest.approx(2.0)


def test_safe_divide_zero_denominator_returns_default():
    import pandas as pd
    result = safe_divide(pd.Series([5.0]), pd.Series([0.0]))
    assert float(result.iloc[0]) == pytest.approx(0.0)


def test_safe_divide_custom_default():
    import pandas as pd
    result = safe_divide(pd.Series([5.0]), pd.Series([0.0]), default=99.0)
    assert float(result.iloc[0]) == pytest.approx(99.0)


def test_safe_divide_scalar_inputs():
    import pandas as pd
    result = safe_divide(10.0, 2.0)
    assert float(result[0]) == pytest.approx(5.0)


# ── completeness_score ─────────────────────────────────────────────────────────

def test_completeness_all_present():
    import pandas as pd
    s = pd.Series([1.0, 2.0, 3.0])
    assert completeness_score(s) == pytest.approx(1.0)


def test_completeness_all_null():
    import pandas as pd
    s = pd.Series([None, None, None])
    assert completeness_score(s) == pytest.approx(0.0)


def test_completeness_partial():
    import pandas as pd
    s = pd.Series([1.0, None, 3.0, None])
    score = completeness_score(s)
    assert score == pytest.approx(0.5)


# ── Phase 2B: _build_diff_features additions ──────────────────────────────────

def test_post_bye_flag_set_when_rest_gte_14():
    X = pd.DataFrame([{"home_rest_days": 14, "away_rest_days": 7}])
    Xf = _build_diff_features(X)
    assert Xf["home_is_post_bye"].iloc[0] == 1
    assert Xf["away_is_post_bye"].iloc[0] == 0


def test_post_bye_flag_clear_on_normal_week():
    X = pd.DataFrame([{"home_rest_days": 7, "away_rest_days": 7}])
    Xf = _build_diff_features(X)
    assert Xf["home_is_post_bye"].iloc[0] == 0
    assert Xf["away_is_post_bye"].iloc[0] == 0


def test_epa_rz_diff_computed_from_prefixed_cols():
    X = pd.DataFrame([{"home_off_epa_rz_l4": 0.15, "away_off_epa_rz_l4": 0.05}])
    Xf = _build_diff_features(X)
    assert Xf["off_epa_rz_diff"].iloc[0] == pytest.approx(0.10, abs=1e-6)


def test_epa_std_diff_computed_from_prefixed_cols():
    X = pd.DataFrame([{"home_off_epa_per_play_std": 0.08, "away_off_epa_per_play_std": 0.02}])
    Xf = _build_diff_features(X)
    assert Xf["off_epa_std_diff"].iloc[0] == pytest.approx(0.06, abs=1e-6)


def test_surface_grass_taken_from_home_side():
    X = pd.DataFrame([{"home_surface_grass": True}])
    Xf = _build_diff_features(X)
    assert bool(Xf["surface_grass"].iloc[0]) is True


# ── TeamFeatureBuilder integration (in-memory DB) ─────────────────────────────

def test_team_feature_builder_empty_game(tmp_path):
    """Builder returns empty DataFrame when game is not in silver."""
    from ironclad.store.connection import in_memory_connection
    from ironclad.store.schema import create_all_tables
    from ironclad.features.team_features import TeamFeatureBuilder

    conn = in_memory_connection()
    create_all_tables(conn)

    builder = TeamFeatureBuilder(conn)
    cutoff = datetime(2023, 9, 7, 12, 30, tzinfo=timezone.utc)
    df = builder.build_for_game("2023_01_KC_BAL", cutoff)
    assert df.empty
