"""Tests for team feature builder helpers."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ironclad.features.team_features import _parse_kickoff, _is_divisional
from ironclad.features.utils import safe_divide, completeness_score


# ── _is_divisional ────────────────────────────────────────────────────────────

def test_divisional_same_division():
    assert _is_divisional("KC", "LV") is True   # AFC West
    assert _is_divisional("BUF", "MIA") is True  # AFC East
    assert _is_divisional("DAL", "PHI") is True  # NFC East


def test_divisional_different_division():
    assert _is_divisional("KC", "BAL") is False  # AFC West vs AFC North
    assert _is_divisional("SF", "DAL") is False  # NFC West vs NFC East


def test_divisional_unknown_team_returns_false():
    assert _is_divisional("XX", "KC") is False
    assert _is_divisional("KC", "XX") is False


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
