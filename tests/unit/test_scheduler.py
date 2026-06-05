"""Unit tests for the weekly scheduler logic."""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from ironclad.workflow.scheduler import (
    WeeklyScheduler,
    _iso_week_key,
    detect_nfl_season,
)


# ── detect_nfl_season ─────────────────────────────────────────────────────────

def test_detect_nfl_season_regular():
    assert detect_nfl_season(date(2025, 9, 7)) == 2025
    assert detect_nfl_season(date(2025, 12, 25)) == 2025
    assert detect_nfl_season(date(2026, 3, 1)) == 2026


def test_detect_nfl_season_jan_feb_belongs_to_prior_year():
    # January and February belong to the previous calendar year's season
    assert detect_nfl_season(date(2026, 1, 15)) == 2025
    assert detect_nfl_season(date(2026, 2, 8)) == 2025  # Super Bowl weekend
    assert detect_nfl_season(date(2026, 2, 28)) == 2025


# ── _iso_week_key ─────────────────────────────────────────────────────────────

def test_iso_week_key_format():
    key = _iso_week_key(date(2025, 9, 10))
    assert key.startswith("2025-W")
    assert len(key) == len("2025-W37")


def test_iso_week_key_same_week():
    # Two dates in the same ISO week produce the same key
    assert _iso_week_key(date(2025, 9, 8)) == _iso_week_key(date(2025, 9, 12))


# ── WeeklyScheduler._tick() ───────────────────────────────────────────────────

def _make_scheduler(**kwargs) -> WeeklyScheduler:
    defaults = {"n_draws": 100, "dry_run": True}
    defaults.update(kwargs)
    return WeeklyScheduler(**defaults)


def test_pre_game_fires_on_wednesday(tmp_path, monkeypatch):
    """Pre-game job runs on Wednesday >= 10am when upcoming game exists."""
    monkeypatch.setattr(
        "ironclad.workflow.scheduler._STATE_FILE",
        tmp_path / "state.json",
    )
    wednesday_10am = datetime(2025, 9, 10, 10, 0)  # Wednesday

    with (
        patch("ironclad.workflow.scheduler.detect_upcoming_week",
              return_value=(2025, 1)),
        patch("ironclad.workflow.scheduler.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = wednesday_10am
        result = _make_scheduler().run_once()

    assert "pre_game" not in result  # dry_run=True skips actual run
    # State should record the pre-game as done
    import json
    state = json.loads((tmp_path / "state.json").read_text())
    assert any(k.startswith("pre_game_") and v == "done" for k, v in state.items())


def test_pre_game_does_not_fire_before_hour(tmp_path, monkeypatch):
    """Pre-game job does NOT run on Wednesday before 10am."""
    monkeypatch.setattr(
        "ironclad.workflow.scheduler._STATE_FILE",
        tmp_path / "state.json",
    )
    wednesday_9am = datetime(2025, 9, 10, 9, 0)

    with (
        patch("ironclad.workflow.scheduler.detect_upcoming_week",
              return_value=(2025, 1)),
        patch("ironclad.workflow.scheduler.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = wednesday_9am
        result = _make_scheduler().run_once()

    assert result == {}


def test_pre_game_does_not_double_fire(tmp_path, monkeypatch):
    """Pre-game job does not re-run if already done this week."""
    import json

    state_file = tmp_path / "state.json"
    monkeypatch.setattr("ironclad.workflow.scheduler._STATE_FILE", state_file)

    wednesday_10am = datetime(2025, 9, 10, 10, 0)
    week_key = _iso_week_key(wednesday_10am.date())
    state_file.write_text(json.dumps({f"pre_game_{week_key}": "done"}))

    with (
        patch("ironclad.workflow.scheduler.detect_upcoming_week",
              return_value=(2025, 1)) as mock_upcom,
        patch("ironclad.workflow.scheduler.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = wednesday_10am
        _make_scheduler().run_once()

    mock_upcom.assert_not_called()


def test_post_game_fires_on_tuesday(tmp_path, monkeypatch):
    """Post-game backfill runs on Tuesday >= 6am when completed game exists."""
    monkeypatch.setattr(
        "ironclad.workflow.scheduler._STATE_FILE",
        tmp_path / "state.json",
    )
    tuesday_6am = datetime(2025, 9, 9, 6, 0)  # Tuesday

    with (
        patch("ironclad.workflow.scheduler.detect_completed_week",
              return_value=(2025, 1)),
        patch("ironclad.workflow.scheduler.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = tuesday_6am
        result = _make_scheduler().run_once()

    import json
    state = json.loads((tmp_path / "state.json").read_text())
    assert any(k.startswith("post_game_") and v == "done" for k, v in state.items())


def test_end_of_season_retrain_fires_in_february(tmp_path, monkeypatch):
    """Retrain fires in February (>= day 10) when no upcoming games."""
    monkeypatch.setattr(
        "ironclad.workflow.scheduler._STATE_FILE",
        tmp_path / "state.json",
    )
    feb_15 = datetime(2026, 2, 15, 10, 0)

    with (
        patch("ironclad.workflow.scheduler.detect_upcoming_week", return_value=None),
        patch("ironclad.workflow.scheduler.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = feb_15
        result = _make_scheduler().run_once()

    import json
    state = json.loads((tmp_path / "state.json").read_text())
    assert any(k.startswith("retrain_") and v == "done" for k, v in state.items())


def test_end_of_season_retrain_skipped_if_games_upcoming(tmp_path, monkeypatch):
    """Retrain is skipped if upcoming REG games are still detected (season ongoing)."""
    monkeypatch.setattr(
        "ironclad.workflow.scheduler._STATE_FILE",
        tmp_path / "state.json",
    )
    feb_15 = datetime(2026, 2, 15, 10, 0)

    with (
        patch("ironclad.workflow.scheduler.detect_upcoming_week",
              return_value=(2025, 22)),
        patch("ironclad.workflow.scheduler.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = feb_15
        _make_scheduler().run_once()

    # State file should not have a retrain entry
    import json
    if (tmp_path / "state.json").exists():
        state = json.loads((tmp_path / "state.json").read_text())
        assert not any(k.startswith("retrain_") for k in state)


def test_end_of_season_retrain_not_before_day_10(tmp_path, monkeypatch):
    """Retrain does not fire before February 10th."""
    monkeypatch.setattr(
        "ironclad.workflow.scheduler._STATE_FILE",
        tmp_path / "state.json",
    )
    feb_5 = datetime(2026, 2, 5, 10, 0)

    with (
        patch("ironclad.workflow.scheduler.detect_upcoming_week", return_value=None),
        patch("ironclad.workflow.scheduler.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = feb_5
        _make_scheduler().run_once()

    assert not (tmp_path / "state.json").exists()
