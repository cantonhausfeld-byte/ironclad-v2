"""Smoke tests for the weekly workflow structure."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ironclad.workflow.weekly import WeeklyWorkflow


def _empty_games():
    return pd.DataFrame(columns=["game_id", "gameday", "gametime_local", "home_team", "away_team"])


def _sample_games():
    return pd.DataFrame([{
        "game_id": "2024_05_KC_BAL",
        "gameday": date(2024, 10, 13),
        "gametime_local": "20:20",
        "home_team": "BAL",
        "away_team": "KC",
    }])


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    conn.execute.return_value.df.return_value = _empty_games()
    return conn


def test_weekly_returns_dict_with_expected_keys():
    """WeeklyWorkflow.run() returns a dict with the four expected keys."""
    with patch("ironclad.workflow.weekly.get_connection") as mock_get_conn, \
         patch("ironclad.workflow.weekly.create_all_tables"), \
         patch("ironclad.workflow.weekly.IngestPipeline") as MockPipeline, \
         patch("ironclad.workflow.weekly.SilverTransformer") as MockSilver, \
         patch("ironclad.workflow.weekly.TeamFeatureBuilder"), \
         patch("ironclad.workflow.weekly.PlayerFeatureBuilder"), \
         patch("ironclad.workflow.weekly.TargetBackfiller") as MockTarget:

        conn = MagicMock()
        conn.execute.return_value.df.return_value = _empty_games()
        mock_get_conn.return_value = conn

        MockPipeline.return_value.run.return_value = {"schedules": 10}
        MockSilver.return_value.run.return_value = {"games": 10}
        MockTarget.return_value.run.return_value = {"team": 0, "player": 0}

        result = WeeklyWorkflow().run(2024, 5)

    assert "ingest" in result
    assert "silver" in result
    assert "features_built" in result
    assert "targets" in result


def test_weekly_builds_features_for_games():
    """When there are games in silver, feature builders are called once per game."""
    with patch("ironclad.workflow.weekly.get_connection") as mock_get_conn, \
         patch("ironclad.workflow.weekly.create_all_tables"), \
         patch("ironclad.workflow.weekly.IngestPipeline") as MockPipeline, \
         patch("ironclad.workflow.weekly.SilverTransformer") as MockSilver, \
         patch("ironclad.workflow.weekly.TeamFeatureBuilder") as MockTeam, \
         patch("ironclad.workflow.weekly.PlayerFeatureBuilder") as MockPlayer, \
         patch("ironclad.workflow.weekly.TargetBackfiller") as MockTarget:

        conn = MagicMock()
        conn.execute.return_value.df.return_value = _sample_games()
        mock_get_conn.return_value = conn

        MockPipeline.return_value.run.return_value = {}
        MockSilver.return_value.run.return_value = {}
        MockTarget.return_value.run.return_value = {}

        result = WeeklyWorkflow().run(2024, 5)

    assert result["features_built"] == 1
    MockTeam.return_value.build_for_game.assert_called_once()
    MockPlayer.return_value.build_for_game.assert_called_once()


def test_weekly_calls_target_backfiller():
    """TargetBackfiller.run() is always called at the end of the cycle."""
    with patch("ironclad.workflow.weekly.get_connection") as mock_get_conn, \
         patch("ironclad.workflow.weekly.create_all_tables"), \
         patch("ironclad.workflow.weekly.IngestPipeline") as MockPipeline, \
         patch("ironclad.workflow.weekly.SilverTransformer") as MockSilver, \
         patch("ironclad.workflow.weekly.TeamFeatureBuilder"), \
         patch("ironclad.workflow.weekly.PlayerFeatureBuilder"), \
         patch("ironclad.workflow.weekly.TargetBackfiller") as MockTarget:

        conn = MagicMock()
        conn.execute.return_value.df.return_value = _empty_games()
        mock_get_conn.return_value = conn

        MockPipeline.return_value.run.return_value = {}
        MockSilver.return_value.run.return_value = {}
        MockTarget.return_value.run.return_value = {"team": 5, "player": 20}

        result = WeeklyWorkflow().run(2024, 5)

    MockTarget.return_value.run.assert_called_once_with([2024])
    assert result["targets"] == {"team": 5, "player": 20}


def test_weekly_feature_build_failure_does_not_crash():
    """A feature build exception for one game should not abort the workflow."""
    with patch("ironclad.workflow.weekly.get_connection") as mock_get_conn, \
         patch("ironclad.workflow.weekly.create_all_tables"), \
         patch("ironclad.workflow.weekly.IngestPipeline") as MockPipeline, \
         patch("ironclad.workflow.weekly.SilverTransformer") as MockSilver, \
         patch("ironclad.workflow.weekly.TeamFeatureBuilder") as MockTeam, \
         patch("ironclad.workflow.weekly.PlayerFeatureBuilder"), \
         patch("ironclad.workflow.weekly.TargetBackfiller") as MockTarget:

        conn = MagicMock()
        conn.execute.return_value.df.return_value = _sample_games()
        mock_get_conn.return_value = conn

        MockPipeline.return_value.run.return_value = {}
        MockSilver.return_value.run.return_value = {}
        MockTarget.return_value.run.return_value = {}
        MockTeam.return_value.build_for_game.side_effect = RuntimeError("feature failed")

        # Should not raise
        result = WeeklyWorkflow().run(2024, 5)
    assert result["features_built"] == 0
