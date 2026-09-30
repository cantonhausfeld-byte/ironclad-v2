"""Depth chart ingest: 2016–2024 weekly format and 2025+ daily snapshot format."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from ironclad.ingest.depth_charts import (
    DepthChartIngestor,
    _clean_daily,
    _is_daily_format,
    _week_start_dates,
)

WEEK_STARTS = {1: date(2026, 9, 10), 2: date(2026, 9, 17)}


def _snap(dt, player, gsis, pos, slot, rank, team="KC", grp="3WR 1TE"):
    return {"dt": dt, "team": team, "player_name": player, "gsis_id": gsis,
            "pos_grp": grp, "pos_abb": pos, "pos_slot": slot, "pos_rank": rank}


def _daily(rows):
    return pd.DataFrame(rows)


def test_detects_daily_format():
    assert _is_daily_format(_daily([_snap("2026-09-01T00:00:00Z", "A", "1", "QB", 9, 1)]))
    assert not _is_daily_format(pd.DataFrame({"season": [2024], "club_code": ["KC"]}))


def test_week_start_dates_uses_first_game_of_week():
    sched = pd.DataFrame({
        "week": [1, 1, 2],
        "gameday": ["2026-09-13", "2026-09-10", "2026-09-17"],
    })
    assert _week_start_dates(sched) == WEEK_STARTS


def test_snapshot_assigned_to_next_week_and_latest_wins():
    df = _clean_daily(_daily([
        _snap("2026-08-20T06:00:00Z", "Old QB", "1", "QB", 9, 1),   # preseason → wk1, superseded
        _snap("2026-09-09T06:00:00Z", "New QB", "2", "QB", 9, 1),   # day before wk1 → wk1
        _snap("2026-09-10T06:00:00Z", "Game-day QB", "3", "QB", 9, 1),  # on wk1 day → wk2
    ]), 2026, WEEK_STARTS)
    by_week = df.set_index("week")["player_name"].to_dict()
    assert by_week == {1: "New QB", 2: "Game-day QB"}
    assert (df["season"] == 2026).all()


def test_snapshots_after_last_week_are_dropped():
    df = _clean_daily(_daily([_snap("2027-02-20T06:00:00Z", "A", "1", "QB", 9, 1)]),
                      2026, WEEK_STARTS)
    assert df.empty


def test_depth_is_ranked_within_slot():
    dt = "2026-09-09T06:00:00Z"
    df = _clean_daily(_daily([
        _snap(dt, "WR1", "1", "WR", 1, 1),
        _snap(dt, "WR2", "2", "WR", 2, 2),
        _snap(dt, "WR3", "3", "WR", 8, 3),
        _snap(dt, "WR1b", "4", "WR", 1, 4),
    ]), 2026, WEEK_STARTS)
    depth = df.set_index("player_name")["depth_team"].to_dict()
    assert depth == {"WR1": 1, "WR2": 1, "WR3": 1, "WR1b": 2}


def test_special_teams_only_kickers_and_best_depth_per_player():
    dt = "2026-09-09T06:00:00Z"
    df = _clean_daily(_daily([
        _snap(dt, "Returner", "1", "WR", 1, 3),
        _snap(dt, "Returner", "1", "KR", 1, 1, grp="Special Teams"),
        _snap(dt, "Kicker", "2", "PK", 1, 1, grp="Special Teams"),
        _snap(dt, "Tackle", "3", "LT", 2, 1),
    ]), 2026, WEEK_STARTS)
    rows = df.set_index("player_name")[["position", "depth_team"]].to_dict("index")
    assert rows == {
        "Returner": {"position": "WR", "depth_team": 1},
        "Kicker": {"position": "K", "depth_team": 1},
        "Tackle": {"position": "T", "depth_team": 1},
    }


def test_ingest_loads_one_season_at_a_time():
    old = pd.DataFrame({
        "season": [2024], "club_code": ["KC"], "week": [1.0], "depth_team": ["1"],
        "position": ["QB"], "gsis_id": ["00-1"], "full_name": ["Old Format QB"],
    })
    new = _daily([_snap("2025-09-03T06:00:00Z", "New Format QB", "00-2", "QB", 9, 1)])
    sched = pd.DataFrame({"week": [1], "gameday": ["2025-09-04"]})

    class _PL:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    written = []

    class _Writer:
        def write_depth_charts(self, df):
            written.append(df)
            return len(df)

    with patch("ironclad.ingest.depth_charts.nfl") as nfl:
        nfl.load_depth_charts.side_effect = lambda seasons: _PL(old if seasons == [2024] else new)
        nfl.load_schedules.return_value = _PL(sched)
        n = DepthChartIngestor(_Writer()).ingest([2024, 2025])

    assert n == 2
    assert [call.args[0] for call in nfl.load_depth_charts.call_args_list] == [[2024], [2025]]
    assert written[0]["player_name"].tolist() == ["Old Format QB"]
    assert written[0]["team"].tolist() == ["KC"]
    assert written[1]["player_name"].tolist() == ["New Format QB"]
    assert written[1][["season", "week"]].values.tolist() == [[2025, 1]]
