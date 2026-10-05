"""Engine context allocation: starting QB choice and team volume normalization."""
from __future__ import annotations

import pytest

from ironclad.simulation.engine import (
    _TEAM_CARRIES,
    _TEAM_TARGETS,
    _allocate_qb_pass_volume,
    _normalize_team_volume,
)
from ironclad.simulation.player_draw import PlayerContext


def _ctx(pid, pos, att=0.0, carries=0.0, targets=0.0, depth=None, avail=1.0):
    return PlayerContext(
        player_id=pid, player_name=pid, team="NO", position=pos, is_home=True,
        availability=avail, targets_projected=targets, carries_projected=carries,
        pass_attempts_projected=att, catch_rate=0.65, yards_per_target=8.0,
        yards_per_carry=4.2, td_rate_per_target=0.05, td_rate_per_carry=0.04,
        depth_team=depth,
    )


def test_depth_chart_starter_beats_previous_starters_volume():
    # Rattler started earlier games (more projected volume); Shough is QB1 now.
    old = _ctx("rattler", "QB", att=34.0, carries=3.0, depth=2)
    new = _ctx("shough", "QB", att=12.0, carries=2.0, depth=1)
    _allocate_qb_pass_volume([old, new])
    assert new.pass_attempts_projected == 34.0      # inherits team passing volume
    assert new.carries_projected == 2.0             # keeps his own rushing
    assert old.pass_attempts_projected == old.carries_projected == 0.0


def test_injured_qb1_falls_to_next_on_chart():
    qb1 = _ctx("qb1", "QB", att=34.0, depth=1, avail=0.0)
    qb2 = _ctx("qb2", "QB", att=5.0, depth=2)
    _allocate_qb_pass_volume([qb1, qb2])
    assert qb2.pass_attempts_projected == 34.0
    assert qb1.pass_attempts_projected == 0.0


def test_without_depth_chart_uses_projected_volume():
    a = _ctx("a", "QB", att=30.0)
    b = _ctx("b", "QB", att=8.0)
    _allocate_qb_pass_volume([a, b])
    assert a.pass_attempts_projected == 30.0 and b.pass_attempts_projected == 0.0


def test_team_volume_normalized_and_groups_keep_their_share():
    ctxs = [
        _ctx("wr1", "WR", targets=8.0), _ctx("wr2", "WR", targets=5.0), _ctx("wr5", "WR", targets=1.5),
        _ctx("te", "TE", targets=5.0),
        _ctx("rb1", "RB", targets=4.0, carries=16.0), _ctx("rb2", "RB", targets=2.0, carries=7.0),
        _ctx("qb", "QB", att=34.0, carries=4.0),
    ]
    _normalize_team_volume(ctxs, gamma=2.0)
    by = {c.player_id: c for c in ctxs}
    total_t = sum(c.targets_projected for c in ctxs)
    assert total_t == pytest.approx(_TEAM_TARGETS)
    # RB/FB group keeps 6/25.5 of targets; TE keeps 5/25.5
    assert by["rb1"].targets_projected + by["rb2"].targets_projected == pytest.approx(_TEAM_TARGETS * 6 / 25.5)
    assert by["te"].targets_projected == pytest.approx(_TEAM_TARGETS * 5 / 25.5)
    # Sharpening concentrates within a group
    assert by["wr5"].targets_projected / by["wr1"].targets_projected < 1.5 / 8.0
    # QB carries untouched; other carriers fill the rest of team carries
    assert by["qb"].carries_projected == 4.0
    assert by["rb1"].carries_projected + by["rb2"].carries_projected == pytest.approx(_TEAM_CARRIES - 4.0)
