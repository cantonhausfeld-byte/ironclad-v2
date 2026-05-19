"""End-to-end integration tests confirming production readiness fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables


@pytest.fixture
def conn():
    c = in_memory_connection()
    create_all_tables(c)
    return c


# ── P1-A: FTN enrichment works without team column ───────────────────────────

def test_ftn_enrichment_no_team_column(conn):
    """FTN charting rows without a team column should still enrich silver stats."""
    conn.execute("""
        INSERT INTO silver.team_game_stats
            (game_id, season, week, team, opponent, is_home, play_action_rate)
        VALUES ('2024_01_KC_BAL', 2024, 1, 'KC', 'BAL', true, NULL)
    """)
    conn.execute("""
        INSERT INTO bronze.ftn_charting
            (game_id, season, week, is_play_action, is_motion, n_blitzers, n_defense_box)
        VALUES
            ('2024_01_KC_BAL', 2024, 1, 1, 1, 4, 6),
            ('2024_01_KC_BAL', 2024, 1, 0, 0, 3, 7)
    """)

    from ironclad.store.silver import SilverTransformer
    SilverTransformer(conn)._enrich_with_ftn(2024)

    row = conn.execute(
        "SELECT play_action_rate FROM silver.team_game_stats WHERE game_id = '2024_01_KC_BAL'"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(0.5, abs=0.01)


# ── P1-B: PBP REG filter excludes playoff plays ───────────────────────────────

def test_pbp_reg_filter():
    """_clean() should drop non-REG rows from play_by_play data."""
    from ironclad.ingest.play_by_play import _clean

    raw = pd.DataFrame({
        "season_type": ["REG", "POST", "PRE"],
        "game_id": ["g1", "g2", "g3"],
        "play_id": [1, 2, 3],
        "season": [2024, 2024, 2024],
    })
    result = _clean(raw)
    assert len(result) == 1
    assert result.iloc[0]["game_id"] == "g1"


# ── P1-C: Reconciler invariant — no negative player yards ────────────────────

def test_reconciler_no_negative_yards():
    """200 simulation draws should produce no negative receiving or rush yards."""
    from ironclad.simulation.player_draw import PlayerDraw, PlayerContext

    rng = np.random.default_rng(seed=0)
    ctx = PlayerContext(
        player_id="test",
        player_name="Test Player",
        team="KC",
        position="WR",
        is_home=True,
        availability=1.0,
        targets_projected=6.0,
        carries_projected=0.5,
        pass_attempts_projected=0.0,
        catch_rate=0.65,
        yards_per_target=8.0,
        yards_per_carry=4.2,
        td_rate_per_target=0.05,
        td_rate_per_carry=0.0,
        yards_per_target_std=3.0,
        yards_per_carry_std=3.5,
    )
    drawer = PlayerDraw()
    n_negative_rec = 0
    n_negative_rush = 0
    for _ in range(200):
        draw = drawer.draw(rng, ctx, team_pass_att=35, team_rush_att=25)
        if draw.rec_yards < 0:
            n_negative_rec += 1
        if draw.rush_yards < 0:
            n_negative_rush += 1

    assert n_negative_rec == 0, f"Got {n_negative_rec} negative rec_yards draws"
    assert n_negative_rush == 0, f"Got {n_negative_rush} negative rush_yards draws"


# ── P2-A: Zero-player simulation raises RuntimeError ─────────────────────────

def test_simulate_raises_on_empty_roster(conn):
    """MatchupWorkflow.simulate() should raise RuntimeError if player features are missing."""
    import datetime
    from unittest.mock import patch

    conn.execute("""
        INSERT INTO silver.games
            (game_id, season, week, home_team, away_team, gameday, season_type)
        VALUES ('2024_01_KC_BAL', 2024, 1, 'KC', 'BAL', '2024-09-05', 'REG')
    """)
    conn.execute("""
        INSERT INTO gold.team_game_features
            (game_id, season, week, team, opponent, is_home, cutoff_ts, feature_version)
        VALUES
            ('2024_01_KC_BAL', 2024, 1, 'KC', 'BAL', true, NOW(), 'v1'),
            ('2024_01_KC_BAL', 2024, 1, 'BAL', 'KC', false, NOW(), 'v1')
    """)
    # No player features inserted → should trigger the guard

    from ironclad.workflow.matchup import MatchupWorkflow

    with patch("ironclad.workflow.matchup.get_connection", return_value=conn):
        wf = MatchupWorkflow(n_draws=100)
        with pytest.raises(RuntimeError, match="Insufficient player features"):
            wf.simulate("2024_01_KC_BAL")


# ── P5-A: PropAnalyzer output contains no skip rows ──────────────────────────

def test_prop_analyzer_no_skip_rows_in_output():
    """PropAnalyzer.analyze() must not return rows with side == 'skip'."""
    from unittest.mock import MagicMock
    import numpy as np
    from ironclad.betting.props import PropAnalyzer, PropLine

    # Build a minimal fake SimulationResult._player_df
    player_df = pd.DataFrame({
        "player_id": ["p1", "p1", "p1"],
        "player_name": ["Test", "Test", "Test"],
        "team": ["KC", "KC", "KC"],
        "position": ["WR", "WR", "WR"],
        "rec_yards": [55.0, 70.0, 40.0],
        # NOTE: no 'rush_yards' column → forces skip for that stat
    })

    sim = MagicMock()
    sim._player_df = player_df

    props = [
        PropLine(player_id="p1", stat_type="rec_yards", line=50.0, over_odds=-110, under_odds=-110),
        PropLine(player_id="p1", stat_type="rush_yards", line=10.0, over_odds=-115),  # will skip
    ]

    result = PropAnalyzer().analyze(sim, props)
    assert "skip" not in result["side"].values, "PropAnalyzer returned skip rows"
    assert len(result) == 1
    assert result.iloc[0]["stat_type"] == "rec_yards"
