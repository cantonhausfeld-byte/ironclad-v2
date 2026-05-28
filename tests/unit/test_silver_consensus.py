"""Tests for multi-book odds consensus overlay in the silver transform."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ironclad.store.silver import (
    SilverTransformer,
    _devig_home_prob,
    _remove_vig,
)


# ── de-vig helpers ──────────────────────────────────────────────────────────

def test_devig_even_market_is_half():
    assert _devig_home_prob(-110, -110) == pytest.approx(0.5, abs=1e-9)


def test_devig_missing_side_returns_none():
    assert _devig_home_prob(-150, None) is None
    assert _devig_home_prob(None, None) is None


def test_remove_vig_falls_back_when_missing():
    assert _remove_vig(None, None) == (0.573, 0.427)


def test_remove_vig_sums_to_one():
    p_h, p_a = _remove_vig(-150, 130)
    assert p_h + p_a == pytest.approx(1.0, abs=1e-9)
    assert p_h > p_a  # home is favored


# ── consensus aggregation ───────────────────────────────────────────────────

def _insert_odds(conn, game_id, source, spread_home, total_over, ml_home, ml_away):
    conn.execute(
        "INSERT INTO bronze.odds (game_id, source, retrieved_at, spread_home, "
        "total_over, moneyline_home, moneyline_away) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [game_id, source, datetime.now(tz=timezone.utc), spread_home,
         total_over, ml_home, ml_away],
    )


def test_consensus_empty_when_no_odds(conn):
    assert SilverTransformer(conn)._consensus_from_odds().empty


def test_consensus_aggregates_and_flips_spread_sign(conn):
    # Two books, home favored (negative handicap on The Odds API).
    _insert_odds(conn, "2025_01_AAA_BBB", "dk", -3.0, 47.0, -150, 130)
    _insert_odds(conn, "2025_01_AAA_BBB", "fd", -4.0, 48.0, -140, 120)

    cons = SilverTransformer(conn)._consensus_from_odds()
    row = cons[cons["game_id"] == "2025_01_AAA_BBB"].iloc[0]

    # median spread_home = -3.5 → stored as +3.5 (positive = home favored)
    assert row["cons_spread"] == pytest.approx(3.5)
    assert row["cons_total"] == pytest.approx(47.5)
    # de-vig home prob per book ≈ 0.580 and 0.562 → median ≈ 0.571
    assert row["cons_home_prob"] == pytest.approx(0.571, abs=0.01)


# ── overlay behaviour ───────────────────────────────────────────────────────

def test_overlay_prefers_consensus_over_opening_line(conn):
    _insert_odds(conn, "2025_01_AAA_BBB", "dk", -3.0, 47.0, -150, 130)

    games = pd.DataFrame([
        {"game_id": "2025_01_AAA_BBB", "spread_consensus": -1.0,
         "total_consensus": 41.0, "home_ml_implied": 0.5, "away_ml_implied": 0.5},
        {"game_id": "2025_01_CCC_DDD", "spread_consensus": 2.0,
         "total_consensus": 44.0, "home_ml_implied": 0.55, "away_ml_implied": 0.45},
    ])
    out = SilverTransformer(conn)._overlay_odds_consensus(games)

    # Game with live odds is overridden by consensus...
    live = out[out["game_id"] == "2025_01_AAA_BBB"].iloc[0]
    assert live["spread_consensus"] == pytest.approx(3.0)
    assert live["total_consensus"] == pytest.approx(47.0)
    assert live["home_ml_implied"] + live["away_ml_implied"] == pytest.approx(1.0)
    assert live["home_ml_implied"] > 0.5

    # ...game without live odds keeps its opening line untouched.
    hist = out[out["game_id"] == "2025_01_CCC_DDD"].iloc[0]
    assert hist["spread_consensus"] == pytest.approx(2.0)
    assert hist["total_consensus"] == pytest.approx(44.0)


def test_overlay_noop_when_no_odds(conn):
    games = pd.DataFrame([
        {"game_id": "2025_01_CCC_DDD", "spread_consensus": 2.0,
         "total_consensus": 44.0, "home_ml_implied": 0.55, "away_ml_implied": 0.45},
    ])
    out = SilverTransformer(conn)._overlay_odds_consensus(games)
    pd.testing.assert_frame_equal(out, games)
