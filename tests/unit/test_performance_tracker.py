"""Tests for betting performance tracking (save, settle, P&L)."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ironclad.eval.performance_tracker import (
    load_results,
    pnl_summary,
    save_edges,
    settle_bet,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _edges_df(n: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "player_id":   f"P{i}",
            "player_name": f"Player {i}",
            "team":        "KC",
            "position":    "WR",
            "stat_type":   "rec_yards",
            "market_line": 55.5,
            "side":        "over",
            "odds":        -110,
            "model_prob":  0.58,
            "market_prob": 0.52,
            "edge":        0.06,
            "ev":          0.07,
            "kelly":       0.12,
            "model_p10":   20.0,
            "model_p50":   60.0,
            "model_p90":   100.0,
        })
    return pd.DataFrame(rows)


# ── save_edges ────────────────────────────────────────────────────────────────

def test_save_edges_inserts_rows(conn):
    df = _edges_df(3)
    ids = save_edges(conn, "2024_14_LAC_KC", 1000, df)
    count = conn.execute("SELECT COUNT(*) FROM gold.betting_edges").fetchone()[0]
    assert count == 3


def test_save_edges_returns_edge_ids(conn):
    df = _edges_df(2)
    ids = save_edges(conn, "2024_14_LAC_KC", 1000, df)
    assert len(ids) == 2
    assert all(len(eid) == 8 for eid in ids)


def test_save_edges_empty_df_returns_empty(conn):
    ids = save_edges(conn, "2024_14_LAC_KC", 1000, pd.DataFrame())
    assert ids == []
    count = conn.execute("SELECT COUNT(*) FROM gold.betting_edges").fetchone()[0]
    assert count == 0


# ── settle_bet ────────────────────────────────────────────────────────────────

def _save_one_edge(conn, odds: int = 100) -> str:
    df = _edges_df(1)
    df["odds"] = odds
    ids = save_edges(conn, "2024_14_LAC_KC", 1000, df)
    return ids[0]


def test_settle_win_profit(conn):
    eid = _save_one_edge(conn, odds=100)  # +100 → decimal 2.0 → profit = 1.0 per unit
    outcome = settle_bet(conn, eid, "win", 1.0)
    assert outcome["profit_units"] == pytest.approx(1.0)
    assert outcome["result"] == "win"


def test_settle_loss_profit(conn):
    eid = _save_one_edge(conn, odds=-110)
    outcome = settle_bet(conn, eid, "loss", 2.0)
    assert outcome["profit_units"] == pytest.approx(-2.0)


def test_settle_push_profit(conn):
    eid = _save_one_edge(conn, odds=-110)
    outcome = settle_bet(conn, eid, "push", 1.5)
    assert outcome["profit_units"] == pytest.approx(0.0)


def test_settle_persists_to_db(conn):
    eid = _save_one_edge(conn)
    settle_bet(conn, eid, "win", 1.0)
    count = conn.execute("SELECT COUNT(*) FROM gold.betting_results").fetchone()[0]
    assert count == 1


def test_settle_unknown_edge_id_raises(conn):
    with pytest.raises(ValueError, match="not found"):
        settle_bet(conn, "deadbeef", "win", 1.0)


def test_settle_invalid_result_raises(conn):
    eid = _save_one_edge(conn)
    with pytest.raises(ValueError):
        settle_bet(conn, eid, "draw", 1.0)


# ── load_results ──────────────────────────────────────────────────────────────

def test_load_results_empty_when_none(conn):
    df = load_results(conn)
    assert df.empty


def test_load_results_returns_settled_rows(conn):
    eid = _save_one_edge(conn)
    settle_bet(conn, eid, "win", 1.0)
    df = load_results(conn)
    assert len(df) == 1
    assert df.iloc[0]["result"] == "win"


# ── pnl_summary ───────────────────────────────────────────────────────────────

def test_pnl_summary_correct(conn):
    # 2 wins at +100 (profit +1 each) + 1 loss at -110 (profit -1)
    for _ in range(2):
        eid = _save_one_edge(conn, odds=100)
        settle_bet(conn, eid, "win", 1.0)
    eid = _save_one_edge(conn, odds=-110)
    settle_bet(conn, eid, "loss", 1.0)

    df = load_results(conn)
    summary = pnl_summary(df)
    assert summary["total_bets"] == 3
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["win_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert summary["total_profit_units"] == pytest.approx(1.0)  # +1 +1 -1
    assert summary["roi"] > 0


def test_pnl_summary_empty_df():
    assert pnl_summary(pd.DataFrame()) == {}
