"""Betting performance tracking: persist edges, settle bets, report P&L."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ironclad.betting.ev import american_to_decimal


def save_edges(
    conn,
    game_id: str,
    n_draws: int,
    edges_df: pd.DataFrame,
) -> list[str]:
    """Persist PropAnalyzer output to gold.betting_edges.

    Returns the list of generated edge_ids (8-char hex) for use in
    ``ironclad settle``.
    """
    if edges_df.empty:
        return []

    rows = []
    edge_ids: list[str] = []
    analyzed_at = datetime.now(timezone.utc)
    for _, row in edges_df.iterrows():
        eid = uuid.uuid4().hex[:8]
        edge_ids.append(eid)
        rows.append({
            "edge_id":     eid,
            "analyzed_at": analyzed_at,
            "game_id":     game_id,
            "n_draws":     n_draws,
            "player_id":   str(row.get("player_id", "")),
            "player_name": str(row.get("player_name", "")),
            "team":        str(row.get("team", "")),
            "position":    str(row.get("position", "")),
            "stat_type":   str(row.get("stat_type", "")),
            "market_line": _float_or_none(row.get("market_line")),
            "side":        str(row.get("side", "")),
            "odds":        _int_or_none(row.get("odds")),
            "model_prob":  _float_or_none(row.get("model_prob")),
            "market_prob": _float_or_none(row.get("market_prob")),
            "edge":        _float_or_none(row.get("edge")),
            "ev":          _float_or_none(row.get("ev")),
            "kelly":       _float_or_none(row.get("kelly")),
            "model_p10":   _float_or_none(row.get("model_p10")),
            "model_p50":   _float_or_none(row.get("model_p50")),
            "model_p90":   _float_or_none(row.get("model_p90")),
        })

    df = pd.DataFrame(rows)
    conn.register("_tmp_edges", df)
    conn.execute("INSERT INTO gold.betting_edges SELECT * FROM _tmp_edges")
    conn.unregister("_tmp_edges")
    return edge_ids


def settle_bet(
    conn,
    edge_id: str,
    result: str,
    units_wagered: float,
) -> dict[str, Any]:
    """Record the settled outcome of a placed bet.

    Parameters
    ----------
    edge_id:       8-char hex ID from ``save_edges``
    result:        'win', 'loss', or 'push'
    units_wagered: stake in abstract units (e.g. 1.0 = one Kelly unit)

    Returns the inserted row as a dict.
    """
    if result not in ("win", "loss", "push"):
        raise ValueError(f"result must be 'win', 'loss', or 'push'; got {result!r}")

    edge_row = conn.execute(
        "SELECT * FROM gold.betting_edges WHERE edge_id = ?", [edge_id]
    ).df()
    if edge_row.empty:
        raise ValueError(f"edge_id {edge_id!r} not found in gold.betting_edges")

    row = edge_row.iloc[0]
    odds = int(row["odds"]) if row["odds"] is not None else -110
    if result == "win":
        profit = units_wagered * (american_to_decimal(odds) - 1.0)
    elif result == "loss":
        profit = -units_wagered
    else:
        profit = 0.0

    settled_row: dict[str, Any] = {
        "result_id":     uuid.uuid4().hex[:8],
        "edge_id":       edge_id,
        "game_id":       str(row["game_id"]),
        "player_id":     str(row["player_id"]),
        "stat_type":     str(row["stat_type"]),
        "side":          str(row["side"]),
        "market_line":   _float_or_none(row.get("market_line")),
        "odds":          odds,
        "model_prob":    _float_or_none(row.get("model_prob")),
        "ev":            _float_or_none(row.get("ev")),
        "units_wagered": float(units_wagered),
        "result":        result,
        "profit_units":  round(profit, 4),
        "settled_at":    datetime.now(timezone.utc),
    }
    df = pd.DataFrame([settled_row])
    conn.register("_tmp_result", df)
    conn.execute("INSERT INTO gold.betting_results SELECT * FROM _tmp_result")
    conn.unregister("_tmp_result")
    return settled_row


def load_results(conn, game_id: str | None = None) -> pd.DataFrame:
    """Load settled bets, optionally filtered by game_id."""
    where = f"WHERE game_id = '{game_id}'" if game_id else ""
    return conn.execute(
        f"SELECT * FROM gold.betting_results {where} ORDER BY settled_at DESC"
    ).df()


def pnl_summary(results_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate P&L statistics from a results DataFrame."""
    if results_df.empty:
        return {}
    total = len(results_df)
    wins = int((results_df["result"] == "win").sum())
    losses = int((results_df["result"] == "loss").sum())
    total_profit = float(results_df["profit_units"].sum())
    total_wagered = float(results_df["units_wagered"].sum())
    return {
        "total_bets":          total,
        "wins":                wins,
        "losses":              losses,
        "win_rate":            round(wins / total, 4) if total > 0 else 0.0,
        "total_profit_units":  round(total_profit, 4),
        "total_wagered_units": round(total_wagered, 4),
        "roi":                 round(total_profit / total_wagered, 4) if total_wagered > 0 else 0.0,
    }


_STAT_COL: dict[str, str] = {
    "pass_yards":   "passing_yards",
    "pass_tds":     "passing_tds",
    "rush_yards":   "rushing_yards",
    "rush_tds":     "rushing_tds",
    "receptions":   "receptions",
    "rec_yards":    "receiving_yards",
    "rec_tds":      "receiving_tds",
    "carries":      "carries",
    "completions":  "completions",
    "pass_attempts": "attempts",
    "targets":      "targets",
}


def auto_settle_week(
    conn,
    season: int,
    week: int,
    units_wagered: float = 1.0,
) -> list[dict]:
    """Auto-settle all unsettled edges for a completed week against actual stats.

    Looks up each edge's player actual stat from bronze.player_stats_weekly,
    compares to the market_line + side (over/under), and calls settle_bet().

    Returns a list of settled-row dicts (one per edge that was settled).
    """
    already_settled = conn.execute(
        "SELECT edge_id FROM gold.betting_results"
    ).df()["edge_id"].tolist()

    edges = conn.execute("""
        SELECT e.edge_id, e.player_id, e.stat_type, e.side, e.market_line,
               g.season, g.week
        FROM gold.betting_edges e
        JOIN silver.games g ON e.game_id = g.game_id
        WHERE g.season = ? AND g.week = ? AND g.season_type = 'REG'
    """, [season, week]).df()

    if edges.empty:
        return []

    actuals = conn.execute("""
        SELECT player_id, season, week,
               passing_yards, passing_tds, rushing_yards, rushing_tds,
               receptions, receiving_yards, receiving_tds,
               carries, completions, attempts, targets
        FROM bronze.player_stats_weekly
        WHERE season = ? AND week = ?
    """, [season, week]).df()

    settled = []
    for _, edge in edges.iterrows():
        if edge["edge_id"] in already_settled:
            continue

        col = _STAT_COL.get(edge["stat_type"])
        if col is None:
            continue

        player_rows = actuals[actuals["player_id"] == edge["player_id"]]
        if player_rows.empty:
            continue

        actual_val = float(player_rows.iloc[0][col])
        line = float(edge["market_line"]) if edge["market_line"] is not None else None
        if line is None:
            continue

        side = str(edge["side"]).lower()
        if side == "over":
            outcome = "win" if actual_val > line else ("push" if actual_val == line else "loss")
        elif side == "under":
            outcome = "win" if actual_val < line else ("push" if actual_val == line else "loss")
        else:
            continue

        row = settle_bet(conn, edge["edge_id"], outcome, units_wagered)
        settled.append(row)

    return settled


# ── Helpers ───────────────────────────────────────────────────────────────────

def _float_or_none(v: Any) -> float | None:
    try:
        return float(v) if v is not None and str(v) not in ("nan", "None", "") else None
    except (TypeError, ValueError):
        return None


def _int_or_none(v: Any) -> int | None:
    try:
        return int(v) if v is not None and str(v) not in ("nan", "None", "") else None
    except (TypeError, ValueError):
        return None
