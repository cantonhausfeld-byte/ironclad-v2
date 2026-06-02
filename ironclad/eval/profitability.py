"""Game-level profitability and CLV harness for backtested predictions.

Simulates spread and over/under bets using gold.backtest_predictions, settles
against actual outcomes, and reports ROI + CLV proxy. Uses the stored Vegas
line as both the betting line and the market-efficiency benchmark.

CLV proxy: model edge = model_implied_prob - market_implied_prob (from the
Vegas spread/total). Positive average edge → positive expected CLV. True CLV
requires closing-line odds we don't have for historical games, so this is a
necessary (but not sufficient) condition for profitability.

-110 juice is standard for spread/total bets. P(win) at breakeven = 110/210 ≈ 0.524.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

# NFL margin std-dev used to convert spread → win prob
_SPREAD_STD = 13.86
_VIG_BREAKEVEN = 110 / 210  # ≈ 0.5238 — min win rate to profit at -110


def _spread_to_prob(spread: float) -> float:
    """Convert home spread to P(home wins outright).

    `spread` follows the silver.games convention: **positive = home favored**.
    A +9.5 (home favored by 9.5) returns ≈ 0.75; 0 returns 0.5.
    """
    return float(norm.cdf(spread / _SPREAD_STD))


def simulate_spread_bets(
    predictions: pd.DataFrame,
    min_edge: float = 0.03,
    juice: int = -110,
) -> pd.DataFrame:
    """Simulate home/away spread bets wherever model disagrees with the line.

    For each game with both model predictions and a Vegas spread, compute:
    - market_implied_prob = spread_to_prob(vegas_spread)
    - model_edge = home_win_prob - market_implied_prob
    - Bet home when model_edge > min_edge; bet away when < -min_edge.
    - Settle at standard juice.

    Returns a DataFrame of simulated bets with columns:
        game_id, season, week, side, model_prob, market_prob, edge,
        bet_result, profit_units.
    """
    decimal_odds = _american_to_decimal(juice)
    rows: list[dict] = []

    df = predictions.dropna(subset=["home_win_prob", "vegas_spread", "home_margin_actual"])

    for _, row in df.iterrows():
        market_prob = _spread_to_prob(float(row["vegas_spread"]))
        model_prob = float(row["home_win_prob"])
        edge = model_prob - market_prob

        if abs(edge) < min_edge:
            continue

        # Decide side and compute actual cover
        actual_margin = float(row["home_margin_actual"])
        vegas_spread = float(row["vegas_spread"])
        # positive vegas_spread = home favored by that many points
        # home covers when actual_margin > vegas_spread (home beats their handicap)
        home_covers = actual_margin > vegas_spread

        if edge > 0:
            side = "home"
            win = home_covers
            prob = model_prob
        else:
            side = "away"
            win = not home_covers
            prob = 1.0 - model_prob

        profit = (decimal_odds - 1.0) if win else -1.0
        rows.append({
            "game_id":        str(row["game_id"]),
            "season":         int(row["season"]),
            "week":           int(row["week"]),
            "bet_type":       "spread",
            "side":           side,
            "model_prob":     round(prob, 4),
            "market_prob":    round(market_prob if side == "home" else 1 - market_prob, 4),
            "edge":           round(abs(edge), 4),
            "vegas_line":     float(row["vegas_spread"]),
            "result":         "win" if win else "loss",
            "profit_units":   round(profit, 4),
        })

    return pd.DataFrame(rows)


def simulate_total_bets(
    predictions: pd.DataFrame,
    min_line_diff: float = 1.5,
    juice: int = -110,
) -> pd.DataFrame:
    """Simulate over/under bets where model total differs from Vegas total.

    Uses a threshold on the raw line difference (in points) rather than a
    probability edge, since we don't have a calibrated total distribution.
    min_line_diff=1.5 means only bet when model is >1.5 pts from the line.

    Returns bets with the same schema as simulate_spread_bets.
    """
    decimal_odds = _american_to_decimal(juice)
    rows: list[dict] = []

    df = predictions.dropna(subset=["total_pred", "vegas_total", "total_actual"])

    for _, row in df.iterrows():
        diff = float(row["total_pred"]) - float(row["vegas_total"])
        if abs(diff) < min_line_diff:
            continue

        actual_total = float(row["total_actual"])
        vegas_total = float(row["vegas_total"])

        if diff > 0:
            side = "over"
            win = actual_total > vegas_total
        else:
            side = "under"
            win = actual_total < vegas_total

        profit = (decimal_odds - 1.0) if win else -1.0
        rows.append({
            "game_id":        str(row["game_id"]),
            "season":         int(row["season"]),
            "week":           int(row["week"]),
            "bet_type":       "total",
            "side":           side,
            "model_prob":     None,
            "market_prob":    None,
            "edge":           round(abs(diff), 2),
            "vegas_line":     vegas_total,
            "result":         "win" if win else "loss",
            "profit_units":   round(profit, 4),
        })

    return pd.DataFrame(rows)


def profitability_summary(bets: pd.DataFrame) -> dict[str, Any]:
    """Aggregate ROI and hit-rate statistics, split by bet type and overall."""
    if bets.empty:
        return {"total_bets": 0}

    def _stats(df: pd.DataFrame) -> dict[str, Any]:
        n = len(df)
        wins = int((df["result"] == "win").sum())
        total_profit = float(df["profit_units"].sum())
        avg_edge = float(df["edge"].mean()) if "edge" in df.columns and df["edge"].notna().any() else None
        return {
            "n_bets":      n,
            "win_rate":    round(wins / n, 4) if n > 0 else 0.0,
            "roi":         round(total_profit / n, 4) if n > 0 else 0.0,
            "profit":      round(total_profit, 4),
            "avg_edge":    round(avg_edge, 4) if avg_edge is not None else None,
            "breakeven":   round(_VIG_BREAKEVEN, 4),
        }

    result: dict[str, Any] = {"overall": _stats(bets)}
    for bet_type, grp in bets.groupby("bet_type"):
        result[str(bet_type)] = _stats(grp)

    return result


def clv_summary(bets: pd.DataFrame) -> dict[str, Any]:
    """Compute CLV proxy: average model edge vs market across spread bets.

    A positive mean edge indicates the model is consistently finding value
    on the correct side of the line — a necessary condition for profitability
    in an efficient market. This is a CLV proxy since we don't have true
    closing-line odds.
    """
    spread_bets = bets[bets["bet_type"] == "spread"].copy() if not bets.empty else bets
    if spread_bets.empty:
        return {}

    edge_col = spread_bets["edge"]
    # Edge direction: positive = bet was in the direction model favored
    # (already absolute in the bets table; signed version = edge for wins, -edge for losses)
    signed_edge = spread_bets.apply(
        lambda r: r["edge"] if r["result"] == "win" else -r["edge"], axis=1
    )
    return {
        "mean_edge":         round(float(edge_col.mean()), 4),
        "mean_signed_edge":  round(float(signed_edge.mean()), 4),
        "pct_positive_edge": round(float((edge_col > 0).mean()), 4),
        "n_bets":            len(spread_bets),
    }


def run_profitability_harness(
    conn,
    run_id: str | None = None,
    min_edge: float = 0.03,
    min_total_diff: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full profitability harness against stored backtest predictions.

    Returns (bets_df, summary_dict).
    """
    query = """
        SELECT * FROM gold.backtest_predictions
        ORDER BY season, week
    """ if run_id is None else """
        SELECT * FROM gold.backtest_predictions
        WHERE backtest_run_id = ?
        ORDER BY season, week
    """

    try:
        if run_id is None:
            preds = conn.execute(query).df()
        else:
            preds = conn.execute(query, [run_id]).df()
    except Exception as exc:
        raise RuntimeError(f"Could not load backtest predictions: {exc}") from exc

    if preds.empty:
        return pd.DataFrame(), {"total_bets": 0}

    spread_bets = simulate_spread_bets(preds, min_edge=min_edge)
    total_bets = simulate_total_bets(preds, min_line_diff=min_total_diff)
    all_bets = pd.concat([spread_bets, total_bets], ignore_index=True)

    summary = profitability_summary(all_bets)
    summary["clv"] = clv_summary(all_bets)
    summary["n_games_in_backtest"] = len(preds)
    summary["seasons"] = sorted(preds["season"].unique().tolist()) if "season" in preds.columns else []

    return all_bets, summary


def _american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)
