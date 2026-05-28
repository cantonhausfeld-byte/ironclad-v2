"""Approximate prop-betting profitability from PropBacktester output.

We have NO historical player-prop lines (The Odds API free tier only returns
current odds), so true prop ROI/CLV cannot be backtested. This module gives an
*approximate* profitability signal by betting the model against a synthetic
line and settling on actual outcomes.

Line modes:
  - "season_mean": line = the player's mean actual for the stat across the
    backtested games. This is a SHARP-BOOK PROXY and an optimistic upper bound:
    it uses in-sample (end-of-period) information to set the line, so a real
    book would be at least this accurate. Read it as a necessary condition —
    if the model can't beat a static season-mean line, it can't beat a real
    book. Not a tradeable ROI.
  - "model_p50": line = the model's own median. A null baseline that should
    come out near break-even-minus-vig; useful to sanity-check the harness.

Both sides are priced at standard juice (default -110). The model bets the side
its distribution favors when the EV clears `min_ev`, and we settle on the actual.
P(over) is estimated by interpolating the percentiles PropBacktester stores
(p10/p25/p50/p75/p90), so it inherits their granularity — adequate for an
explicitly approximate signal, not for cent-precise pricing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ironclad.betting.ev import american_to_decimal, expected_value

_PCTL_COLS = [
    (0.10, "predicted_p10"),
    (0.25, "predicted_p25"),
    (0.50, "predicted_p50"),
    (0.75, "predicted_p75"),
    (0.90, "predicted_p90"),
]
_PROB_FLOOR, _PROB_CEIL = 0.02, 0.98


def _prob_over(row: pd.Series, line: float) -> float:
    """Estimate P(stat > line) from the stored predicted percentiles."""
    xs = [float(row[c]) for _, c in _PCTL_COLS]
    ps = [p for p, _ in _PCTL_COLS]
    # np.interp needs non-decreasing xs (percentiles are); clamp the tails.
    cdf = float(np.interp(line, xs, ps, left=_PROB_FLOOR, right=_PROB_CEIL))
    p_over = 1.0 - cdf
    return min(_PROB_CEIL, max(_PROB_FLOOR, p_over))


def _lines(df: pd.DataFrame, line_mode: str) -> pd.Series:
    if line_mode == "model_p50":
        return df["predicted_p50"].astype(float)
    if line_mode == "season_mean":
        # Book proxy: the player's mean actual for this stat over the sample.
        return df.groupby(["player_id", "stat_type"])["actual_value"].transform("mean")
    raise ValueError(f"Unknown line_mode: {line_mode!r}")


def approximate_prop_roi(
    df: pd.DataFrame,
    line_mode: str = "season_mean",
    juice: int = -110,
    min_ev: float = 0.0,
) -> pd.DataFrame:
    """Return one row per simulated bet (empty if df is empty).

    Columns: player_id, player_name, position, stat_type, line, side,
    model_prob, ev, actual_value, result ('win'|'loss'|'push'), profit_units.
    """
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["_line"] = _lines(work, line_mode)
    payout = american_to_decimal(juice) - 1.0

    bets: list[dict] = []
    for _, row in work.iterrows():
        line = row["_line"]
        if line is None or pd.isna(line):
            continue
        line = float(line)

        p_over = _prob_over(row, line)
        side = "over" if p_over >= 0.5 else "under"
        model_prob = p_over if side == "over" else 1.0 - p_over
        ev = expected_value(model_prob, juice)
        if ev < min_ev:
            continue

        actual = float(row["actual_value"])
        if actual == line:
            result, profit = "push", 0.0
        else:
            won = (actual > line) if side == "over" else (actual < line)
            result = "win" if won else "loss"
            profit = payout if won else -1.0

        bets.append({
            "player_id":    row.get("player_id"),
            "player_name":  row.get("player_name"),
            "position":     row.get("position"),
            "stat_type":    row.get("stat_type"),
            "line":         line,
            "side":         side,
            "model_prob":   model_prob,
            "ev":           ev,
            "actual_value": actual,
            "result":       result,
            "profit_units": profit,
        })

    return pd.DataFrame(bets)


def summarize_roi(bets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ROI/hit-rate by stat_type × position, plus an 'all' row each."""
    cols = ["stat_type", "position", "n_bets", "wins", "losses",
            "hit_rate", "roi", "avg_ev", "profit_units"]
    if bets.empty:
        return pd.DataFrame(columns=cols)

    def _agg(grp: pd.DataFrame) -> dict:
        decided = grp[grp["result"] != "push"]
        n = len(grp)
        wins = int((grp["result"] == "win").sum())
        losses = int((grp["result"] == "loss").sum())
        staked = float(len(decided))  # 1 unit per non-push bet
        profit = float(grp["profit_units"].sum())
        return {
            "n_bets":       n,
            "wins":         wins,
            "losses":       losses,
            "hit_rate":     (wins / len(decided)) if len(decided) else 0.0,
            "roi":          (profit / staked) if staked else 0.0,
            "avg_ev":       float(grp["ev"].mean()),
            "profit_units": profit,
        }

    rows = []
    for (stat, pos), grp in bets.groupby(["stat_type", "position"]):
        rows.append({"stat_type": stat, "position": pos, **_agg(grp)})
    for stat, grp in bets.groupby("stat_type"):
        rows.append({"stat_type": stat, "position": "all", **_agg(grp)})
    rows.append({"stat_type": "ALL", "position": "all", **_agg(bets)})

    return pd.DataFrame(rows)[cols].sort_values(
        ["stat_type", "position"]
    ).reset_index(drop=True)
