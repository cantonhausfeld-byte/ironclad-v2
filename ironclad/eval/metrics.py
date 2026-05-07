"""Evaluation metrics for model backtest predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error


def brier_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    return float(brier_score_loss(actuals, probs))


def log_loss_score(probs: np.ndarray, actuals: np.ndarray) -> float:
    return float(log_loss(actuals, np.clip(probs, 1e-7, 1 - 1e-7)))


def calibration_curve(
    probs: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return a reliability diagram table: predicted prob vs. observed win rate per bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(probs, bins[1:-1])
    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": round(float((bins[i] + bins[i + 1]) / 2), 3),
            "predicted_prob": round(float(probs[mask].mean()), 3),
            "observed_rate": round(float(actuals[mask].mean()), 3),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def ats_record(
    margin_pred: np.ndarray,
    spread_line: np.ndarray,
    actual_margin: np.ndarray,
) -> dict:
    """
    ATS (Against The Spread) record from the home team's perspective.
    spread_line: Vegas home spread (e.g. -3.0 means home favored by 3).
    Home covers when actual_margin > -spread_line.
    """
    valid = ~(np.isnan(spread_line) | np.isnan(actual_margin))
    s = spread_line[valid]
    am = actual_margin[valid]
    push = am == -s
    cover = am > -s
    wins = int(cover[~push].sum())
    losses = int((~cover[~push]).sum())
    pushes = int(push.sum())
    pct = wins / (wins + losses) if (wins + losses) > 0 else float("nan")
    return {"wins": wins, "losses": losses, "pushes": pushes, "pct": round(pct, 4), "n": int(valid.sum())}


def ou_record(
    total_pred: np.ndarray,
    total_line: np.ndarray,
    actual_total: np.ndarray,
) -> dict:
    """Over/under record. over when actual_total > total_line."""
    valid = ~(np.isnan(total_line) | np.isnan(actual_total))
    tl = total_line[valid]
    at = actual_total[valid]
    push = at == tl
    over = at > tl
    overs = int(over[~push].sum())
    unders = int((~over[~push]).sum())
    pushes = int(push.sum())
    pct = overs / (overs + unders) if (overs + unders) > 0 else float("nan")
    return {"overs": overs, "unders": unders, "pushes": pushes, "pct": round(pct, 4), "n": int(valid.sum())}


def by_team_bias(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Per-team margin MAE and bias (positive = model overestimates home margin)."""
    rows = []
    for team, g in predictions_df.groupby("home_team"):
        valid = g.dropna(subset=["home_margin_actual", "home_margin_pred"])
        if len(valid) < 3:
            continue
        errors = valid["home_margin_pred"] - valid["home_margin_actual"]
        rows.append({
            "team": team,
            "n_games": len(valid),
            "margin_mae": round(float(mean_absolute_error(valid["home_margin_actual"], valid["home_margin_pred"])), 2),
            "margin_bias": round(float(errors.mean()), 2),
        })
    return pd.DataFrame(rows).sort_values("margin_mae", ascending=False).reset_index(drop=True)


def by_week_bias(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Per-week margin MAE — checks if early-season predictions are systematically worse."""
    rows = []
    for week, g in predictions_df.groupby("week"):
        valid = g.dropna(subset=["home_margin_actual", "home_margin_pred"])
        if len(valid) < 2:
            continue
        rows.append({
            "week": int(week),
            "n_games": len(valid),
            "margin_mae": round(float(mean_absolute_error(valid["home_margin_actual"], valid["home_margin_pred"])), 2),
            "margin_bias": round(float((valid["home_margin_pred"] - valid["home_margin_actual"]).mean()), 2),
        })
    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def summary_table(predictions_df: pd.DataFrame) -> dict:
    """Compute the full suite of metrics from a backtest predictions DataFrame."""
    df = predictions_df.dropna(subset=["home_win_prob", "home_win_actual"])
    if df.empty:
        return {}

    probs = df["home_win_prob"].to_numpy(dtype=float)
    actuals = df["home_win_actual"].astype(int).to_numpy()

    metrics: dict = {
        "n_games": len(df),
        "brier_score": round(brier_score(probs, actuals), 4),
        "log_loss": round(log_loss_score(probs, actuals), 4),
        "vegas_brier_baseline": _vegas_brier(df),
    }

    # Margin MAE
    margin_valid = df.dropna(subset=["home_margin_pred", "home_margin_actual"])
    if not margin_valid.empty:
        metrics["margin_mae"] = round(float(mean_absolute_error(
            margin_valid["home_margin_actual"], margin_valid["home_margin_pred"]
        )), 2)

    # Total MAE
    total_valid = df.dropna(subset=["total_pred", "total_actual"])
    if not total_valid.empty:
        metrics["total_mae"] = round(float(mean_absolute_error(
            total_valid["total_actual"], total_valid["total_pred"]
        )), 2)

    # ATS
    ats_valid = df.dropna(subset=["vegas_spread", "home_margin_actual"])
    if not ats_valid.empty:
        metrics["ats"] = ats_record(
            ats_valid["home_margin_pred"].to_numpy(float),
            ats_valid["vegas_spread"].to_numpy(float),
            ats_valid["home_margin_actual"].to_numpy(float),
        )

    # Over/Under
    ou_valid = df.dropna(subset=["vegas_total", "total_actual"])
    if not ou_valid.empty:
        metrics["ou"] = ou_record(
            ou_valid["total_pred"].to_numpy(float),
            ou_valid["vegas_total"].to_numpy(float),
            ou_valid["total_actual"].to_numpy(float),
        )

    return metrics


def _vegas_brier(df: pd.DataFrame) -> float | None:
    """Brier score using Vegas-implied home win probability as the prediction."""
    col = "home_win_prob_vegas"
    if col not in df.columns:
        return None
    valid = df.dropna(subset=[col, "home_win_actual"])
    if valid.empty:
        return None
    return round(brier_score(valid[col].to_numpy(float), valid["home_win_actual"].astype(int).to_numpy()), 4)
