"""Text-based backtest metrics dashboard."""
from __future__ import annotations

import pandas as pd

from ironclad.eval import metrics as m


def print_dashboard(predictions_df: pd.DataFrame) -> None:
    """Print a full backtest metrics summary to stdout."""
    if predictions_df.empty:
        print("No backtest predictions found. Run: ironclad backtest")
        return

    run_id = predictions_df["backtest_run_id"].iloc[0]
    seasons = sorted(predictions_df["season"].unique())

    print(f"\n{'='*60}")
    print(f"  ironclad-v2 Backtest Dashboard")
    print(f"  Run: {run_id}")
    print(f"  Seasons tested: {seasons[0]}–{seasons[-1]}")
    print(f"{'='*60}\n")

    # ── Overall metrics ──────────────────────────────────────────────────────
    summary = m.summary_table(predictions_df)
    print("OVERALL METRICS")
    print(f"  Games:          {summary.get('n_games', 0):>6}")
    print(f"  Brier score:    {summary.get('brier_score', float('nan')):>8.4f}  (lower = better; market ≈ 0.235)")
    if summary.get("vegas_brier_baseline"):
        print(f"  Vegas baseline: {summary['vegas_brier_baseline']:>8.4f}")
    print(f"  Log-loss:       {summary.get('log_loss', float('nan')):>8.4f}")
    print(f"  Margin MAE:     {summary.get('margin_mae', float('nan')):>7.2f} pts")
    print(f"  Total MAE:      {summary.get('total_mae', float('nan')):>7.2f} pts")

    ats = summary.get("ats")
    if ats:
        ats_pct = f"{ats['pct']:.1%}" if ats["pct"] == ats["pct"] else "N/A"
        print(f"\n  ATS record:     {ats['wins']}-{ats['losses']}-{ats['pushes']}  ({ats_pct})")
    ou = summary.get("ou")
    if ou:
        ou_pct = f"{ou['pct']:.1%}" if ou["pct"] == ou["pct"] else "N/A"
        print(f"  O/U record:     {ou['overs']}o-{ou['unders']}u-{ou['pushes']}p  ({ou_pct})")

    # ── Per-season breakdown ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("METRICS BY SEASON")
    header = f"  {'Season':>6}  {'Games':>5}  {'Brier':>7}  {'MarMAE':>7}  {'TotMAE':>7}  {'ATS':>10}  {'O/U':>10}"
    print(header)
    print(f"  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*10}")

    for season in seasons:
        sg = predictions_df[predictions_df["season"] == season]
        ss = m.summary_table(sg)
        n = ss.get("n_games", 0)
        brier = f"{ss.get('brier_score', float('nan')):.4f}"
        mar_mae = f"{ss.get('margin_mae', float('nan')):.2f}"
        tot_mae = f"{ss.get('total_mae', float('nan')):.2f}"
        ats_s = ss.get("ats")
        ou_s = ss.get("ou")
        ats_str = f"{ats_s['wins']}-{ats_s['losses']}" if ats_s else " —"
        ou_str = f"{ou_s['overs']}o-{ou_s['unders']}u" if ou_s else " —"
        print(f"  {season:>6}  {n:>5}  {brier:>7}  {mar_mae:>7}  {tot_mae:>7}  {ats_str:>10}  {ou_str:>10}")

    # ── Calibration curve ────────────────────────────────────────────────────
    valid_cal = predictions_df.dropna(subset=["home_win_prob", "home_win_actual"])
    if not valid_cal.empty:
        probs = valid_cal["home_win_prob"].to_numpy(float)
        actuals = valid_cal["home_win_actual"].astype(int).to_numpy()
        cal = m.calibration_curve(probs, actuals, n_bins=10)
        if not cal.empty:
            print(f"\n{'─'*60}")
            print("CALIBRATION (predicted prob → actual win rate)")
            print(f"  {'Pred Prob':>10}  {'Obs Rate':>10}  {'N':>6}  Δ")
            for _, r in cal.iterrows():
                delta = r["observed_rate"] - r["predicted_prob"]
                bar = "▲" if delta > 0.03 else ("▼" if delta < -0.03 else "≈")
                print(f"  {r['predicted_prob']:>10.3f}  {r['observed_rate']:>10.3f}  {int(r['count']):>6}  {bar}")

    # ── Most-biased teams ────────────────────────────────────────────────────
    team_bias = m.by_team_bias(predictions_df)
    if not team_bias.empty:
        print(f"\n{'─'*60}")
        print("MOST-BIASED TEAMS (top 10 by margin MAE)")
        print(f"  {'Team':>5}  {'Games':>5}  {'MAE':>6}  {'Bias':>6}  direction")
        for _, r in team_bias.head(10).iterrows():
            direction = "overestimated" if r["margin_bias"] > 0 else "underestimated"
            print(f"  {r['team']:>5}  {int(r['n_games']):>5}  {r['margin_mae']:>6.2f}  {r['margin_bias']:>+6.2f}  {direction}")

    # ── Early vs late season ─────────────────────────────────────────────────
    week_bias = m.by_week_bias(predictions_df)
    if not week_bias.empty:
        early = week_bias[week_bias["week"] <= 4]["margin_mae"].mean()
        late = week_bias[week_bias["week"] >= 14]["margin_mae"].mean()
        print(f"\n{'─'*60}")
        print("EARLY vs LATE SEASON ACCURACY")
        print(f"  Weeks 1-4  margin MAE: {early:.2f} pts")
        print(f"  Weeks 14+  margin MAE: {late:.2f} pts")
        diff = early - late
        print(f"  Early-season penalty: {diff:+.2f} pts  {'(expected — small sample)' if diff > 0 else ''}")

    print(f"\n{'='*60}\n")
