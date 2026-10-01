# Game model baseline — 2026-10-01

Walk-forward over **2019–2025**:
- Each test season T is predicted by a model fit on 2016…T−2 and Platt-calibrated on held-out T−1.
- 1,871 games are predicted.
- Data is the full 2016–2026 rebuild with every fix through `ab5296e` (cutoff-safe, depth charts present).

Reproduce with `ironclad backtest --test-seasons 2019-2025`, then `ironclad dashboard` and `ironclad profitability`.

## Headline

| Model | Brier | Log-loss | Margin MAE | ATS picks | O/U picks | Bet ROI* |
|---|---|---|---|---|---|---|
| Vegas (closing consensus) | **0.2107** | — | — | — | — | — |
| XGB, 24 team features (pre-Elo baseline) | 0.2410 | 0.675 | 11.12 | 48.6% | 51.9% | −2.5% (3,096 bets) |
| XGB, 24 features + `elo_diff` (shipped) | 0.2331 | 0.659 | 10.83 | 48.4% | 50.1% | −5.4% (3,032 bets) |

\*`ironclad profitability` defaults: min edge 3%, min total diff 1.5 pts, −110 juice. Breakeven is 52.4%.
With ~1,500 spread bets, one standard error of ROI is about ±2.5 pts, so the two ROI figures are not
distinguishable.

## Offline comparison (same folds, uncalibrated, not shipped)

| Model | Brier | Log-loss |
|---|---|---|
| Logistic regression, 24 features | 0.2331 | 0.659 |
| Logistic regression, 24 features + Elo | 0.2243 | 0.641 |
| **Logistic regression, Elo only** | **0.2232** | **0.638** |
| Elo formula alone (no fitting) | 0.2225 | 0.637 |
| XGB (150 trees, depth 2) + Elo | 0.2251 | 0.642 |

## What this says

1. **The model does not beat the market.** It sits about 0.012–0.03 Brier behind Vegas, and ATS and O/U picks are
   below breakeven. Treat no game-line edge as actionable yet (roadmap 4.3).
2. **The current XGB classifier overfits.** 300 depth-4 trees lose to a plain logistic regression on the same
   features, and to a one-line Elo formula.
3. **The 24 rolling team features add ~nothing on top of Elo** at the game level. Phase 2's remaining feature
   ports (2.5–2.7) should be judged against an Elo-inclusive model, and a simpler, regularized model is
   likely the bigger win.
4. In-sample residual std understated simulation spread (margin 9.0 vs 13.4 out of sample; total 9.3 vs 14.1).
   This was fixed in the same commit as this file by estimating it out-of-fold.

## Bugs found while producing this baseline

- Backtest calibration was fit in-sample (Brier 0.327 → 0.241 after the fix).
- Dashboard ATS/O-U ignored the model and had the spread sign backwards. It showed a fake 55.9% ATS.
- Stub models inverted the expected margin.
