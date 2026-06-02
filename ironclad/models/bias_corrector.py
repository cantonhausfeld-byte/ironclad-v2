"""Per-team margin bias correction learned from backtest residuals."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Shrinkage factor: 0.5 means we apply half the learned bias correction.
# Prevents over-fitting on small per-team sample sizes (≈50 home games / team).
_SHRINKAGE = 0.5


class TeamBiasCorrector:
    """Corrects systematic per-team margin prediction errors from backtest data.

    Usage:
        corrector = TeamBiasCorrector()
        corrector.fit(backtest_df)               # needs home_team, away_team,
                                                  # home_margin_pred, home_margin_actual
        corrected = corrector.correct("KC", "LAC", model_margin)
    """

    name = "team_bias"
    version = "v1.0"

    def __init__(self) -> None:
        self._team_bias: dict[str, float] = {}
        self._fitted = False

    def fit(self, backtest_df: pd.DataFrame) -> None:
        """Compute per-team residuals from walk-forward backtest predictions.

        margin_error = actual_home_margin − predicted_home_margin.
        A positive bias for team X means X outperforms predictions when home;
        a negative bias means X underperforms.
        We use the same bias dictionary for both home and away corrections:
          - when X is home: add X's bias
          - when X is away: subtract X's bias (they're still X; perspective flips)
        """
        required = {"home_team", "away_team", "home_margin_pred", "home_margin_actual"}
        if not required.issubset(backtest_df.columns):
            missing = required - set(backtest_df.columns)
            logger.warning("TeamBiasCorrector.fit: missing columns %s, skipping", missing)
            return

        df = backtest_df.dropna(subset=["home_margin_actual", "home_margin_pred"]).copy()
        if df.empty:
            logger.warning("TeamBiasCorrector.fit: no complete rows, skipping")
            return

        df["margin_error"] = df["home_margin_actual"] - df["home_margin_pred"]

        # Average residual per team when playing at home.  Sufficient for a first
        # correction; separating home vs. away roles adds complexity for little gain
        # given ~50 samples per team.
        self._team_bias = df.groupby("home_team")["margin_error"].mean().to_dict()
        self._fitted = True

        top = sorted(self._team_bias.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        logger.info(
            "TeamBiasCorrector fitted on %d games (%d teams). "
            "Largest biases: %s",
            len(df), len(self._team_bias),
            [(t, round(b, 2)) for t, b in top],
        )

    def correct(self, home_team: str, away_team: str, margin_mean: float) -> float:
        """Return bias-corrected home_margin_mean.

        Applies half the learned bias (shrinkage=0.5) to avoid over-correcting
        on small samples.  Gracefully returns margin_mean unchanged when no
        backtest data is available.
        """
        if not self._fitted:
            return margin_mean
        home_adj = self._team_bias.get(home_team, 0.0) * _SHRINKAGE
        away_adj = self._team_bias.get(away_team, 0.0) * _SHRINKAGE
        # When the away team has a positive home-bias it means they're better
        # than average — that should lower the home team's advantage.
        return margin_mean + home_adj - away_adj

    def team_biases(self) -> dict[str, float]:
        """Return raw (unshrunk) per-team bias values, sorted by magnitude."""
        return dict(sorted(self._team_bias.items(), key=lambda kv: abs(kv[1]), reverse=True))
