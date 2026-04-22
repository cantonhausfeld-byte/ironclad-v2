"""Feature engineering utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_mean(
    df: pd.DataFrame,
    value_col: str,
    sort_col: str = "gameday",
    n: int = 4,
    min_periods: int = 1,
) -> pd.Series:
    """Rolling mean over last n rows, sorted by sort_col."""
    return (
        df.sort_values(sort_col)[value_col]
        .rolling(n, min_periods=min_periods)
        .mean()
        .shift(1)  # exclude current game
    )


def safe_divide(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
    default: float = 0.0,
) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (denominator == 0) | pd.isna(denominator),
            default,
            numerator / denominator,
        )
    return pd.Series(result, index=getattr(numerator, "index", None))


def completeness_score(series: pd.Series) -> float:
    """Fraction of non-null values in a feature row."""
    return float(series.notna().mean())
