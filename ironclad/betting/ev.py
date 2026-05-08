"""Odds conversion and expected-value math."""
from __future__ import annotations


def american_to_decimal(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def american_to_prob(odds: int) -> float:
    """Implied probability (with vig) from American odds."""
    if odds == 0:
        raise ValueError("American odds cannot be zero")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def decimal_to_prob(odds: float) -> float:
    if odds <= 1.0:
        raise ValueError(f"Decimal odds must be > 1.0; got {odds}")
    return 1.0 / odds


def no_vig_prob(over_odds: int, under_odds: int) -> tuple[float, float]:
    """Strip the bookmaker margin from a two-sided market.

    Returns (p_over, p_under) summing to 1.0.
    """
    p_over_raw = american_to_prob(over_odds)
    p_under_raw = american_to_prob(under_odds)
    total = p_over_raw + p_under_raw
    if total <= 0:
        raise ValueError("Implied probabilities sum to zero")
    return p_over_raw / total, p_under_raw / total


def expected_value(model_prob: float, american_odds: int, stake: float = 1.0) -> float:
    """Expected profit per unit staked at the given American odds.

    EV = p_win * payout - p_lose * stake
    where payout = stake * (decimal_odds - 1).
    """
    decimal = american_to_decimal(american_odds)
    payout = stake * (decimal - 1.0)
    return model_prob * payout - (1.0 - model_prob) * stake


def edge(model_prob: float, american_odds: int) -> float:
    """Probabilistic edge: model_prob - market_implied_prob (vig included)."""
    return model_prob - american_to_prob(american_odds)
