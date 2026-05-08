"""Kelly criterion bet sizing."""
from __future__ import annotations

from ironclad.betting.ev import american_to_decimal


def kelly_fraction(model_prob: float, american_odds: int) -> float:
    """Full-Kelly fraction of bankroll to wager.

    f* = (b * p - q) / b
        b = decimal odds - 1 (net payout per unit staked)
        p = model win probability
        q = 1 - p
    Returns 0 when the bet has no edge (or negative edge).
    """
    b = american_to_decimal(american_odds) - 1.0
    if b <= 0.0:
        return 0.0
    q = 1.0 - model_prob
    f = (b * model_prob - q) / b
    return max(0.0, f)


def fractional_kelly(model_prob: float, american_odds: int, fraction: float = 0.25) -> float:
    """Fractional Kelly — defaults to f/4 to absorb model-probability uncertainty.

    Full Kelly maximizes log-bankroll growth assuming the win probability is
    known exactly. Real model probabilities have estimation error, and full
    Kelly is brutally punished by overestimates. f/4 is the standard
    professional choice for sports betting.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    return kelly_fraction(model_prob, american_odds) * fraction


class KellyBankroll:
    """Track a bankroll and size bets at a fixed Kelly fraction."""

    def __init__(self, bankroll: float, fraction: float = 0.25) -> None:
        if bankroll <= 0:
            raise ValueError(f"bankroll must be positive; got {bankroll}")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
        self.bankroll = float(bankroll)
        self.fraction = fraction

    def stake(self, model_prob: float, american_odds: int) -> float:
        f = fractional_kelly(model_prob, american_odds, self.fraction)
        return self.bankroll * f

    def settle(self, won: bool, stake: float, american_odds: int) -> float:
        if stake < 0 or stake > self.bankroll:
            raise ValueError(f"invalid stake {stake} (bankroll={self.bankroll})")
        if won:
            payout = stake * (american_to_decimal(american_odds) - 1.0)
            self.bankroll += payout
            return payout
        self.bankroll -= stake
        return -stake
