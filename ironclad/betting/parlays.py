"""Same-game parlay probability via Monte Carlo joint distribution."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from ironclad.betting.ev import american_to_prob
from ironclad.betting.props import _ANYTIME_TD, _OU_STATS
from ironclad.simulation.results import SimulationResult

logger = logging.getLogger(__name__)

_VALID_DIRECTIONS = {"over", "under", "yes"}


@dataclass
class ParlayLeg:
    player_id: str
    stat_type: str
    direction: str          # "over", "under", or "yes" (anytime_td)
    line: float | None      # None for anytime_td
    market_odds: int | None = None

    def validate(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}; got '{self.direction}'")
        if self.stat_type == _ANYTIME_TD:
            if self.direction != "yes":
                raise ValueError("anytime_td legs must have direction='yes'")
        elif self.stat_type in _OU_STATS:
            if self.direction not in ("over", "under"):
                raise ValueError(f"{self.stat_type} legs must have direction 'over' or 'under'")
            if self.line is None:
                raise ValueError(f"{self.stat_type} leg requires a numeric line")
        else:
            raise ValueError(f"Unsupported stat_type: '{self.stat_type}'")


def load_parlay_legs(path: Path) -> list[ParlayLeg]:
    """Load parlay legs from CSV.

    Required columns: player_id, stat_type, direction, line
    Optional column:  market_odds
    """
    legs: list[ParlayLeg] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_val = row.get("line", "").strip()
            odds_val = row.get("market_odds", "").strip()
            leg = ParlayLeg(
                player_id=row["player_id"].strip(),
                stat_type=row["stat_type"].strip(),
                direction=row["direction"].strip().lower(),
                line=float(line_val) if line_val else None,
                market_odds=int(odds_val) if odds_val else None,
            )
            leg.validate()
            legs.append(leg)
    return legs


def _leg_hit(stats: dict, leg: ParlayLeg) -> bool:
    """Return True if the player's stats satisfy the leg condition."""
    if not stats or not stats.get("played", True):
        return False
    if leg.stat_type == _ANYTIME_TD:
        return float(stats.get("tds", 0)) >= 1.0
    val = stats.get(leg.stat_type)
    if val is None:
        return False
    val = float(val)
    if leg.direction == "over":
        return val > leg.line  # type: ignore[operator]
    return val < leg.line      # type: ignore[operator]


def parlay_probability(sim: SimulationResult, legs: list[ParlayLeg]) -> float:
    """Fraction of simulation draws where every leg is simultaneously satisfied.

    Uses the joint distribution directly — player outcomes within the same draw
    share the game-quality correlation factor, so WR/TE/QB stats on the same
    team are positively correlated. Market pricing assumes independence; this
    function does not.
    """
    if not legs:
        raise ValueError("legs list is empty")
    n_hit = 0
    for draw in sim._draws:
        stats_by_player: dict[str, dict] = {s["player_id"]: s for s in draw.player_stats}
        if all(_leg_hit(stats_by_player.get(leg.player_id, {}), leg) for leg in legs):
            n_hit += 1
    return n_hit / len(sim._draws)


def market_independence_prob(legs: list[ParlayLeg]) -> float | None:
    """What the market implicitly prices assuming all legs are independent.

    Multiplies the individual vig-inclusive implied probabilities. Returns None
    if any leg is missing market_odds.
    """
    if any(leg.market_odds is None for leg in legs):
        return None
    result = 1.0
    for leg in legs:
        result *= american_to_prob(leg.market_odds)  # type: ignore[arg-type]
    return result
