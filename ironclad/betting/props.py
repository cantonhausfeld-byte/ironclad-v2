"""Compare Monte Carlo prop distributions to market lines for +EV identification."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ironclad.betting.ev import (
    american_to_prob,
    edge as ev_edge,
    expected_value,
    no_vig_prob,
)
from ironclad.betting.kelly import fractional_kelly
from ironclad.simulation.results import SimulationResult

logger = logging.getLogger(__name__)


# Stat types where the model prob is P(stat > line). "anytime_td" is special:
# single-sided P(tds >= 1).
_OU_STATS = {
    "rec_yards", "rush_yards", "pass_yards",
    "receptions", "targets", "carries",
    "completions", "pass_attempts", "tds",
}
_ANYTIME_TD = "anytime_td"


@dataclass
class PropLine:
    player_id: str
    stat_type: str
    line: float | None
    over_odds: int | None
    under_odds: int | None = None

    def validate(self) -> None:
        if self.stat_type == _ANYTIME_TD:
            if self.over_odds is None:
                raise ValueError(f"{self.player_id} anytime_td requires over_odds (the 'yes' price)")
        elif self.stat_type in _OU_STATS:
            if self.line is None:
                raise ValueError(f"{self.player_id} {self.stat_type} requires a line")
            if self.over_odds is None and self.under_odds is None:
                raise ValueError(f"{self.player_id} {self.stat_type} requires at least one of over/under odds")
        else:
            raise ValueError(f"Unsupported stat_type: {self.stat_type}")


def load_prop_lines(path: Path) -> list[PropLine]:
    """Load prop lines from CSV.

    Required columns: player_id, stat_type, line, over_odds, under_odds.
    Empty cells are interpreted as None. anytime_td rows leave line and
    under_odds blank.
    """
    lines: list[PropLine] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_val = row.get("line", "").strip()
            over_val = row.get("over_odds", "").strip()
            under_val = row.get("under_odds", "").strip()
            pl = PropLine(
                player_id=row["player_id"].strip(),
                stat_type=row["stat_type"].strip(),
                line=float(line_val) if line_val else None,
                over_odds=int(over_val) if over_val else None,
                under_odds=int(under_val) if under_val else None,
            )
            pl.validate()
            lines.append(pl)
    return lines


class PropAnalyzer:
    def __init__(self, kelly_fraction: float = 0.25) -> None:
        self.kelly_fraction = kelly_fraction

    def analyze(
        self,
        sim: SimulationResult,
        prop_lines: list[PropLine],
    ) -> pd.DataFrame:
        df = sim._player_df
        if df.empty:
            logger.warning("Simulation produced no player draws")
            return pd.DataFrame()

        rows: list[dict] = []
        for pl in prop_lines:
            player_rows = df[df["player_id"] == pl.player_id]
            if player_rows.empty:
                logger.warning("No simulation draws for player_id=%s", pl.player_id)
                continue

            meta = player_rows.iloc[0]
            base = {
                "player_id":   pl.player_id,
                "player_name": meta.get("player_name", ""),
                "team":        meta.get("team", ""),
                "position":    meta.get("position", ""),
                "stat_type":   pl.stat_type,
                "market_line": pl.line,
            }

            if pl.stat_type == _ANYTIME_TD:
                rows.append(self._anytime_td_row(player_rows, pl, base))
            else:
                rows.append(self._ou_row(player_rows, pl, base))

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ou_row(self, player_rows: pd.DataFrame, pl: PropLine, base: dict) -> dict:
        if pl.stat_type not in player_rows.columns:
            logger.warning("Stat %s not in simulation columns for %s", pl.stat_type, pl.player_id)
            return {**base, "side": "skip", "reason": f"missing stat column {pl.stat_type}"}

        vals = player_rows[pl.stat_type].to_numpy(dtype=float)
        # Strict > line for the over (matches DraftKings push handling on .5 lines).
        p_over_model = float(np.mean(vals > pl.line))
        p_under_model = float(np.mean(vals < pl.line))

        if pl.over_odds is not None and pl.under_odds is not None:
            mp_over, mp_under = no_vig_prob(pl.over_odds, pl.under_odds)
        else:
            mp_over = american_to_prob(pl.over_odds) if pl.over_odds is not None else None
            mp_under = american_to_prob(pl.under_odds) if pl.under_odds is not None else None

        ev_over = expected_value(p_over_model, pl.over_odds) if pl.over_odds is not None else None
        ev_under = expected_value(p_under_model, pl.under_odds) if pl.under_odds is not None else None

        # Pick the better side; ties broken by EV.
        candidates = []
        if pl.over_odds is not None:
            candidates.append(("over", p_over_model, mp_over, pl.over_odds, ev_over))
        if pl.under_odds is not None:
            candidates.append(("under", p_under_model, mp_under, pl.under_odds, ev_under))
        side, model_prob, market_prob, odds, ev = max(candidates, key=lambda c: c[4])

        return {
            **base,
            "side":         side,
            "odds":         odds,
            "model_prob":   round(model_prob, 4),
            "market_prob":  round(market_prob, 4) if market_prob is not None else None,
            "edge":         round(ev_edge(model_prob, odds), 4),
            "ev":           round(ev, 4),
            "kelly":        round(fractional_kelly(model_prob, odds, self.kelly_fraction), 4),
            "n_draws":      len(vals),
            "model_p10":    float(np.percentile(vals, 10)),
            "model_p50":    float(np.percentile(vals, 50)),
            "model_p90":    float(np.percentile(vals, 90)),
        }

    def _anytime_td_row(self, player_rows: pd.DataFrame, pl: PropLine, base: dict) -> dict:
        if "tds" not in player_rows.columns:
            return {**base, "side": "skip", "reason": "missing tds column"}
        tds = player_rows["tds"].to_numpy(dtype=float)
        p_yes_model = float(np.mean(tds >= 1.0))
        market_prob = american_to_prob(pl.over_odds)
        ev = expected_value(p_yes_model, pl.over_odds)
        return {
            **base,
            "side":         "yes",
            "odds":         pl.over_odds,
            "model_prob":   round(p_yes_model, 4),
            "market_prob":  round(market_prob, 4),
            "edge":         round(ev_edge(p_yes_model, pl.over_odds), 4),
            "ev":           round(ev, 4),
            "kelly":        round(fractional_kelly(p_yes_model, pl.over_odds, self.kelly_fraction), 4),
            "n_draws":      len(tds),
            "model_p10":    float(np.percentile(tds, 10)),
            "model_p50":    float(np.percentile(tds, 50)),
            "model_p90":    float(np.percentile(tds, 90)),
        }
