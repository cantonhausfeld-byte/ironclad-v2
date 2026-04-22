"""Aggregate simulation draws into summary statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DrawRecord:
    home_score: int
    away_score: int
    home_pass_yards: float
    away_pass_yards: float
    home_rush_yards: float
    away_rush_yards: float
    home_pass_att: int
    away_pass_att: int
    player_stats: list[dict] = field(default_factory=list)


class SimulationResult:
    def __init__(
        self,
        home_team: str,
        away_team: str,
        draws: list[DrawRecord],
    ) -> None:
        self.home_team = home_team
        self.away_team = away_team
        self._draws = draws
        self._n = len(draws)
        self._player_df = self._build_player_df()

    # ── Aggregated outcomes ───────────────────────────────────────────────────

    def win_probability(self) -> tuple[float, float]:
        home_wins = sum(1 for d in self._draws if d.home_score > d.away_score)
        home_prob = home_wins / self._n
        return round(home_prob, 4), round(1.0 - home_prob, 4)

    def score_summary(self) -> dict[str, Any]:
        home_scores = np.array([d.home_score for d in self._draws])
        away_scores = np.array([d.away_score for d in self._draws])
        return {
            "home_score_mean": float(np.mean(home_scores)),
            "away_score_mean": float(np.mean(away_scores)),
            "home_score_p10": float(np.percentile(home_scores, 10)),
            "home_score_p90": float(np.percentile(home_scores, 90)),
            "away_score_p10": float(np.percentile(away_scores, 10)),
            "away_score_p90": float(np.percentile(away_scores, 90)),
            "total_mean": float(np.mean(home_scores + away_scores)),
            "total_p10": float(np.percentile(home_scores + away_scores, 10)),
            "total_p90": float(np.percentile(home_scores + away_scores, 90)),
        }

    def team_summary(self) -> pd.DataFrame:
        rows = []
        for team, is_home in [(self.home_team, True), (self.away_team, False)]:
            pass_yards = np.array([d.home_pass_yards if is_home else d.away_pass_yards for d in self._draws])
            rush_yards = np.array([d.home_rush_yards if is_home else d.away_rush_yards for d in self._draws])
            scores = np.array([d.home_score if is_home else d.away_score for d in self._draws])
            pass_att = np.array([d.home_pass_att if is_home else d.away_pass_att for d in self._draws])

            for metric, vals in [
                ("points_scored", scores),
                ("pass_yards", pass_yards),
                ("rush_yards", rush_yards),
                ("total_yards", pass_yards + rush_yards),
                ("pass_attempts", pass_att),
            ]:
                rows.append({
                    "team": team,
                    "metric": metric,
                    "p10": float(np.percentile(vals, 10)),
                    "p25": float(np.percentile(vals, 25)),
                    "p50": float(np.percentile(vals, 50)),
                    "p75": float(np.percentile(vals, 75)),
                    "p90": float(np.percentile(vals, 90)),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                })
        return pd.DataFrame(rows)

    def player_summary(self) -> pd.DataFrame:
        if self._player_df.empty:
            return pd.DataFrame()

        metrics = ["targets", "receptions", "rec_yards", "carries", "rush_yards",
                   "pass_attempts", "completions", "pass_yards", "tds"]
        id_cols = ["player_id", "player_name", "team", "position", "is_home"]

        rows = []
        for (pid, pname, team, pos, is_home), grp in self._player_df.groupby(id_cols):
            for metric in metrics:
                if metric not in grp.columns:
                    continue
                vals = grp[metric].values.astype(float)
                if vals.sum() == 0 and metric not in ("tds",):
                    continue
                rows.append({
                    "player_id": pid,
                    "player_name": pname,
                    "team": team,
                    "position": pos,
                    "is_home": is_home,
                    "metric": metric,
                    "p10": float(np.percentile(vals, 10)),
                    "p25": float(np.percentile(vals, 25)),
                    "p50": float(np.percentile(vals, 50)),
                    "p75": float(np.percentile(vals, 75)),
                    "p90": float(np.percentile(vals, 90)),
                    "mean": float(np.mean(vals)),
                })
        return pd.DataFrame(rows)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_player_df(self) -> pd.DataFrame:
        records = []
        for draw in self._draws:
            for stat in draw.player_stats:
                records.append(stat)
        return pd.DataFrame(records) if records else pd.DataFrame()
