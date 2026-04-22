"""Build gold.team_game_features from silver tables."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from ironclad.config import LEAGUE_PRIORS, FEATURE_VERSION, ROLLING_WINDOW
from ironclad.features.snapshot import FeatureSnapshot
from ironclad.features.utils import safe_divide, completeness_score
from ironclad.store.writer import GoldWriter

logger = logging.getLogger(__name__)


class TeamFeatureBuilder:
    def __init__(self, conn=None) -> None:
        from ironclad.store.connection import get_connection
        self._conn = conn or get_connection()
        self._writer = GoldWriter(self._conn)

    def build_for_game(self, game_id: str, cutoff_ts: datetime) -> pd.DataFrame:
        snap = FeatureSnapshot(cutoff_ts, self._conn)
        game = snap.game_row(game_id)
        if game is None:
            logger.warning("Game %s not found in silver.games", game_id)
            return pd.DataFrame()

        rows = []
        for team, opp, is_home in [
            (game["home_team"], game["away_team"], True),
            (game["away_team"], game["home_team"], False),
        ]:
            row = self._build_team_row(team, opp, is_home, game, snap)
            rows.append(row)

        df = pd.DataFrame(rows)
        self._writer.write_team_features(df)
        return df

    def build_for_season(self, season: int, cutoff_ts: datetime | None = None) -> int:
        games = self._conn.execute(
            "SELECT game_id, gameday, gametime_local FROM silver.games WHERE season = ?",
            [season],
        ).df()
        total = 0
        for _, g in games.iterrows():
            try:
                game_cutoff = _parse_kickoff(g["gameday"], g.get("gametime_local")) if cutoff_ts is None else cutoff_ts
                self.build_for_game(g["game_id"], game_cutoff)
                total += 1
            except Exception as exc:
                logger.warning("Features failed for %s: %s", g["game_id"], exc)
        return total

    def _build_team_row(
        self,
        team: str,
        opponent: str,
        is_home: bool,
        game: pd.Series,
        snap: FeatureSnapshot,
    ) -> dict:
        season = int(game["season"])
        week = int(game["week"])

        recent = snap.team_recent_games(team, n=ROLLING_WINDOW)
        season_games = snap.team_season_games(team, season)

        # ── Offense rolling L4 ────────────────────────────────────────────────
        def off(col, default_key=None, default=None):
            d = default if default is not None else LEAGUE_PRIORS.get(default_key or col, 0.0)
            if recent.empty or col not in recent.columns:
                return d
            vals = recent[col].dropna()
            return float(vals.mean()) if len(vals) else d

        # ── Defense: opponent stats FROM recent opponents against this team ───
        def def_from_opp(col, default_key=None, default=None):
            d = default if default is not None else LEAGUE_PRIORS.get(default_key or col, 0.0)
            if recent.empty:
                return d
            all_stats = snap.reader.read_as_of("silver.team_game_stats")
            opp_rows = all_stats[
                all_stats["game_id"].isin(recent["game_id"]) &
                (all_stats["opponent"] == team)
            ]
            if opp_rows.empty or col not in opp_rows.columns:
                return d
            vals = opp_rows[col].dropna()
            return float(vals.mean()) if len(vals) else d

        # Season-to-date
        def std(col, default_key=None, default=None):
            d = default if default is not None else LEAGUE_PRIORS.get(default_key or col, 0.0)
            if season_games.empty or col not in season_games.columns:
                return d
            vals = season_games[col].dropna()
            return float(vals.mean()) if len(vals) else d

        rest_days = _compute_rest_days(team, game, snap)

        # Odds context
        implied_total = game.get("total_consensus")
        spread = game.get("spread_consensus")
        home_win_prob = game.get("home_ml_implied")

        # Data completeness: how many L4 games do we have?
        n_games = len(recent)
        completeness = min(1.0, n_games / ROLLING_WINDOW)

        # Sack rate: sacks_allowed / pass_attempts (offense perspective)
        off_sack_rate = safe_divide(
            pd.Series([off("sacks_allowed", default=0.0)]),
            pd.Series([off("pass_attempts", default=30.0)]),
            LEAGUE_PRIORS["sack_rate"],
        ).iloc[0]

        feature_row = {
            "game_id": game["game_id"],
            "season": season,
            "week": week,
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "cutoff_ts": snap.cutoff_ts,
            "feature_version": FEATURE_VERSION,
            # Offense L4
            "off_epa_per_play_l4":    off("epa_per_play",    "off_epa_per_play"),
            "off_pass_epa_l4":        off("epa_pass",        "off_epa_per_play"),
            "off_rush_epa_l4":        off("epa_rush",        "off_epa_per_play"),
            "off_pass_rate_l4":       off("pass_rate",       "pass_rate"),
            "off_yards_per_play_l4":  safe_divide(
                pd.Series([off("total_yards", default=None)]),
                pd.Series([off("plays_total", default=None)]),
                LEAGUE_PRIORS["yards_per_play"],
            ).iloc[0],
            "off_success_rate_l4":    off("success_rate",    "success_rate"),
            "off_points_per_game_l4": off("points_scored",   "points_per_game"),
            # Defense L4 (from opponent's stats against this team)
            "def_epa_per_play_l4":    def_from_opp("epa_per_play",  "def_epa_per_play"),
            "def_pass_epa_l4":        def_from_opp("epa_pass",      "def_epa_per_play"),
            "def_rush_epa_l4":        def_from_opp("epa_rush",      "def_epa_per_play"),
            "def_yards_allowed_l4":   def_from_opp("total_yards",   default=330.0),
            "def_success_rate_l4":    def_from_opp("success_rate",  "success_rate"),
            "def_points_allowed_l4":  off("points_allowed",         "points_per_game"),
            "def_sack_rate_l4":       off_sack_rate,
            # Season-to-date
            "off_epa_per_play_std":   std("epa_per_play",   "off_epa_per_play"),
            "def_epa_per_play_std":   def_from_opp("epa_per_play",  "def_epa_per_play"),
            # Context
            "rest_days":              rest_days,
            "is_divisional":          None,
            "implied_total_from_odds": float(implied_total) if pd.notna(implied_total) else None,
            "spread_from_odds":       float(spread) if (is_home and pd.notna(spread)) else
                                      (-float(spread) if pd.notna(spread) else None),
            "home_win_prob_from_odds": float(home_win_prob) if (is_home and pd.notna(home_win_prob)) else
                                       (1 - float(home_win_prob) if pd.notna(home_win_prob) else None),
            "altitude_ft":            game.get("altitude_ft"),
            "is_dome":                game.get("is_dome"),
            "temp_f":                 game.get("temp_f"),
            "wind_mph":               game.get("wind_mph"),
            "precip_in":              game.get("precip_in"),
            "surface_grass":          _surface_grass(game.get("surface")),
            # Targets (filled post-game by TargetBackfiller)
            "target_points_scored":   None,
            "target_yards_total":     None,
            "target_pass_rate":       None,
            "data_completeness_score": completeness,
        }
        return feature_row


def _compute_rest_days(team: str, game: pd.Series, snap: FeatureSnapshot) -> int | None:
    games = snap.reader.read_as_of("silver.games", ts_col="_silver_ts")
    team_games = games[(games["home_team"] == team) | (games["away_team"] == team)].copy()
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    current_day = pd.to_datetime(game["gameday"])
    prior = team_games[team_games["gameday"] < current_day].sort_values("gameday")
    if prior.empty:
        return None
    return int((current_day - prior.iloc[-1]["gameday"]).days)


def _surface_grass(surface) -> bool | None:
    if surface is None:
        return None
    return "grass" in str(surface).lower()


def _parse_kickoff(gameday, gametime_local) -> datetime:
    day = pd.to_datetime(gameday)
    if gametime_local and isinstance(gametime_local, str):
        try:
            h, m = gametime_local.split(":")
            dt = day + timedelta(hours=int(h), minutes=int(m)) + timedelta(hours=5)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return (day + timedelta(hours=17)).replace(tzinfo=timezone.utc)
