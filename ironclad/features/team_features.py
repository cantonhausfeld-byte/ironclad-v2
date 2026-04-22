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
            row = self._build_team_row(
                team=team,
                opponent=opp,
                is_home=is_home,
                game=game,
                snap=snap,
            )
            rows.append(row)

        df = pd.DataFrame(rows)
        self._writer.write_team_features(df)
        return df

    def build_for_season(self, season: int, cutoff_ts: datetime | None = None) -> int:
        """Build features for all games in a season."""
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
                logger.warning("Failed features for %s: %s", g["game_id"], exc)
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

        # Offense rolling L4
        def off_l4(col, default_key):
            if recent.empty or col not in recent.columns:
                return LEAGUE_PRIORS.get(default_key)
            return recent[col].mean()

        # Defense rolling L4 (opponent column = what this team allowed)
        def def_l4(col, default_key):
            if recent.empty or col not in recent.columns:
                return LEAGUE_PRIORS.get(default_key)
            # Flip: defense is opponent's offense against us
            opp_games = self._opponent_games_against(team, recent, snap)
            if opp_games.empty or col not in opp_games.columns:
                return LEAGUE_PRIORS.get(default_key)
            return opp_games[col].mean()

        # Season-to-date
        def std(col, default_key):
            if season_games.empty or col not in season_games.columns:
                return LEAGUE_PRIORS.get(default_key)
            return season_games[col].mean()

        # Rest days
        rest_days = _compute_rest_days(team, game, snap)

        # Odds context
        implied_total = game.get("total_consensus")
        spread = game.get("spread_consensus")
        home_win_prob = game.get("home_ml_implied")

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
            "off_epa_per_play_l4": off_l4("epa_per_play", "off_epa_per_play"),
            "off_pass_epa_l4": off_l4("epa_pass", "off_epa_per_play"),
            "off_rush_epa_l4": off_l4("epa_rush", "off_epa_per_play"),
            "off_pass_rate_l4": off_l4("pass_rate", "pass_rate"),
            "off_yards_per_play_l4": safe_divide(
                pd.Series([recent["total_yards"].mean() if not recent.empty and "total_yards" in recent.columns else None]),
                pd.Series([recent["plays_total"].mean() if not recent.empty and "plays_total" in recent.columns else None]),
                default=LEAGUE_PRIORS["yards_per_play"],
            ).iloc[0],
            "off_success_rate_l4": off_l4("success_rate", "success_rate"),
            "off_points_per_game_l4": off_l4("points_scored", "points_per_game"),
            # Defense L4 (approximated as allowed columns from recent games)
            "def_epa_per_play_l4": _def_epa(team, recent, snap, self._conn),
            "def_pass_epa_l4": None,  # populated in advanced phase
            "def_rush_epa_l4": None,
            "def_yards_allowed_l4": recent["points_allowed"].mean() * 6.5 if not recent.empty and "points_allowed" in recent.columns else None,
            "def_success_rate_l4": None,
            "def_points_allowed_l4": off_l4("points_allowed", "points_per_game"),
            "def_sack_rate_l4": safe_divide(
                pd.Series([recent["sacks_allowed"].mean() if not recent.empty and "sacks_allowed" in recent.columns else None]),
                pd.Series([recent["pass_attempts"].mean() if not recent.empty and "pass_attempts" in recent.columns else None]),
                default=LEAGUE_PRIORS["sack_rate"],
            ).iloc[0],
            # STD
            "off_epa_per_play_std": std("epa_per_play", "off_epa_per_play"),
            "def_epa_per_play_std": None,
            # Context
            "rest_days": rest_days,
            "is_divisional": bool(game.get("overtime")) if False else None,  # filled from schedule
            "implied_total_from_odds": implied_total,
            "spread_from_odds": spread if is_home else (-spread if spread is not None else None),
            "home_win_prob_from_odds": home_win_prob if is_home else (1 - home_win_prob if home_win_prob else None),
            "altitude_ft": game.get("altitude_ft"),
            "is_dome": game.get("is_dome"),
            "temp_f": game.get("temp_f"),
            "wind_mph": game.get("wind_mph"),
            "precip_in": game.get("precip_in"),
            "surface_grass": _surface_grass(game.get("surface")),
            # Targets (filled post-game)
            "target_points_scored": None,
            "target_yards_total": None,
            "target_pass_rate": None,
        }

        # Compute completeness
        feat_values = pd.Series({
            k: v for k, v in feature_row.items()
            if k not in ("game_id", "season", "week", "team", "opponent",
                         "is_home", "cutoff_ts", "feature_version",
                         "data_completeness_score",
                         "target_points_scored", "target_yards_total", "target_pass_rate")
        })
        feature_row["data_completeness_score"] = completeness_score(feat_values)
        return feature_row

    def _opponent_games_against(self, team: str, recent: pd.DataFrame, snap: FeatureSnapshot) -> pd.DataFrame:
        """Get opponent-side stats for team's recent opponents (approximates defense)."""
        if recent.empty:
            return pd.DataFrame()
        opp_ids = recent["game_id"].tolist()
        opp_teams = recent["opponent"].tolist() if "opponent" in recent.columns else []
        if not opp_teams:
            return pd.DataFrame()
        all_stats = snap.reader.read_as_of("silver.team_game_stats")
        mask = all_stats["game_id"].isin(opp_ids) & all_stats["team"].isin(opp_teams)
        return all_stats[mask]


def _def_epa(team: str, recent: pd.DataFrame, snap: FeatureSnapshot, conn) -> float | None:
    """Approximate defensive EPA by looking at what opponents gained against this team."""
    if recent.empty:
        return LEAGUE_PRIORS["def_epa_per_play"]
    all_stats = snap.reader.read_as_of("silver.team_game_stats")
    game_ids = recent["game_id"].tolist()
    opp_rows = all_stats[all_stats["game_id"].isin(game_ids) & (all_stats["opponent"] == team)]
    if opp_rows.empty:
        return LEAGUE_PRIORS["def_epa_per_play"]
    return opp_rows["epa_per_play"].mean()


def _compute_rest_days(team: str, game: pd.Series, snap: FeatureSnapshot) -> int | None:
    games = snap.reader.read_as_of("silver.games", ts_col="_silver_ts")
    team_games = games[(games["home_team"] == team) | (games["away_team"] == team)].copy()
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    current_day = pd.to_datetime(game["gameday"])
    prior = team_games[team_games["gameday"] < current_day].sort_values("gameday")
    if prior.empty:
        return None
    last_day = prior.iloc[-1]["gameday"]
    return (current_day - last_day).days


def _surface_grass(surface) -> bool | None:
    if surface is None:
        return None
    return "grass" in str(surface).lower()


def _parse_kickoff(gameday, gametime_local) -> datetime:
    """Convert gameday + local time string to UTC datetime (approx)."""
    day = pd.to_datetime(gameday)
    if gametime_local and isinstance(gametime_local, str):
        try:
            h, m = gametime_local.split(":")
            dt = day + timedelta(hours=int(h), minutes=int(m)) + timedelta(hours=5)  # EST→UTC approx
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    # Default: noon EST = 17:00 UTC
    return (day + timedelta(hours=17)).replace(tzinfo=timezone.utc)
