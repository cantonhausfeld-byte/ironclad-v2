"""FeatureSnapshot: enforces strict knowledge cutoff."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from ironclad.store.reader import SnapshotReader


class FeatureSnapshot:
    """All feature lookups go through this class to enforce cutoff_ts."""

    def __init__(
        self,
        cutoff_ts: datetime,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.replace(tzinfo=timezone.utc)
        self.cutoff_ts = cutoff_ts
        self.reader = SnapshotReader(cutoff_ts, conn)

    # ── Team history ──────────────────────────────────────────────────────────

    def team_recent_games(self, team: str, n: int = 4) -> pd.DataFrame:
        """Last n completed games for team, prior to cutoff."""
        df = self.reader.read_as_of("silver.team_game_stats")
        df = df[df["team"] == team].copy()
        # Filter to games before cutoff (use silver.games for dates)
        games = self.reader.read_as_of("silver.games", ts_col="_silver_ts")
        games = games[games["gameday"].notna()]
        games["gameday"] = pd.to_datetime(games["gameday"])
        cutoff_date = self.cutoff_ts.date()
        past_ids = games[games["gameday"].dt.date < cutoff_date]["game_id"]
        df = df[df["game_id"].isin(past_ids)]
        return df.sort_values("week").tail(n)

    def team_season_games(self, team: str, season: int) -> pd.DataFrame:
        """All completed games for team in season, prior to cutoff."""
        df = self.reader.read_as_of("silver.team_game_stats")
        df = df[(df["team"] == team) & (df["season"] == season)].copy()
        games = self.reader.read_as_of("silver.games", ts_col="_silver_ts")
        games["gameday"] = pd.to_datetime(games["gameday"])
        cutoff_date = self.cutoff_ts.date()
        past_ids = games[games["gameday"].dt.date < cutoff_date]["game_id"]
        return df[df["game_id"].isin(past_ids)]

    # ── Player history ────────────────────────────────────────────────────────

    def player_recent_games(self, player_id: str, n: int = 4) -> pd.DataFrame:
        """Last n games for a player prior to cutoff."""
        df = self.reader.read_as_of("silver.player_game_stats")
        df = df[df["player_id"] == player_id].copy()
        games = self.reader.read_as_of("silver.games", ts_col="_silver_ts")
        games["gameday"] = pd.to_datetime(games["gameday"])
        cutoff_date = self.cutoff_ts.date()
        past_ids = games[games["gameday"].dt.date < cutoff_date]["game_id"]
        df = df[df["game_id"].isin(past_ids)]
        return df.sort_values("week").tail(n)

    def player_status(self, player_id: str, season: int, week: int) -> pd.Series | None:
        """Most recent injury/depth status for player as of cutoff."""
        df = self.reader.read_as_of("silver.player_weekly_status")
        df = df[(df["player_id"] == player_id) & (df["season"] == season) & (df["week"] <= week)]
        if df.empty:
            return None
        return df.sort_values("week").iloc[-1]

    def player_recent_status(self, player_id: str, season: int, week: int, n: int = 4) -> pd.DataFrame:
        """Last n weekly status rows for a player prior to the target week (includes snap_rate)."""
        df = self.reader.read_as_of("silver.player_weekly_status")
        df = df[(df["player_id"] == player_id) & (df["season"] == season) & (df["week"] < week)]
        return df.sort_values("week").tail(n)

    # ── Game context ──────────────────────────────────────────────────────────

    def game_row(self, game_id: str) -> pd.Series | None:
        df = self.reader.read_as_of("silver.games", ts_col="_silver_ts")
        df = df[df["game_id"] == game_id]
        return df.iloc[0] if not df.empty else None

    def team_players_for_game(self, team: str, season: int, week: int) -> pd.DataFrame:
        """All players with weekly status for team in given week."""
        df = self.reader.read_as_of("silver.player_weekly_status")
        return df[(df["team"] == team) & (df["season"] == season) & (df["week"] == week)]
