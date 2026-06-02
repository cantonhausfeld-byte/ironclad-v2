"""FeatureSnapshot: enforces strict knowledge cutoff."""
from __future__ import annotations

from datetime import datetime, timezone

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
        self._cache: dict[str, pd.DataFrame] = {}
        self._past_ids: pd.Series | None = None

    def _read_cached(self, table: str) -> pd.DataFrame:
        if table not in self._cache:
            self._cache[table] = self.reader.read_table(table)
        return self._cache[table]

    # ── Team history ──────────────────────────────────────────────────────────

    def _games(self) -> pd.DataFrame:
        """All silver games; knowledge cutoff enforced downstream via gameday."""
        games = self._read_cached("silver.games")
        games = games[games["gameday"].notna()].copy()
        games["gameday"] = pd.to_datetime(games["gameday"])
        return games

    def _past_game_ids(self) -> pd.Series:
        """game_ids whose gameday is strictly before the cutoff date."""
        if self._past_ids is None:
            games = self._games()
            self._past_ids = games[games["gameday"].dt.date < self.cutoff_ts.date()]["game_id"]
        return self._past_ids

    def team_recent_games(self, team: str, n: int = 4) -> pd.DataFrame:
        """Last n completed games for team, prior to cutoff."""
        df = self._read_cached("silver.team_game_stats")
        df = df[df["team"] == team].copy()
        df = df[df["game_id"].isin(self._past_game_ids())]
        return df.sort_values(["season", "week"]).tail(n)

    def team_season_games(self, team: str, season: int) -> pd.DataFrame:
        """All completed games for team in season, prior to cutoff."""
        df = self._read_cached("silver.team_game_stats")
        df = df[(df["team"] == team) & (df["season"] == season)].copy()
        return df[df["game_id"].isin(self._past_game_ids())]

    # ── Player history ────────────────────────────────────────────────────────

    def player_recent_games(self, player_id: str, n: int = 4) -> pd.DataFrame:
        """Last n games for a player prior to cutoff."""
        df = self._read_cached("silver.player_game_stats")
        df = df[df["player_id"] == player_id].copy()
        df = df[df["game_id"].isin(self._past_game_ids())]
        return df.sort_values(["season", "week"]).tail(n)

    def player_status(self, player_id: str, season: int, week: int) -> pd.Series | None:
        """Most recent injury/depth status for player as of cutoff."""
        df = self._read_cached("silver.player_weekly_status")
        df = df[(df["player_id"] == player_id) & (df["season"] == season) & (df["week"] <= week)]
        if df.empty:
            return None
        return df.sort_values("week").iloc[-1]

    def player_recent_status(self, player_id: str, season: int, week: int, n: int = 4) -> pd.DataFrame:
        """Last n weekly status rows for a player prior to the target week (includes snap_rate)."""
        df = self._read_cached("silver.player_weekly_status")
        df = df[(df["player_id"] == player_id) & (df["season"] == season) & (df["week"] < week)]
        return df.sort_values("week").tail(n)

    # ── NGS tracking data ────────────────────────────────────────────────────

    def _past_week_pairs(self) -> pd.DataFrame:
        """DataFrame of (season, week) int pairs for completed games before cutoff."""
        games = self._games()
        past = games[games["gameday"].dt.date < self.cutoff_ts.date()][["season", "week"]]
        return past.drop_duplicates().astype({"season": int, "week": int})

    def player_ngs_receiving(self, player_id: str, n: int = 4) -> pd.DataFrame:
        """Last n NGS receiving rows for player from completed games before cutoff."""
        df = self._read_cached("bronze.ngs_receiving")
        df = df[df["player_id"] == player_id].copy()
        if df.empty:
            return df
        past = self._past_week_pairs()
        if not past.empty:
            df["season"] = df["season"].astype(int)
            df["week"] = df["week"].astype(int)
            df = df.merge(past, on=["season", "week"], how="inner")
        return df.sort_values(["season", "week"]).tail(n)

    def player_ngs_passing(self, player_id: str, n: int = 4) -> pd.DataFrame:
        """Last n NGS passing rows for player from completed games before cutoff."""
        df = self._read_cached("bronze.ngs_passing")
        df = df[df["player_id"] == player_id].copy()
        if df.empty:
            return df
        past = self._past_week_pairs()
        if not past.empty:
            df["season"] = df["season"].astype(int)
            df["week"] = df["week"].astype(int)
            df = df.merge(past, on=["season", "week"], how="inner")
        return df.sort_values(["season", "week"]).tail(n)

    def player_ngs_rushing(self, player_id: str, n: int = 4) -> pd.DataFrame:
        """Last n NGS rushing rows for player from completed games before cutoff."""
        df = self._read_cached("bronze.ngs_rushing")
        df = df[df["player_id"] == player_id].copy()
        if df.empty:
            return df
        past = self._past_week_pairs()
        if not past.empty:
            df["season"] = df["season"].astype(int)
            df["week"] = df["week"].astype(int)
            df = df.merge(past, on=["season", "week"], how="inner")
        return df.sort_values(["season", "week"]).tail(n)

    # ── Game context ──────────────────────────────────────────────────────────

    def game_row(self, game_id: str) -> pd.Series | None:
        df = self._read_cached("silver.games")
        df = df[df["game_id"] == game_id]
        return df.iloc[0] if not df.empty else None

    def team_players_for_game(self, team: str, season: int, week: int) -> pd.DataFrame:
        """All players with weekly status for team in given week."""
        df = self._read_cached("silver.player_weekly_status")
        return df[(df["team"] == team) & (df["season"] == season) & (df["week"] == week)]
