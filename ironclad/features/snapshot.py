"""FeatureSnapshot: enforces strict knowledge cutoff."""
from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

from ironclad.config import NFL_TZ
from ironclad.features.elo import INITIAL_ELO, compute_ratings
from ironclad.store.reader import SnapshotReader


class FeatureSnapshot:
    """All feature lookups go through this class to enforce cutoff_ts."""

    def __init__(
        self,
        cutoff_ts: datetime,
        conn: duckdb.DuckDBPyConnection | None = None,
        table_cache: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """table_cache: share full-table reads across snapshots (e.g. one per
        builder run). Safe because tables are read unfiltered and the cutoff is
        applied in memory; the cache must not outlive a silver/bronze refresh.
        Cached frames are shared — callers must filter or copy, never mutate.
        """
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.replace(tzinfo=timezone.utc)
        self.cutoff_ts = cutoff_ts
        self.reader = SnapshotReader(cutoff_ts, conn)
        self._cache: dict[str, pd.DataFrame] = table_cache if table_cache is not None else {}
        self._past_ids: pd.Series | None = None
        self._elo: dict[int, dict[str, float]] = {}

    def _read_cached(self, table: str) -> pd.DataFrame:
        if table not in self._cache:
            self._cache[table] = self.reader.read_table(table)
        return self._cache[table]

    def table(self, table: str) -> pd.DataFrame:
        """Full (unfiltered, shared) table — filter by cutoff yourself, don't mutate."""
        return self._read_cached(table)

    def rosters(self) -> pd.DataFrame:
        """bronze.rosters with normalized team codes (shared, don't mutate)."""
        key = "bronze.rosters#normalized"
        if key not in self._cache:
            from ironclad.store.normalization import normalize_teams
            df = self.reader.read_table("bronze.rosters")
            df["team"] = normalize_teams(df["team"])
            self._cache[key] = df
        return self._cache[key]

    # ── Team history ──────────────────────────────────────────────────────────

    def _games(self) -> pd.DataFrame:
        """All silver games; knowledge cutoff enforced downstream via gameday."""
        games = self._read_cached("silver.games")
        games = games[games["gameday"].notna()].copy()
        games["gameday"] = pd.to_datetime(games["gameday"])
        return games

    @property
    def _cutoff_day(self):
        """Cutoff as a calendar date in the schedule's time zone (US Eastern)."""
        return self.cutoff_ts.astimezone(NFL_TZ).date()

    def _past_game_ids(self) -> pd.Series:
        """game_ids whose gameday is strictly before the cutoff date (US Eastern)."""
        if self._past_ids is None:
            games = self._games()
            self._past_ids = games[games["gameday"].dt.date < self._cutoff_day]["game_id"]
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

    def team_elo(self, team: str, season: int) -> float:
        """Pre-game Elo for team, replaying only games completed before cutoff."""
        if season not in self._elo:
            games = self._games()
            games = games[games["game_id"].isin(self._past_game_ids())]
            self._elo[season] = compute_ratings(games, as_of_season=season)
        return self._elo[season].get(team, INITIAL_ELO)

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
        """(season, week) pairs whose games were all played before the cutoff date.

        Weekly tables (NGS) have one row per player-week, so a week only counts
        once every game in it is done — otherwise a Sunday game would see its
        own week's row as soon as Thursday's game was played.
        """
        games = self._games()
        last_day = games.groupby(["season", "week"])["gameday"].max().dt.date
        past = last_day[last_day < self._cutoff_day].reset_index()[["season", "week"]]
        return past.astype({"season": int, "week": int})

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
