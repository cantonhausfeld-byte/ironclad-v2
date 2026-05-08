"""Typed writers for bronze / silver / gold layers."""
from __future__ import annotations

from datetime import datetime, timezone

import duckdb
import pandas as pd

from ironclad.store.connection import get_connection


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class _BaseWriter:
    def __init__(self, conn: duckdb.DuckDBPyConnection | None = None) -> None:
        self._conn = conn or get_connection()

    def _upsert(self, df: pd.DataFrame, table: str, pk_cols: list[str]) -> int:
        """Insert rows that don't already exist (idempotent by primary key)."""
        if df.empty:
            return 0
        # Deduplicate within the batch before hitting DB constraints
        df = df.drop_duplicates(subset=pk_cols, keep="last")
        tmp = f"_tmp_{table.replace('.', '_')}"
        self._conn.register(tmp, df)
        pk_cond = " AND ".join(f"t.{c} = s.{c}" for c in pk_cols)
        cols = ", ".join(df.columns)
        sql = f"""
        INSERT INTO {table} ({cols})
        SELECT {cols} FROM {tmp} s
        WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {pk_cond})
        """
        self._conn.execute(sql)
        self._conn.unregister(tmp)
        return len(df)

    def _append(self, df: pd.DataFrame, table: str) -> int:
        """Append all rows unconditionally (bronze append-only log)."""
        if df.empty:
            return 0
        tmp = f"_tmp_{table.replace('.', '_')}"
        self._conn.register(tmp, df)
        cols = ", ".join(df.columns)
        self._conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM {tmp}")
        self._conn.unregister(tmp)
        return len(df)


class BronzeWriter(_BaseWriter):
    def write_schedules(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._upsert(df, "bronze.schedules", ["game_id"])

    def write_play_by_play(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._upsert(df, "bronze.play_by_play", ["game_id", "play_id"])

    def write_rosters(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._upsert(df, "bronze.rosters", ["season", "week", "player_id"])

    def write_injuries(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._append(df, "bronze.injuries")

    def write_depth_charts(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._append(df, "bronze.depth_charts")

    def write_odds(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._append(df, "bronze.odds")

    def write_player_props(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._append(df, "bronze.player_props")

    def write_weather(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._append(df, "bronze.weather")

    def write_stadiums(self, df: pd.DataFrame) -> int:
        df = df.copy()
        return self._upsert(df, "bronze.stadiums", ["stadium_id"])

    def write_snap_counts(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._upsert(df, "bronze.snap_counts", ["season", "week", "player_id"])

    def write_player_stats_weekly(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_ingest_ts"] = _now()
        return self._upsert(df, "bronze.player_stats_weekly", ["season", "week", "player_id"])


class SilverWriter(_BaseWriter):
    def write_games(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_silver_ts"] = _now()
        return self._upsert(df, "silver.games", ["game_id"])

    def write_team_game_stats(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_silver_ts"] = _now()
        return self._upsert(df, "silver.team_game_stats", ["game_id", "team"])

    def write_player_game_stats(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_silver_ts"] = _now()
        return self._upsert(df, "silver.player_game_stats", ["game_id", "player_id"])

    def write_player_weekly_status(self, df: pd.DataFrame) -> int:
        df = df.copy()
        df["_silver_ts"] = _now()
        return self._upsert(df, "silver.player_weekly_status", ["season", "week", "player_id"])


class GoldWriter(_BaseWriter):
    def write_team_features(self, df: pd.DataFrame) -> int:
        df = df.copy()
        return self._upsert(df, "gold.team_game_features", ["game_id", "team"])

    def write_player_features(self, df: pd.DataFrame) -> int:
        df = df.copy()
        return self._upsert(df, "gold.player_game_features", ["game_id", "player_id"])

    def write_prediction(self, df: pd.DataFrame) -> int:
        df = df.copy()
        return self._upsert(df, "gold.model_predictions", ["prediction_id"])
