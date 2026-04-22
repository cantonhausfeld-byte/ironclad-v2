"""Time-travel-safe DuckDB reader."""
from __future__ import annotations

from datetime import datetime

import duckdb
import pandas as pd

from ironclad.store.connection import get_connection


class SnapshotReader:
    """Reads any table filtered to rows ingested on or before cutoff_ts."""

    def __init__(
        self,
        cutoff_ts: datetime,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> None:
        self.cutoff_ts = cutoff_ts
        self._conn = conn or get_connection()

    def read_as_of(self, table: str, ts_col: str = "_ingest_ts") -> pd.DataFrame:
        """Return all rows from *table* where ts_col <= cutoff_ts."""
        sql = f"SELECT * FROM {table} WHERE {ts_col} <= $1"
        return self._conn.execute(sql, [self.cutoff_ts]).df()

    def read_table(self, table: str) -> pd.DataFrame:
        """Return all rows (no time filter) — use only for reference tables."""
        return self._conn.execute(f"SELECT * FROM {table}").df()
