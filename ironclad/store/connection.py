"""DuckDB connection factory."""
import threading

import duckdb

from ironclad.config import DB_PATH

_local = threading.local()


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a thread-local DuckDB connection, creating schemas on first use."""
    attr = "_conn_ro" if read_only else "_conn"
    if not hasattr(_local, attr):
        conn = duckdb.connect(str(DB_PATH), read_only=read_only)
        conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
        conn.execute("CREATE SCHEMA IF NOT EXISTS silver")
        conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
        setattr(_local, attr, conn)
    return getattr(_local, attr)


def in_memory_connection() -> duckdb.DuckDBPyConnection:
    """Return a fresh in-memory DuckDB connection (for tests)."""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    conn.execute("CREATE SCHEMA IF NOT EXISTS silver")
    conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
    return conn
