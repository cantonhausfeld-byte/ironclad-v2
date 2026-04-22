"""Shared fixtures for ironclad-v2 tests."""
from __future__ import annotations

import pytest

from ironclad.store.connection import in_memory_connection
from ironclad.store.schema import create_all_tables


@pytest.fixture
def conn():
    """In-memory DuckDB connection with all tables created."""
    c = in_memory_connection()
    create_all_tables(c)
    return c
