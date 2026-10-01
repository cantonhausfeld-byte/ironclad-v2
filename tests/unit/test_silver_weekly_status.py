"""silver.player_weekly_status: full outer join of injuries and depth charts."""
from __future__ import annotations

from ironclad.store.silver import SilverTransformer


def _depth(conn, season, week, player_id, team="KC", position="WR", depth=1):
    conn.execute(
        "INSERT INTO bronze.depth_charts (season, week, team, position, depth_team, player_id, "
        "player_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [season, week, team, position, depth, player_id, player_id],
    )


def _injury(conn, season, week, player_id, status="Questionable", team="KC", position="WR"):
    conn.execute(
        "INSERT INTO bronze.injuries (season, week, player_id, player_name, team, position, "
        "report_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [season, week, player_id, player_id, team, position, status],
    )


def _status(conn):
    return conn.execute(
        "SELECT season, week, player_id, depth_team, injury_status "
        "FROM silver.player_weekly_status ORDER BY season, player_id"
    ).fetchall()


def test_healthy_depth_chart_players_survive_season_filter(conn):
    _depth(conn, 2026, 4, "healthy")             # not on the injury report
    _depth(conn, 2026, 4, "hurt", depth=2)
    _injury(conn, 2026, 4, "hurt")
    _injury(conn, 2026, 4, "ir_only", status="Out")  # no depth row
    _depth(conn, 2025, 4, "other_season")

    SilverTransformer(conn)._build_player_weekly_status([2026])

    assert _status(conn) == [
        (2026, 4, "healthy", 1, None),
        (2026, 4, "hurt", 2, "Questionable"),
        (2026, 4, "ir_only", None, "Out"),
    ]


def test_no_season_filter_keeps_everything(conn):
    _depth(conn, 2025, 4, "a")
    _injury(conn, 2026, 4, "b")

    SilverTransformer(conn)._build_player_weekly_status(None)

    assert [r[2] for r in _status(conn)] == ["a", "b"]
