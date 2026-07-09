"""Shared text formatters for CLI + Discord (single source of truth)."""
from __future__ import annotations

import pandas as pd


def format_top_edges_block(edges_df: pd.DataFrame | None, limit: int = 10) -> str:
    """Render top-N edges as a fixed-width text block (no fence)."""
    if edges_df is None or edges_df.empty:
        return "No edges found."

    df = edges_df.sort_values("ev", ascending=False).head(limit)
    total = len(edges_df)

    header = f"Top {min(limit, total)} edges ({total} total)"
    cols = f"{'Player':<22} {'Stat':<14} {'Side':<5} {'Line':<7} {'EV':<8} {'Kelly':<6}"
    sep = "-" * len(cols)

    lines = [header, "", cols, sep]
    for _, r in df.iterrows():
        player = str(r.get("player_name") or "")[:22]
        stat = str(r.get("stat_type") or "")[:14]
        side = str(r.get("side") or "")[:5]
        line_val = r.get("market_line")
        line_str = f"{line_val:g}" if line_val is not None and not pd.isna(line_val) else "—"
        ev_val = r.get("ev")
        ev_str = f"{ev_val:+.3f}" if ev_val is not None and not pd.isna(ev_val) else "—"
        kelly_val = r.get("kelly")
        kelly_str = f"{kelly_val:.3f}" if kelly_val is not None and not pd.isna(kelly_val) else "—"
        lines.append(f"{player:<22} {stat:<14} {side:<5} {line_str:<7} {ev_str:<8} {kelly_str:<6}")

    return "\n".join(lines)


_COMPARE_ROWS = [
    ("Games", "games", "int"),
    ("Points/G", "points_scored", "float"),
    ("Yards/G", "total_yards", "float"),
    ("Pass YPG", "pass_yards", "float"),
    ("Rush YPG", "rush_yards", "float"),
    ("TOs/G", "turnovers", "float"),
]


def _team_season_summary(conn, team: str, season: int) -> dict:
    """Aggregate silver.team_game_stats for a team-season. Returns {} if no rows."""
    df = conn.execute(
        """
        SELECT points_scored, total_yards, pass_yards, rush_yards, turnovers
        FROM silver.team_game_stats
        WHERE team = ? AND season = ?
        """,
        [team, season],
    ).df()
    if df.empty:
        return {}
    return {
        "games": len(df),
        "points_scored": df["points_scored"].mean(),
        "total_yards": df["total_yards"].mean(),
        "pass_yards": df["pass_yards"].mean(),
        "rush_yards": df["rush_yards"].mean(),
        "turnovers": df["turnovers"].mean(),
    }


def format_team_comparison_block(conn, team_a: str, team_b: str, season: int) -> str | None:
    """Render side-by-side team stats. Returns None if neither team has data."""
    a = _team_season_summary(conn, team_a, season)
    b = _team_season_summary(conn, team_b, season)
    if not a and not b:
        return None

    header = f"{team_a} vs {team_b} — {season}"
    cols = f"{'':<14} {team_a:>8}  {team_b:>8}"
    sep = "-" * len(cols)

    lines = [header, "", cols, sep]
    for label, key, kind in _COMPARE_ROWS:
        va = a.get(key) if a else None
        vb = b.get(key) if b else None
        if kind == "int":
            sa = f"{int(va):d}" if va is not None else "—"
            sb = f"{int(vb):d}" if vb is not None else "—"
        else:
            sa = f"{va:.1f}" if va is not None else "—"
            sb = f"{vb:.1f}" if vb is not None else "—"
        lines.append(f"{label:<14} {sa:>8}  {sb:>8}")

    return "\n".join(lines)
