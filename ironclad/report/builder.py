"""Assemble report sections from a SimulationResult."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ironclad.simulation.results import SimulationResult


def build_report_context(
    result: SimulationResult,
    game_meta: dict,
    cutoff_ts: datetime,
    model_version: str = "stub_v1",
    n_draws: int = 5000,
) -> dict[str, Any]:
    """Return a template context dict for Jinja2 rendering."""
    home_win_prob, away_win_prob = result.win_probability()
    scores = result.score_summary()
    team_df = result.team_summary()
    player_df = result.player_summary()

    home_team = result.home_team
    away_team = result.away_team

    home_score_p50 = round(scores["home_score_mean"])
    away_score_p50 = round(scores["away_score_mean"])
    total_p50 = round(scores["total_mean"])

    # Implied spread (negative = home favored)
    spread_implied = round(scores["away_score_mean"] - scores["home_score_mean"], 1)

    # Confidence tier based on data_completeness_score from meta
    completeness = game_meta.get("data_completeness_score", 0.5)
    if completeness >= 0.75:
        confidence = "HIGH"
    elif completeness >= 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Team stat tables
    def team_stats(team: str) -> dict:
        rows = team_df[team_df["team"] == team].set_index("metric")
        def get(m, key="mean"):
            return round(rows.loc[m, key], 1) if m in rows.index else "—"
        def rng(m):
            if m not in rows.index:
                return "—"
            lo = round(rows.loc[m, "p10"])
            hi = round(rows.loc[m, "p90"])
            return f"{lo}–{hi}"
        return {
            "total_yards": get("total_yards"),
            "total_yards_range": rng("total_yards"),
            "pass_yards": get("pass_yards"),
            "pass_yards_range": rng("pass_yards"),
            "rush_yards": get("rush_yards"),
            "rush_yards_range": rng("rush_yards"),
            "points": get("points_scored"),
            "points_range": rng("points_scored"),
            "pass_att": get("pass_attempts"),
            "pass_rate": f"{round(get('pass_attempts') / max(1, get('pass_attempts') + get('rush_yards') / 4.3) * 100)}%" if isinstance(get("pass_attempts"), float) else "—",
        }

    # Player tables by position group
    def player_table(team: str, positions: list[str], metrics: list[str], top_n: int | None = None) -> list[dict]:
        if player_df.empty:
            return []
        tdf = player_df[(player_df["team"] == team) & (player_df["position"].isin(positions))]
        if tdf.empty:
            return []

        def pivot(pid: str, pname: str, pos: str) -> dict | None:
            pdf = tdf[(tdf["player_id"] == pid)]
            if pdf.empty:
                return None
            row: dict = {"player_name": pname, "position": pos}
            for m in metrics:
                mdf = pdf[pdf["metric"] == m]
                if mdf.empty:
                    row[m] = None
                    row[f"{m}_range"] = "—"
                else:
                    row[m] = round(mdf.iloc[0]["mean"], 1)
                    row[f"{m}_range"] = f"{round(mdf.iloc[0]['p25'])}–{round(mdf.iloc[0]['p75'])}"
            return row

        players = tdf[["player_id", "player_name", "position"]].drop_duplicates()
        result_rows = []
        for _, p in players.iterrows():
            r = pivot(p["player_id"], p["player_name"], p["position"])
            if r:
                result_rows.append(r)

        # Sort by mean of first metric descending, then trim to top_n
        primary = metrics[0]
        result_rows.sort(
            key=lambda r: r.get(primary, 0) if isinstance(r.get(primary), (int, float)) else 0,
            reverse=True,
        )
        if top_n is not None:
            result_rows = result_rows[:top_n]
        return result_rows

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": f"{home_win_prob * 100:.1f}%",
        "away_win_prob": f"{away_win_prob * 100:.1f}%",
        "home_score": home_score_p50,
        "away_score": away_score_p50,
        "home_score_range": f"{round(scores['home_score_p10'])}–{round(scores['home_score_p90'])}",
        "away_score_range": f"{round(scores['away_score_p10'])}–{round(scores['away_score_p90'])}",
        "total_projected": total_p50,
        "total_range": f"{round(scores['total_p10'])}–{round(scores['total_p90'])}",
        "spread_implied": spread_implied,
        "season": game_meta.get("season", ""),
        "week": game_meta.get("week", ""),
        "gameday": game_meta.get("gameday", ""),
        "stadium": game_meta.get("stadium", ""),
        "weather_desc": _weather_desc(game_meta),
        "surface": game_meta.get("surface", ""),
        "is_dome": game_meta.get("is_dome", False),
        "altitude_ft": game_meta.get("altitude_ft", 0),
        "vegas_total": game_meta.get("total_consensus", "N/A"),
        "vegas_spread": game_meta.get("spread_consensus", "N/A"),
        "home_stats": team_stats(home_team),
        "away_stats": team_stats(away_team),
        "home_qb": player_table(home_team, ["QB"], ["pass_attempts", "completions", "pass_yards", "carries", "rush_yards", "tds"], top_n=2),
        "away_qb": player_table(away_team, ["QB"], ["pass_attempts", "completions", "pass_yards", "carries", "rush_yards", "tds"], top_n=2),
        "home_rb": player_table(home_team, ["RB", "FB"], ["carries", "rush_yards", "targets", "receptions", "rec_yards", "tds"], top_n=4),
        "away_rb": player_table(away_team, ["RB", "FB"], ["carries", "rush_yards", "targets", "receptions", "rec_yards", "tds"], top_n=4),
        "home_wr_te": player_table(home_team, ["WR", "TE"], ["targets", "receptions", "rec_yards", "tds"], top_n=6),
        "away_wr_te": player_table(away_team, ["WR", "TE"], ["targets", "receptions", "rec_yards", "tds"], top_n=6),
        "confidence": confidence,
        "cutoff_ts": cutoff_ts.strftime("%Y-%m-%d %H:%M UTC"),
        "model_version": model_version,
        "n_draws": n_draws,
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _weather_desc(meta: dict) -> str:
    dome = meta.get("is_dome", False)
    if dome:
        return "Indoor (dome)"
    temp = meta.get("temp_f")
    wind = meta.get("wind_mph")
    precip = meta.get("precip_in", 0)
    parts = []
    if temp is not None:
        parts.append(f"{round(temp)}°F")
    if wind is not None:
        parts.append(f"wind {round(wind)} mph")
    if precip and precip > 0.1:
        parts.append("rain/snow")
    return ", ".join(parts) if parts else "Outdoor"
