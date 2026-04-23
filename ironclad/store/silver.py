"""Transform bronze → silver: join, clean, and aggregate."""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from ironclad.config import AVAILABILITY_MAP, AVAILABILITY_DEFAULT
from ironclad.store.connection import get_connection
from ironclad.store.writer import SilverWriter

logger = logging.getLogger(__name__)


class SilverTransformer:
    def __init__(self, conn=None) -> None:
        self._conn = conn or get_connection()
        self._writer = SilverWriter(self._conn)

    def run(self, seasons: list[int] | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        counts["games"] = self._build_games(seasons)
        counts["team_game_stats"] = self._build_team_game_stats(seasons)
        counts["player_game_stats"] = self._build_player_game_stats(seasons)
        counts["player_weekly_status"] = self._build_player_weekly_status(seasons)
        return counts

    # ── silver.games ──────────────────────────────────────────────────────────

    def _build_games(self, seasons: list[int] | None) -> int:
        where = self._season_filter("s.season", seasons)
        df = self._conn.execute(f"""
            SELECT
                s.game_id,
                s.season,
                COALESCE(s.season_type, 'REG') AS season_type,
                s.week,
                s.gameday,
                s.gametime            AS gametime_local,
                s.away_team,
                s.home_team,
                s.stadium_id,
                st.is_dome,
                COALESCE(w.temp_f,   s.temp)         AS temp_f,
                COALESCE(w.wind_mph, s.wind)          AS wind_mph,
                w.precip_in,
                s.spread_line         AS spread_consensus,
                s.total_line          AS total_consensus,
                s.home_moneyline,
                s.away_moneyline,
                s.away_score,
                s.home_score,
                (s.away_score + s.home_score) AS total_score,
                (s.home_score - s.away_score) AS home_margin,
                CASE WHEN s.home_score > s.away_score THEN true
                     WHEN s.home_score < s.away_score THEN false
                     ELSE NULL END             AS home_win,
                s.overtime,
                st.surface,
                COALESCE(st.altitude_ft, 0)   AS altitude_ft
            FROM bronze.schedules s
            LEFT JOIN bronze.stadiums st ON s.home_team = st.team
            LEFT JOIN (
                SELECT game_id, AVG(temp_f) AS temp_f, AVG(wind_mph) AS wind_mph,
                       AVG(precip_in) AS precip_in
                FROM bronze.weather
                GROUP BY game_id
            ) w ON s.game_id = w.game_id
            {where}
        """).df()

        if df.empty:
            return 0

        df["home_ml_implied"], df["away_ml_implied"] = zip(
            *df.apply(
                lambda r: _remove_vig(r["home_moneyline"], r["away_moneyline"]),
                axis=1,
            )
        )
        df = df.drop(columns=["home_moneyline", "away_moneyline"])
        return self._writer.write_games(df)

    # ── silver.team_game_stats ────────────────────────────────────────────────

    def _build_team_game_stats(self, seasons: list[int] | None) -> int:
        where = self._season_filter("season", seasons)
        and_where = where.replace("WHERE", "AND")
        df = self._conn.execute(f"""
            SELECT
                game_id, season, week,
                posteam                                        AS team,
                defteam                                        AS opponent,
                COUNT(*)                                       AS plays_total,
                SUM(pass_attempt)                              AS pass_attempts,
                SUM(complete_pass)                             AS completions,
                SUM(CASE WHEN pass_attempt=1 THEN yards_gained ELSE 0 END)   AS pass_yards,
                SUM(rush_attempt)                              AS rush_attempts,
                SUM(CASE WHEN rush_attempt=1 THEN yards_gained ELSE 0 END)   AS rush_yards,
                SUM(yards_gained)                              AS total_yards,
                SUM(touchdown)                                 AS touchdowns,
                SUM(interception + fumble_lost)                AS turnovers,
                SUM(sack)                                      AS sacks_allowed,
                SUM(CASE WHEN sack=1 THEN -yards_gained ELSE 0 END)          AS sack_yards_lost,
                SUM(field_goal_attempt)                        AS field_goals_att,
                SUM(CASE WHEN field_goal_result='made' THEN 1 ELSE 0 END)    AS field_goals_made,
                AVG(epa)                                       AS epa_per_play,
                AVG(CASE WHEN pass_attempt=1 THEN epa ELSE NULL END)         AS epa_pass,
                AVG(CASE WHEN rush_attempt=1 THEN epa ELSE NULL END)         AS epa_rush,
                AVG(CASE WHEN epa > 0 THEN 1.0 ELSE 0.0 END)  AS success_rate,
                CASE WHEN COUNT(*) > 0
                     THEN SUM(pass_attempt)::FLOAT / COUNT(*)
                     ELSE NULL END                             AS pass_rate,
                -- Red zone (inside opponent 20)
                SUM(CASE WHEN yardline_100 <= 20 AND pass_attempt=1 THEN 1 ELSE 0 END) AS rz_pass_attempts,
                SUM(CASE WHEN yardline_100 <= 20 AND rush_attempt=1 THEN 1 ELSE 0 END) AS rz_rush_attempts,
                SUM(CASE WHEN yardline_100 <= 20 AND touchdown=1    THEN 1 ELSE 0 END) AS rz_touchdowns,
                -- Air yards (passing)
                SUM(air_yards)                                 AS total_air_yards,
                -- Pressure proxy: sack rate
                CASE WHEN SUM(pass_attempt) > 0
                     THEN SUM(sack)::FLOAT / SUM(pass_attempt)
                     ELSE NULL END                             AS sack_rate
            FROM bronze.play_by_play
            WHERE posteam IS NOT NULL
              AND play_type IN ('pass','run','qb_kneel','qb_spike')
            {and_where}
            GROUP BY game_id, season, week, posteam, defteam
        """).df()

        if df.empty:
            return 0

        # Join scores from bronze schedules
        games = self._conn.execute(
            "SELECT game_id, home_team, away_team, home_score, away_score FROM bronze.schedules"
        ).df()
        df = df.merge(games, on="game_id", how="left")
        df["is_home"] = df["team"] == df["home_team"]
        df["points_scored"] = np.where(df["is_home"], df["home_score"], df["away_score"])
        df["points_allowed"] = np.where(df["is_home"], df["away_score"], df["home_score"])
        df = df.drop(columns=["home_team", "away_team", "home_score", "away_score"])
        df["punts"] = None
        df["first_downs"] = None
        return self._writer.write_team_game_stats(df)

    # ── silver.player_game_stats ──────────────────────────────────────────────

    def _build_player_game_stats(self, seasons: list[int] | None) -> int:
        where = self._season_filter("season", seasons)
        and_where = where.replace("WHERE", "AND")

        passing = self._conn.execute(f"""
            SELECT game_id, season, week, posteam AS team,
                   passer_player_id AS player_id, passer_player_name AS player_name,
                   COUNT(*)                      AS pass_attempts,
                   SUM(complete_pass)            AS completions,
                   SUM(CASE WHEN pass_attempt=1 THEN yards_gained ELSE 0 END) AS pass_yards,
                   SUM(CASE WHEN pass_attempt=1 THEN touchdown ELSE 0 END)    AS pass_tds,
                   SUM(interception)             AS interceptions
            FROM bronze.play_by_play
            WHERE pass_attempt = 1 AND passer_player_id IS NOT NULL
            {and_where}
            GROUP BY game_id, season, week, posteam, passer_player_id, passer_player_name
        """).df()

        rushing = self._conn.execute(f"""
            SELECT game_id, season, week, posteam AS team,
                   rusher_player_id AS player_id, rusher_player_name AS player_name,
                   SUM(rush_attempt) AS carries,
                   SUM(CASE WHEN rush_attempt=1 THEN yards_gained ELSE 0 END) AS rush_yards,
                   SUM(CASE WHEN rush_attempt=1 THEN touchdown ELSE 0 END)    AS rush_tds,
                   SUM(CASE WHEN yardline_100 <= 10 AND rush_attempt=1 THEN 1 ELSE 0 END) AS rz_carries
            FROM bronze.play_by_play
            WHERE rush_attempt = 1 AND rusher_player_id IS NOT NULL
            {and_where}
            GROUP BY game_id, season, week, posteam, rusher_player_id, rusher_player_name
        """).df()

        receiving = self._conn.execute(f"""
            SELECT game_id, season, week, posteam AS team,
                   receiver_player_id AS player_id, receiver_player_name AS player_name,
                   COUNT(*)                                              AS targets,
                   SUM(complete_pass)                                    AS receptions,
                   SUM(CASE WHEN complete_pass=1 THEN yards_gained ELSE 0 END) AS rec_yards,
                   SUM(CASE WHEN complete_pass=1 THEN touchdown ELSE 0 END)    AS rec_tds,
                   SUM(air_yards)                                        AS air_yards,
                   SUM(yards_after_catch)                                AS yards_after_catch,
                   SUM(CASE WHEN yardline_100 <= 20 THEN 1 ELSE 0 END)  AS rz_targets
            FROM bronze.play_by_play
            WHERE pass_attempt = 1 AND receiver_player_id IS NOT NULL
            {and_where}
            GROUP BY game_id, season, week, posteam, receiver_player_id, receiver_player_name
        """).df()

        if passing.empty and rushing.empty and receiving.empty:
            return 0

        all_ids = pd.concat([
            passing[["game_id", "season", "week", "team", "player_id", "player_name"]],
            rushing[["game_id", "season", "week", "team", "player_id", "player_name"]],
            receiving[["game_id", "season", "week", "team", "player_id", "player_name"]],
        ]).drop_duplicates(subset=["game_id", "player_id"])

        df = all_ids.copy()
        for src, cols in [
            (passing,   ["pass_attempts", "completions", "pass_yards", "pass_tds", "interceptions"]),
            (rushing,   ["carries", "rush_yards", "rush_tds", "rz_carries"]),
            (receiving, ["targets", "receptions", "rec_yards", "rec_tds",
                         "air_yards", "yards_after_catch", "rz_targets"]),
        ]:
            if not src.empty:
                df = df.merge(src[["game_id", "player_id"] + cols], on=["game_id", "player_id"], how="left")

        int_cols = ["pass_attempts", "completions", "pass_tds", "interceptions",
                    "carries", "rush_tds", "targets", "receptions", "rec_tds",
                    "rz_carries", "rz_targets"]
        float_cols = ["pass_yards", "rush_yards", "rec_yards", "air_yards", "yards_after_catch"]
        for c in int_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(int)
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0.0)

        df["total_tds"] = (
            df.get("pass_tds", 0) + df.get("rush_tds", 0) + df.get("rec_tds", 0)
        )

        games = self._conn.execute(
            "SELECT game_id, home_team, away_team FROM silver.games"
        ).df()
        df = df.merge(games, on="game_id", how="left")
        df["is_home"] = df["team"] == df["home_team"]
        df["opponent"] = np.where(df["is_home"], df["away_team"], df["home_team"])
        df = df.drop(columns=["home_team", "away_team"], errors="ignore")

        # Enrich position from bronze.rosters (latest week for each season)
        try:
            rosters = self._conn.execute("""
                SELECT player_id, position,
                       ROW_NUMBER() OVER (
                           PARTITION BY player_id, season
                           ORDER BY week DESC, _ingest_ts DESC
                       ) AS rn,
                       season
                FROM bronze.rosters
                WHERE position IS NOT NULL AND position != ''
            """).df()
            rosters = rosters[rosters["rn"] == 1][["player_id", "season", "position"]]
            if not rosters.empty:
                df = df.merge(rosters, on=["player_id", "season"], how="left", suffixes=("_old", ""))
                if "position_old" in df.columns:
                    df["position"] = df["position"].fillna(df["position_old"])
                    df = df.drop(columns=["position_old"])
        except Exception as exc:
            logger.debug("Position enrichment from rosters failed: %s", exc)

        if "position" not in df.columns:
            df["position"] = "UNK"
        else:
            df["position"] = df["position"].fillna("UNK")
        return self._writer.write_player_game_stats(df)

    # ── silver.player_weekly_status ───────────────────────────────────────────

    def _build_player_weekly_status(self, seasons: list[int] | None) -> int:
        where = self._season_filter("i.season", seasons)
        and_where = where.replace("WHERE", "AND")
        df = self._conn.execute(f"""
            SELECT
                COALESCE(i.season, d.season)           AS season,
                COALESCE(i.week,   d.week)             AS week,
                COALESCE(i.player_id, d.player_id)     AS player_id,
                COALESCE(i.player_name, d.player_name) AS player_name,
                COALESCE(i.team, d.team)               AS team,
                COALESCE(i.position, d.position)       AS position,
                d.depth_team,
                i.report_status                        AS injury_status
            FROM (
                SELECT season, week, player_id, player_name, team, position,
                       report_status,
                       ROW_NUMBER() OVER (
                           PARTITION BY season, week, player_id
                           ORDER BY _ingest_ts DESC
                       ) AS rn
                FROM bronze.injuries
            ) i
            FULL OUTER JOIN (
                SELECT season, week, player_id, player_name, team, position, depth_team,
                       ROW_NUMBER() OVER (
                           PARTITION BY season, week, player_id
                           ORDER BY _ingest_ts DESC
                       ) AS rn
                FROM bronze.depth_charts
            ) d ON i.player_id = d.player_id
              AND i.season = d.season
              AND i.week = d.week
            WHERE COALESCE(i.rn, 1) = 1
              AND COALESCE(d.rn, 1) = 1
            {and_where}
        """).df()

        if df.empty:
            return 0

        df["availability"] = df["injury_status"].map(AVAILABILITY_MAP).fillna(AVAILABILITY_DEFAULT)

        # Enrich with snap rate from bronze.snap_counts (nflverse)
        try:
            snap_df = self._conn.execute("""
                SELECT player_id, season, week, offense_pct AS snap_rate
                FROM bronze.snap_counts
                WHERE offense_snaps > 0
            """).df()
            if not snap_df.empty:
                df = df.merge(snap_df, on=["player_id", "season", "week"], how="left")
            else:
                df["snap_rate"] = None
        except Exception as exc:
            logger.debug("snap_counts join skipped (table may not exist yet): %s", exc)
            df["snap_rate"] = None

        return self._writer.write_player_weekly_status(df)

    @staticmethod
    def _season_filter(col: str, seasons: list[int] | None) -> str:
        if not seasons:
            return ""
        s = ", ".join(str(s) for s in seasons)
        return f"WHERE {col} IN ({s})"


def _remove_vig(ml_home, ml_away) -> tuple[float, float]:
    def to_prob(ml):
        if ml is None or pd.isna(ml):
            return None
        ml = float(ml)
        if ml > 0:
            return 100.0 / (ml + 100.0)
        else:
            return abs(ml) / (abs(ml) + 100.0)

    p_h, p_a = to_prob(ml_home), to_prob(ml_away)
    if p_h is None or p_a is None:
        return 0.573, 0.427
    total = p_h + p_a
    return p_h / total, p_a / total
