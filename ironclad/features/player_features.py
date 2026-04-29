"""Build gold.player_game_features from silver tables."""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import numpy as np

from ironclad.config import FEATURE_VERSION, ROLLING_WINDOW, AVAILABILITY_DEFAULT
from ironclad.features.snapshot import FeatureSnapshot
from ironclad.features.utils import safe_divide, completeness_score
from ironclad.store.writer import GoldWriter

logger = logging.getLogger(__name__)

SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}


class PlayerFeatureBuilder:
    def __init__(self, conn=None) -> None:
        from ironclad.store.connection import get_connection
        self._conn = conn or get_connection()
        self._writer = GoldWriter(self._conn)

    def build_for_game(self, game_id: str, cutoff_ts: datetime) -> pd.DataFrame:
        snap = FeatureSnapshot(cutoff_ts, self._conn)
        game = snap.game_row(game_id)
        if game is None:
            return pd.DataFrame()

        season = int(game["season"])
        week = int(game["week"])

        team_feats = self._conn.execute(
            "SELECT * FROM gold.team_game_features WHERE game_id = ?", [game_id]
        ).df()

        rows = []
        for team, opponent, is_home in [
            (game["home_team"], game["away_team"], True),
            (game["away_team"], game["home_team"], False),
        ]:
            players = snap.team_players_for_game(team, season, week)
            if players.empty:
                players = _fallback_roster(team, season, week, snap)

            opp_feats = team_feats[team_feats["team"] == opponent]
            own_feats = team_feats[team_feats["team"] == team]

            # Team-level PBP totals for share denominators
            team_recent = snap.team_recent_games(team, n=ROLLING_WINDOW)

            for _, player in players.iterrows():
                pos = player.get("position", "UNK")
                if pos not in SKILL_POSITIONS:
                    continue
                try:
                    row = self._build_player_row(
                        player=player, team=team, opponent=opponent,
                        is_home=is_home, game=game, snap=snap,
                        team_recent=team_recent, opp_feats=opp_feats, own_feats=own_feats,
                    )
                    rows.append(row)
                except Exception as exc:
                    logger.debug("Player %s failed: %s", player.get("player_id"), exc)

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        self._writer.write_player_features(df)
        return df

    def _build_player_row(
        self,
        player: pd.Series,
        team: str,
        opponent: str,
        is_home: bool,
        game: pd.Series,
        snap: FeatureSnapshot,
        team_recent: pd.DataFrame,
        opp_feats: pd.DataFrame,
        own_feats: pd.DataFrame,
    ) -> dict:
        pid = str(player["player_id"])
        pos = player.get("position", "UNK")
        recent = snap.player_recent_games(pid, n=ROLLING_WINDOW)
        recent_status = snap.player_recent_status(pid, int(game["season"]), int(game["week"]), n=ROLLING_WINDOW)

        # Team volume denominators from rolling L4
        team_pass_att = float(team_recent["pass_attempts"].mean()) if not team_recent.empty and "pass_attempts" in team_recent.columns else 30.0
        team_rush_att = float(team_recent["rush_attempts"].mean()) if not team_recent.empty and "rush_attempts" in team_recent.columns else 25.0
        team_air_yards = float(team_recent["total_air_yards"].mean()) if not team_recent.empty and "total_air_yards" in team_recent.columns else 200.0
        team_rz_pass = float(team_recent["rz_pass_attempts"].mean()) if not team_recent.empty and "rz_pass_attempts" in team_recent.columns else 5.0
        team_rz_rush = float(team_recent["rz_rush_attempts"].mean()) if not team_recent.empty and "rz_rush_attempts" in team_recent.columns else 5.0

        team_pass_att = max(team_pass_att or 30.0, 1.0)
        team_rush_att = max(team_rush_att or 25.0, 1.0)
        team_air_yards = max(team_air_yards or 200.0, 1.0)
        team_rz_pass = max(team_rz_pass or 5.0, 1.0)
        team_rz_rush = max(team_rz_rush or 5.0, 1.0)

        def avg(col, default=0.0):
            if recent.empty or col not in recent.columns:
                return default
            vals = recent[col].dropna()
            return float(vals.mean()) if len(vals) else default

        targets_avg = avg("targets", 0.0)
        carries_avg = avg("carries", 0.0)
        rec_avg = avg("receptions", 0.0)
        air_yards_avg = avg("air_yards", 0.0)
        rz_targets_avg = avg("rz_targets", 0.0)
        rz_carries_avg = avg("rz_carries", 0.0)

        eps = 1e-6
        catch_rate = rec_avg / (targets_avg + eps) if targets_avg > 0 else None
        yds_per_tgt = avg("rec_yards", 0.0) / (targets_avg + eps) if targets_avg > 0 else None
        yds_per_carry = avg("rush_yards", 0.0) / (carries_avg + eps) if carries_avg > 0 else None
        yac_per_rec = avg("yards_after_catch", 0.0) / (rec_avg + eps) if rec_avg > 0 else None
        total_tds_avg = avg("total_tds", 0.0)
        td_per_tgt = total_tds_avg / (targets_avg + eps) if targets_avg > 0 else None
        td_per_carry = total_tds_avg / (carries_avg + eps) if carries_avg > 0 else None

        # Clamp rates to valid range
        if catch_rate is not None:
            catch_rate = max(0.0, min(1.0, catch_rate))
        if td_per_tgt is not None:
            td_per_tgt = max(0.0, min(0.5, td_per_tgt))
        if td_per_carry is not None:
            td_per_carry = max(0.0, min(0.5, td_per_carry))

        def opp_feat(col, default=None):
            if opp_feats.empty or col not in opp_feats.columns:
                return default
            return opp_feats.iloc[0].get(col, default)

        def own_feat(col, default=None):
            if own_feats.empty or col not in own_feats.columns:
                return default
            return own_feats.iloc[0].get(col, default)

        availability = float(player.get("availability", AVAILABILITY_DEFAULT))
        depth_team = player.get("depth_team")
        n_games = len(recent)

        feature_row = {
            "game_id": game["game_id"],
            "season": int(game["season"]),
            "week": int(game["week"]),
            "player_id": pid,
            "player_name": player.get("player_name", ""),
            "team": team,
            "opponent": opponent,
            "position": pos,
            "is_home": is_home,
            "cutoff_ts": snap.cutoff_ts,
            "feature_version": FEATURE_VERSION,
            "availability": availability,
            "depth_team": int(depth_team) if pd.notna(depth_team) else None,
            "snap_rate_l4": _avg_col(recent_status, "snap_rate"),
            "target_share_l4":          targets_avg / team_pass_att,
            "carry_share_l4":           carries_avg / team_rush_att,
            "air_yards_share_l4":       air_yards_avg / team_air_yards,
            "route_rate_l4":            None,
            "redzone_target_share_l4":  rz_targets_avg / team_rz_pass,
            "redzone_carry_share_l4":   rz_carries_avg / team_rz_rush,
            "catch_rate_l4":            catch_rate,
            "yards_per_target_l4":      yds_per_tgt,
            "yards_per_carry_l4":       yds_per_carry,
            "yac_per_rec_l4":           yac_per_rec,
            "td_rate_per_target_l4":    td_per_tgt,
            "td_rate_per_carry_l4":     td_per_carry,
            "opp_def_pass_epa_l4":      opp_feat("def_epa_per_play_l4"),
            "opp_def_rush_epa_l4":      opp_feat("def_rush_epa_l4"),
            "opp_def_sack_rate_l4":     opp_feat("def_sack_rate_l4"),
            "team_off_pass_rate_l4":    own_feat("off_pass_rate_l4"),
            "team_off_epa_l4":          own_feat("off_epa_per_play_l4"),
            "team_implied_total":       own_feat("implied_total_from_odds"),
            # Targets filled post-game
            "target_targets": None, "target_carries": None,
            "target_receptions": None, "target_rec_yards": None,
            "target_rush_yards": None, "target_total_tds": None,
            "data_completeness_score": min(1.0, n_games / ROLLING_WINDOW),
        }
        return feature_row


def _avg_col(df: pd.DataFrame, col: str, default=None):
    """Mean of a column across a DataFrame; returns default if absent or all-null."""
    if df.empty or col not in df.columns:
        return default
    vals = df[col].dropna()
    return float(vals.mean()) if len(vals) else default


def _fallback_roster(team: str, season: int, week: int, snap: FeatureSnapshot) -> pd.DataFrame:
    df = snap.reader.read_table("silver.player_weekly_status")
    df = df[(df["team"] == team) & (df["season"] == season) & (df["week"] <= week)]
    if df.empty:
        return pd.DataFrame()
    return df[df["week"] == df["week"].max()]
