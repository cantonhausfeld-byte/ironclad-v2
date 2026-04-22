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

# Positions we project
SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}
MIN_TARGETS_FOR_RECEIVER = 2  # must have at least 2 targets in recent games to include


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

        # Load team features for game context
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
            team_row_feats = team_feats[team_feats["team"] == team]

            for _, player in players.iterrows():
                pos = player.get("position", "UNK")
                if pos not in SKILL_POSITIONS:
                    continue
                try:
                    row = self._build_player_row(
                        player=player,
                        team=team,
                        opponent=opponent,
                        is_home=is_home,
                        game=game,
                        snap=snap,
                        opp_feats=opp_feats,
                        team_feats_row=team_row_feats,
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
        opp_feats: pd.DataFrame,
        team_feats_row: pd.DataFrame,
    ) -> dict:
        pid = str(player["player_id"])
        pos = player.get("position", "UNK")
        recent = snap.player_recent_games(pid, n=ROLLING_WINDOW)

        # Team pass attempts in recent games (for share calculation)
        team_recent = snap.team_recent_games(team, n=ROLLING_WINDOW)
        team_pass_att = team_recent["pass_attempts"].mean() if not team_recent.empty and "pass_attempts" in team_recent.columns else 30.0
        team_rush_att = team_recent["rush_attempts"].mean() if not team_recent.empty and "rush_attempts" in team_recent.columns else 25.0
        team_pass_att = max(team_pass_att or 30.0, 1.0)
        team_rush_att = max(team_rush_att or 25.0, 1.0)

        def avg(col, default=None):
            if recent.empty or col not in recent.columns:
                return default
            vals = recent[col].dropna()
            return float(vals.mean()) if len(vals) > 0 else default

        # Usage
        targets_avg = avg("targets", 0.0)
        carries_avg = avg("carries", 0.0)
        rec_avg = avg("receptions", 0.0)

        target_share = targets_avg / team_pass_att
        carry_share = carries_avg / team_rush_att
        catch_rate = safe_divide(pd.Series([rec_avg]), pd.Series([targets_avg]), 0.0).iloc[0] if targets_avg > 0 else None
        yds_per_tgt = safe_divide(
            pd.Series([avg("rec_yards", 0.0)]), pd.Series([targets_avg]), 0.0
        ).iloc[0] if targets_avg > 0 else None
        yds_per_carry = safe_divide(
            pd.Series([avg("rush_yards", 0.0)]), pd.Series([carries_avg]), 0.0
        ).iloc[0] if carries_avg > 0 else None
        yac = safe_divide(
            pd.Series([avg("yards_after_catch", 0.0)]), pd.Series([rec_avg]), 0.0
        ).iloc[0] if rec_avg and rec_avg > 0 else None
        td_per_tgt = safe_divide(
            pd.Series([avg("total_tds", 0.0)]), pd.Series([targets_avg]), 0.0
        ).iloc[0] if targets_avg > 0 else None
        td_per_carry = safe_divide(
            pd.Series([avg("total_tds", 0.0)]), pd.Series([carries_avg]), 0.0
        ).iloc[0] if carries_avg > 0 else None

        # Opponent defense context
        def opp_feat(col, default=None):
            if opp_feats.empty or col not in opp_feats.columns:
                return default
            return opp_feats.iloc[0].get(col, default)

        def team_feat(col, default=None):
            if team_feats_row.empty or col not in team_feats_row.columns:
                return default
            return team_feats_row.iloc[0].get(col, default)

        availability = float(player.get("availability", AVAILABILITY_DEFAULT))
        depth_team = player.get("depth_team")

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
            "snap_rate_l4": None,  # snap data not in nfl_data_py PBP directly
            "target_share_l4": target_share,
            "carry_share_l4": carry_share,
            "air_yards_share_l4": safe_divide(
                pd.Series([avg("air_yards", 0.0)]),
                pd.Series([team_pass_att * 7.5]),  # approx air yards per pass
                0.0,
            ).iloc[0],
            "route_rate_l4": None,
            "redzone_target_share_l4": None,  # requires RZ filter from PBP
            "redzone_carry_share_l4": None,
            "catch_rate_l4": catch_rate,
            "yards_per_target_l4": yds_per_tgt,
            "yards_per_carry_l4": yds_per_carry,
            "yac_per_rec_l4": yac,
            "td_rate_per_target_l4": td_per_tgt,
            "td_rate_per_carry_l4": td_per_carry,
            "opp_def_pass_epa_l4": opp_feat("def_epa_per_play_l4"),
            "opp_def_rush_epa_l4": opp_feat("def_epa_per_play_l4"),
            "opp_def_sack_rate_l4": opp_feat("def_sack_rate_l4"),
            "team_off_pass_rate_l4": team_feat("off_pass_rate_l4"),
            "team_off_epa_l4": team_feat("off_epa_per_play_l4"),
            "team_implied_total": team_feat("implied_total_from_odds"),
            # Targets (post-game)
            "target_targets": None,
            "target_carries": None,
            "target_receptions": None,
            "target_rec_yards": None,
            "target_rush_yards": None,
            "target_total_tds": None,
        }

        feat_vals = pd.Series({
            k: v for k, v in feature_row.items()
            if k not in ("game_id", "season", "week", "player_id", "player_name",
                         "team", "opponent", "position", "is_home", "cutoff_ts",
                         "feature_version", "data_completeness_score",
                         "target_targets", "target_carries", "target_receptions",
                         "target_rec_yards", "target_rush_yards", "target_total_tds")
        })
        feature_row["data_completeness_score"] = completeness_score(feat_vals)
        return feature_row


def _fallback_roster(team: str, season: int, week: int, snap: FeatureSnapshot) -> pd.DataFrame:
    """Fall back to latest available roster week for this team."""
    df = snap.reader.read_as_of("silver.player_weekly_status")
    df = df[(df["team"] == team) & (df["season"] == season) & (df["week"] <= week)]
    if df.empty:
        return pd.DataFrame()
    latest_week = df["week"].max()
    return df[df["week"] == latest_week]
