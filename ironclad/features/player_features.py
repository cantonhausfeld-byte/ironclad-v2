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

# Position priors for Bayesian shrinkage (league-average rates, 2016–2024)
_PRIORS: dict[str, dict] = {
    "WR": {"catch_rate": 0.64, "yards_per_target": 7.0, "adot": 8.5},
    "TE": {"catch_rate": 0.68, "yards_per_target": 6.5, "adot": 6.0},
    "RB": {"catch_rate": 0.73, "yards_per_target": 5.0, "adot": 3.5},
    "QB": {"catch_rate": 0.65, "yards_per_target": 7.2, "adot": 10.5},
    "FB": {"catch_rate": 0.70, "yards_per_target": 5.5, "adot": 4.0},
}
_DEFAULT_PRIOR = {"catch_rate": 0.65, "yards_per_target": 6.5, "adot": 7.0}
_K_PRIOR = 4  # prior-equivalent games (shrinks toward prior when n_games < 4)


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
            # Always supplement from roster to capture healthy players not on injury report
            players = _supplement_from_roster(players, team, season, week, snap)

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
        n_games = len(recent)
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
        prior = _PRIORS.get(pos, _DEFAULT_PRIOR)
        shrinkage = n_games / (n_games + _K_PRIOR)  # 0 when n_games=0, →1 as n_games→∞

        # aDOT (avg depth of target) — from rolling air_yards / targets sums
        if not recent.empty and "air_yards" in recent.columns and "targets" in recent.columns:
            air_sum = float(recent["air_yards"].sum())
            tgt_sum = float(recent["targets"].sum())
            adot_raw = air_sum / tgt_sum if tgt_sum > 0 else None
        else:
            adot_raw = None
        adot_l4 = (
            shrinkage * adot_raw + (1 - shrinkage) * prior["adot"]
            if adot_raw is not None
            else (prior["adot"] if n_games == 0 else None)
        )

        catch_rate_raw = rec_avg / (targets_avg + eps) if targets_avg > 0 else None
        yds_per_tgt_raw = avg("rec_yards", 0.0) / (targets_avg + eps) if targets_avg > 0 else None
        yds_per_carry = avg("rush_yards", 0.0) / (carries_avg + eps) if carries_avg > 0 else None
        yac_per_rec = avg("yards_after_catch", 0.0) / (rec_avg + eps) if rec_avg > 0 else None
        total_tds_avg = avg("total_tds", 0.0)
        td_per_tgt = total_tds_avg / (targets_avg + eps) if targets_avg > 0 else None
        td_per_carry = total_tds_avg / (carries_avg + eps) if carries_avg > 0 else None

        # Bayesian shrinkage: blend raw rate toward position prior when sample size is small.
        # Prevents week-1 QB with 2 receiving targets from inflating catch_rate to 1.0.
        if catch_rate_raw is not None:
            catch_rate = shrinkage * max(0.0, min(1.0, catch_rate_raw)) + (1 - shrinkage) * prior["catch_rate"]
        else:
            catch_rate = prior["catch_rate"] if n_games == 0 else None

        if yds_per_tgt_raw is not None:
            yds_per_tgt = shrinkage * yds_per_tgt_raw + (1 - shrinkage) * prior["yards_per_target"]
        else:
            yds_per_tgt = prior["yards_per_target"] if n_games == 0 else None

        # Clamp TD rates (no shrinkage — priors are poorly constrained for rare events)
        if td_per_tgt is not None:
            td_per_tgt = max(0.0, min(0.20, td_per_tgt))
        if td_per_carry is not None:
            td_per_carry = max(0.0, min(0.15, td_per_carry))

        def opp_feat(col, default=None):
            if opp_feats.empty or col not in opp_feats.columns:
                return default
            return opp_feats.iloc[0].get(col, default)

        def own_feat(col, default=None):
            if own_feats.empty or col not in own_feats.columns:
                return default
            return own_feats.iloc[0].get(col, default)

        # NGS tracking features (L4 avg)
        season = int(game["season"])
        week = int(game["week"])
        ngs_recv = _ngs_avg(snap.player_recent_ngs(pid, "receiving", season, week))
        ngs_rush = _ngs_avg(snap.player_recent_ngs(pid, "rushing",   season, week))
        ngs_pass = _ngs_avg(snap.player_recent_ngs(pid, "passing",   season, week))

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
            "adot_l4":                  adot_l4,
            "opp_def_pass_epa_l4":      opp_feat("def_epa_per_play_l4"),
            "opp_def_rush_epa_l4":      opp_feat("def_rush_epa_l4"),
            "opp_def_sack_rate_l4":     opp_feat("def_sack_rate_l4"),
            "team_off_pass_rate_l4":    own_feat("off_pass_rate_l4"),
            "team_off_epa_l4":          own_feat("off_epa_per_play_l4"),
            "team_implied_total":       own_feat("implied_total_from_odds"),
            # NGS tracking
            "ngs_avg_separation":                   ngs_recv.get("avg_separation"),
            "ngs_rush_yards_over_expected":         ngs_rush.get("rush_yards_over_expected"),
            "ngs_completion_pct_above_expected":    ngs_pass.get("completion_percentage_above_expectation"),
            # Targets filled post-game
            "target_targets": None, "target_carries": None,
            "target_receptions": None, "target_rec_yards": None,
            "target_rush_yards": None, "target_total_tds": None,
            "data_completeness_score": min(1.0, n_games / ROLLING_WINDOW),
        }
        return feature_row


def _ngs_avg(df: pd.DataFrame) -> dict:
    """Return column-wise means from an NGS DataFrame, as a plain dict."""
    if df.empty:
        return {}
    return {col: float(df[col].dropna().mean()) for col in df.columns if df[col].notna().any()}


def _avg_col(df: pd.DataFrame, col: str, default=None):
    """Mean of a column across a DataFrame; returns default if absent or all-null."""
    if df.empty or col not in df.columns:
        return default
    vals = df[col].dropna()
    return float(vals.mean()) if len(vals) else default


def _supplement_from_roster(
    players: pd.DataFrame,
    team: str,
    season: int,
    week: int,
    snap: FeatureSnapshot,
) -> pd.DataFrame:
    """Add roster players not already in players (e.g. healthy starters absent from injury report)."""
    roster = snap.reader.read_table("bronze.rosters")
    roster = roster[(roster["team"] == team) & (roster["season"] == season) & (roster["week"] <= week)]
    if roster.empty:
        return players
    roster = roster[roster["week"] == roster["week"].max()].copy()
    # Keep only skill positions worth projecting
    if "position" in roster.columns:
        roster = roster[roster["position"].isin(SKILL_POSITIONS)]
    roster["availability"] = 1.0
    roster["depth_team"] = None
    # Remove players already covered by injury/depth chart data
    if not players.empty and "player_id" in players.columns:
        covered = set(players["player_id"].astype(str))
        roster = roster[~roster["player_id"].astype(str).isin(covered)]
    if roster.empty:
        return players
    return pd.concat([players, roster], ignore_index=True) if not players.empty else roster
