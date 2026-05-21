"""Player prop distribution backtester.

Runs simulations on historical games and compares projected distributions to
actual outcomes from silver.player_game_stats. Reports calibration metrics:
- actual_rank: fraction of draws < actual_value (should be uniform [0,1])
- covered_80pct: actual in [p10, p90] (should be ~80%)
- abs_error_p50: |actual - p50|
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from simulation stat name -> silver.player_game_stats column
_STAT_MAP: dict[str, str] = {
    "rec_yards":     "rec_yards",
    "rush_yards":    "rush_yards",
    "tds":           "total_tds",
    "receptions":    "receptions",
    "targets":       "targets",
    "pass_yards":    "pass_yards",
    "carries":       "carries",
    "completions":   "completions",
    "pass_attempts": "pass_attempts",
}

_DEFAULT_POSITIONS = {"WR", "RB", "TE", "QB"}

# Stats that are structurally zero for a position — skip to avoid reporting
# artifacts (e.g. WRs don't rush; non-QBs don't throw; QBs don't catch).
_POSITION_STAT_SKIP: dict[str, set[str]] = {
    "WR": {"pass_attempts", "completions", "pass_yards", "carries", "rush_yards"},
    "TE": {"pass_attempts", "completions", "pass_yards", "carries", "rush_yards"},
    "RB": {"pass_attempts", "completions", "pass_yards"},
    "QB": {"targets", "receptions", "rec_yards"},
}


class PropBacktester:
    def __init__(self, conn=None) -> None:
        from ironclad.store.connection import get_connection
        from ironclad.store.schema import create_all_tables
        self._conn = conn or get_connection()
        create_all_tables(self._conn)

    def run(
        self,
        season: int,
        weeks: list[int] | None = None,
        n_draws: int = 500,
        positions: set[str] | None = None,
        save: bool = False,
    ) -> pd.DataFrame:
        """Simulate each completed game in season, compare projections to actuals.

        Returns one row per player × stat_type × game with calibration metrics.
        """
        positions = positions or _DEFAULT_POSITIONS
        backtest_id = f"pb_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_at = datetime.now(timezone.utc)

        games = self._load_games(season, weeks)
        if games.empty:
            logger.warning("No completed REG games found for season=%d weeks=%s", season, weeks)
            return pd.DataFrame()

        actuals = self._load_actuals(season, positions)

        rows: list[dict] = []
        for _, game in games.iterrows():
            game_id = game["game_id"]
            week = int(game["week"])
            game_rows = self._simulate_game(
                game_id=game_id,
                season=season,
                week=week,
                n_draws=n_draws,
                positions=positions,
                actuals=actuals,
                backtest_id=backtest_id,
                run_at=run_at,
            )
            rows.extend(game_rows)
            logger.info("Game %s: %d player-stat rows", game_id, len(game_rows))

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if save:
            self._save(df)

        return df

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_games(self, season: int, weeks: list[int] | None) -> pd.DataFrame:
        """Load completed REG games for the season, optionally filtered by weeks."""
        week_filter = ""
        params: list = [season]
        if weeks:
            placeholders = ",".join("?" * len(weeks))
            week_filter = f"AND week IN ({placeholders})"
            params.extend(weeks)

        return self._conn.execute(f"""
            SELECT game_id, week
            FROM silver.games
            WHERE season = ?
              AND season_type = 'REG'
              AND home_score IS NOT NULL
              {week_filter}
            ORDER BY week, game_id
        """, params).df()

    def _load_actuals(self, season: int, positions: set[str]) -> pd.DataFrame:
        """Load actual player game stats for the season, filtered to positions."""
        pos_list = ",".join(f"'{p}'" for p in positions)
        return self._conn.execute(f"""
            SELECT game_id, player_id, player_name, team, position,
                   pass_attempts, completions, pass_yards, carries, rush_yards,
                   targets, receptions, rec_yards, total_tds
            FROM silver.player_game_stats
            WHERE season = ?
              AND position IN ({pos_list})
        """, [season]).df()

    def _simulate_game(
        self,
        game_id: str,
        season: int,
        week: int,
        n_draws: int,
        positions: set[str],
        actuals: pd.DataFrame,
        backtest_id: str,
        run_at: datetime,
    ) -> list[dict]:
        from ironclad.workflow.matchup import MatchupWorkflow
        try:
            result, _, _, _ = MatchupWorkflow(n_draws=n_draws).simulate(game_id)
        except Exception as exc:
            logger.warning("Simulation failed for %s: %s", game_id, exc)
            return []

        game_actuals = actuals[actuals["game_id"] == game_id]
        if game_actuals.empty:
            logger.debug("No actual stats found for game %s", game_id)
            return []

        # Build draw-level arrays keyed by (player_id, stat_type)
        draw_arrays = self._build_draw_arrays(result, positions)
        if not draw_arrays:
            return []

        rows: list[dict] = []
        for _, actual_row in game_actuals.iterrows():
            pid = actual_row["player_id"]
            for sim_stat, silver_col in _STAT_MAP.items():
                key = (pid, sim_stat)
                if key not in draw_arrays:
                    continue

                actual_val = actual_row.get(silver_col)
                if actual_val is None or (isinstance(actual_val, float) and np.isnan(actual_val)):
                    continue
                actual_val = float(actual_val)

                # Skip structurally inapplicable stats for this position
                position = str(actual_row.get("position", ""))
                if sim_stat in _POSITION_STAT_SKIP.get(position, set()):
                    continue

                # Skip players who did not participate (DNP).
                if sim_stat in ("rec_yards", "targets", "receptions") and actual_row.get("targets", 0) == 0:
                    continue
                if sim_stat in ("rush_yards", "carries") and actual_row.get("carries", 0) == 0:
                    continue
                if sim_stat in ("pass_yards", "completions", "pass_attempts") and actual_row.get("pass_attempts", 0) == 0:
                    continue

                draws_arr = draw_arrays[key]
                actual_rank = float(np.mean(draws_arr < actual_val))
                p10 = float(np.percentile(draws_arr, 10))
                p25 = float(np.percentile(draws_arr, 25))
                p50 = float(np.percentile(draws_arr, 50))
                p75 = float(np.percentile(draws_arr, 75))
                p90 = float(np.percentile(draws_arr, 90))
                mean_val = float(np.mean(draws_arr))

                rows.append({
                    "backtest_id":    backtest_id,
                    "run_at":         run_at,
                    "season":         season,
                    "week":           week,
                    "game_id":        game_id,
                    "player_id":      pid,
                    "player_name":    actual_row.get("player_name"),
                    "team":           actual_row.get("team"),
                    "position":       actual_row.get("position"),
                    "stat_type":      sim_stat,
                    "n_draws":        len(draws_arr),
                    "predicted_p10":  p10,
                    "predicted_p25":  p25,
                    "predicted_p50":  p50,
                    "predicted_p75":  p75,
                    "predicted_p90":  p90,
                    "predicted_mean": mean_val,
                    "actual_value":   actual_val,
                    "actual_rank":    actual_rank,
                    "covered_80pct":  bool(p10 <= actual_val <= p90),
                    "abs_error_p50":  abs(actual_val - p50),
                })

        return rows

    def _build_draw_arrays(
        self,
        result,
        positions: set[str],
    ) -> dict[tuple[str, str], np.ndarray]:
        """Build arrays of per-draw values keyed by (player_id, stat_type)."""
        accum: dict[tuple[str, str], list[float]] = {}

        for draw in result._draws:
            for pstat in draw.player_stats:
                pos = pstat.get("position", "")
                if pos not in positions:
                    continue
                pid = pstat.get("player_id", "")
                if not pid:
                    continue
                for sim_col in _STAT_MAP:
                    val = pstat.get(sim_col)
                    if val is None:
                        continue
                    key = (pid, sim_col)
                    if key not in accum:
                        accum[key] = []
                    accum[key].append(float(val))

        return {k: np.array(v) for k, v in accum.items() if len(v) > 0}

    def _save(self, df: pd.DataFrame) -> None:
        self._conn.register("_prop_bt_tmp", df)
        self._conn.execute("""
            INSERT OR REPLACE INTO gold.player_prop_backtest
            SELECT * FROM _prop_bt_tmp
        """)
        self._conn.unregister("_prop_bt_tmp")
        logger.info("Saved %d prop backtest rows", len(df))

    # ── Summary ────────────────────────────────────────────────────────────────

    def load_latest(self) -> pd.DataFrame:
        return self._conn.execute("""
            SELECT * FROM gold.player_prop_backtest
            WHERE backtest_id = (
                SELECT backtest_id FROM gold.player_prop_backtest
                ORDER BY run_at DESC LIMIT 1
            )
        """).df()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate calibration metrics by stat_type × position."""
    if df.empty:
        return pd.DataFrame(columns=[
            "stat_type", "position", "N", "coverage_80pct", "median_p50_mae", "bias_p50"
        ])

    rows = []
    groups = df.groupby(["stat_type", "position"])
    for (stat_type, position), grp in groups:
        rows.append({
            "stat_type":       stat_type,
            "position":        position,
            "N":               len(grp),
            "coverage_80pct":  float(grp["covered_80pct"].mean()),
            "median_p50_mae":  float(grp["abs_error_p50"].median()),
            "bias_p50":        float((grp["actual_value"] - grp["predicted_p50"]).mean()),
        })

    # Also compute an "all" position row per stat_type
    for stat_type, grp in df.groupby("stat_type"):
        rows.append({
            "stat_type":       stat_type,
            "position":        "all",
            "N":               len(grp),
            "coverage_80pct":  float(grp["covered_80pct"].mean()),
            "median_p50_mae":  float(grp["abs_error_p50"].median()),
            "bias_p50":        float((grp["actual_value"] - grp["predicted_p50"]).mean()),
        })

    return pd.DataFrame(rows).sort_values(["stat_type", "position"]).reset_index(drop=True)
