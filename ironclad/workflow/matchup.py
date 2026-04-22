"""End-to-end single matchup report generation."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ironclad.config import (
    DEFAULT_N_DRAWS,
    KNOWLEDGE_CUTOFF_MARGIN_MINUTES,
    REPORTS_DIR,
)
from ironclad.features.team_features import TeamFeatureBuilder, _parse_kickoff
from ironclad.features.player_features import PlayerFeatureBuilder
from ironclad.simulation.engine import MonteCarloEngine
from ironclad.report.builder import build_report_context
from ironclad.report.markdown_renderer import render_markdown
from ironclad.report.csv_exporter import export_player_csv, export_team_csv
from ironclad.store.connection import get_connection
from ironclad.store.schema import create_all_tables

logger = logging.getLogger(__name__)


class MatchupWorkflow:
    def __init__(self, n_draws: int = DEFAULT_N_DRAWS, seed: int = 42) -> None:
        self.n_draws = n_draws
        self.seed = seed

    def run(
        self,
        game_id: str,
        output_dir: Path = REPORTS_DIR,
        fmt: str = "markdown",
        backfill_if_missing: bool = False,
    ) -> Path:
        conn = get_connection()
        create_all_tables(conn)

        # Check game exists in silver
        game_row = self._get_game(conn, game_id)
        if game_row is None:
            if backfill_if_missing:
                season = int(game_id.split("_")[0])
                logger.info("Game not found; backfilling season %d", season)
                from ironclad.workflow.backfill import BackfillWorkflow
                BackfillWorkflow().run([season])
                game_row = self._get_game(conn, game_id)
            if game_row is None:
                raise ValueError(f"Game {game_id} not found in silver.games after backfill")

        cutoff_ts = self._compute_cutoff(game_row)
        logger.info("Using knowledge cutoff: %s", cutoff_ts)

        # Build features
        logger.info("Building team features...")
        team_builder = TeamFeatureBuilder(conn)
        team_builder.build_for_game(game_id, cutoff_ts)

        logger.info("Building player features...")
        player_builder = PlayerFeatureBuilder(conn)
        player_builder.build_for_game(game_id, cutoff_ts)

        # Load gold features
        home_team = str(game_row["home_team"])
        away_team = str(game_row["away_team"])

        home_feats = conn.execute(
            "SELECT * FROM gold.team_game_features WHERE game_id = ? AND team = ?",
            [game_id, home_team],
        ).df()
        away_feats = conn.execute(
            "SELECT * FROM gold.team_game_features WHERE game_id = ? AND team = ?",
            [game_id, away_team],
        ).df()
        home_player_feats = conn.execute(
            "SELECT * FROM gold.player_game_features WHERE game_id = ? AND team = ?",
            [game_id, home_team],
        ).df()
        away_player_feats = conn.execute(
            "SELECT * FROM gold.player_game_features WHERE game_id = ? AND team = ?",
            [game_id, away_team],
        ).df()

        if home_feats.empty or away_feats.empty:
            raise RuntimeError(f"Team features missing for {game_id}")

        # Run simulation
        engine = MonteCarloEngine(n_draws=self.n_draws, seed=self.seed)
        result = engine.run(
            home_team=home_team,
            away_team=away_team,
            home_features=home_feats,
            away_features=away_feats,
            home_player_features=home_player_feats,
            away_player_features=away_player_feats,
        )

        # Build report context
        game_meta = dict(game_row)
        game_meta["stadium"] = game_meta.get("stadium_id", "Unknown Stadium")
        game_meta["data_completeness_score"] = float(home_feats.get("data_completeness_score", pd.Series([0.5])).iloc[0]) if not home_feats.empty else 0.5

        ctx = build_report_context(
            result=result,
            game_meta=game_meta,
            cutoff_ts=cutoff_ts,
            n_draws=self.n_draws,
        )

        # Render outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_str = cutoff_ts.strftime("%Y%m%dT%H%M%SZ")
        base = output_dir / f"{game_id}_pregame_{ts_str}"

        md_path = render_markdown(ctx, Path(str(base) + ".md"))
        export_player_csv(result, Path(str(base) + "_players.csv"))
        export_team_csv(result, Path(str(base) + "_teams.csv"))

        logger.info("Report written to %s", md_path)
        return md_path

    def _get_game(self, conn, game_id: str) -> pd.Series | None:
        df = conn.execute(
            "SELECT * FROM silver.games WHERE game_id = ?", [game_id]
        ).df()
        return df.iloc[0] if not df.empty else None

    def _compute_cutoff(self, game_row: pd.Series) -> datetime:
        kickoff = _parse_kickoff(game_row["gameday"], game_row.get("gametime_local"))
        return kickoff - timedelta(minutes=KNOWLEDGE_CUTOFF_MARGIN_MINUTES)
