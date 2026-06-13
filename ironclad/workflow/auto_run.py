"""Weekly auto-run: data refresh → odds → simulate all games → save edges."""
from __future__ import annotations

import logging
from datetime import date

from ironclad.store.connection import get_connection
from ironclad.store.schema import create_all_tables

logger = logging.getLogger(__name__)


def detect_current_week(conn) -> tuple[int, int]:
    """Return (season, week) of the next upcoming REG-season game."""
    row = conn.execute("""
        SELECT season, week FROM bronze.schedules
        WHERE gameday >= CURRENT_DATE AND season_type = 'REG'
        ORDER BY gameday ASC LIMIT 1
    """).fetchone()
    if not row:
        raise ValueError(
            "No upcoming regular-season games found in bronze.schedules. "
            "Run `ironclad backfill --season <year>` first, or pass --season/--week explicitly."
        )
    return int(row[0]), int(row[1])


class AutoRunWorkflow:
    def __init__(
        self,
        n_draws: int = 5000,
        kelly_fraction: float = 0.25,
        min_ev: float = 0.0,
        skip_odds: bool = False,
        historical: bool = False,
    ) -> None:
        self.n_draws = n_draws
        self.kelly_fraction = kelly_fraction
        self.min_ev = min_ev
        self.skip_odds = skip_odds
        self.historical = historical  # allow targeting completed weeks (no gameday >= today filter)

    def run(self, season: int | None = None, week: int | None = None) -> dict:
        conn = get_connection()
        create_all_tables(conn)

        # ── 1. Auto-detect week if not provided ──────────────────────────────
        if season is None or week is None:
            season, week = detect_current_week(conn)
        logger.info("Auto-run: season=%d week=%d historical=%s", season, week, self.historical)

        # ── 2. Data pipeline refresh ─────────────────────────────────────────
        from ironclad.workflow.weekly import WeeklyWorkflow
        weekly_result = WeeklyWorkflow().run(season, week)

        # ── 3. Injury refresh + odds + player props ───────────────────────────
        from ironclad.store.writer import BronzeWriter
        writer = BronzeWriter(conn=conn)

        # Refresh injury reports for the current season so availability reflects
        # the latest practice reports (updated Wed–Fri before each game).
        try:
            from ironclad.ingest.injuries import InjuryIngestor
            inj_rows = InjuryIngestor(writer=writer).ingest(seasons=[season])
            logger.info("Injury refresh: %d rows for season %d", inj_rows, season)
        except Exception as exc:
            logger.warning("Injury refresh failed (continuing): %s", exc)

        odds_rows = 0
        if not self.skip_odds:
            from ironclad.ingest.odds import OddsIngestor
            try:
                odds_rows = OddsIngestor(writer=writer, conn=conn).ingest()
                logger.info("Odds ingest: %d rows", odds_rows)
            except EnvironmentError:
                raise  # missing ODDS_API_KEY is a config error, not transient
            except Exception as exc:
                logger.warning("Odds fetch failed (continuing without props): %s", exc)

        # ── 4. Simulate games + save edges ────────────────────────────────────
        # In historical mode: include completed games (no date filter).
        # In normal mode: only upcoming games (gameday >= today).
        if self.historical:
            games_df = conn.execute("""
                SELECT game_id FROM silver.games
                WHERE season = ? AND week = ? AND season_type = 'REG'
                ORDER BY gameday
            """, [season, week]).df()
        else:
            today = date.today().isoformat()
            games_df = conn.execute("""
                SELECT game_id FROM silver.games
                WHERE season = ? AND week = ? AND season_type = 'REG'
                  AND CAST(gameday AS VARCHAR) >= ?
                ORDER BY gameday
            """, [season, week, today]).df()

        from ironclad.betting.props import PropAnalyzer, load_prop_lines_from_db
        from ironclad.eval.performance_tracker import save_edges
        from ironclad.workflow.matchup import MatchupWorkflow

        games_simulated = 0
        edges_saved = 0
        skipped: list[str] = []

        from ironclad import config as _cfg
        _discord_enabled = bool(_cfg.DISCORD_WEBHOOK_URL)

        for game_id in games_df["game_id"].tolist():
            try:
                prop_lines = load_prop_lines_from_db(conn, game_id)
                if not prop_lines:
                    skipped.append(f"{game_id} (no props in DB)")
                    continue

                result, game_meta, cutoff_ts, _ = MatchupWorkflow(n_draws=self.n_draws).simulate(game_id)

                edges_df = PropAnalyzer(kelly_fraction=self.kelly_fraction).analyze(
                    result, prop_lines
                )
                if self.min_ev > 0 and not edges_df.empty:
                    edges_df = edges_df[edges_df["ev"] >= self.min_ev].reset_index(drop=True)

                if not edges_df.empty:
                    ids = save_edges(conn, game_id, self.n_draws, edges_df)
                    edges_saved += len(ids)

                games_simulated += 1

                # Post model output to Discord if webhook is configured and edges were found
                if _discord_enabled and not edges_df.empty:
                    try:
                        from ironclad.notifications.discord import post_model_output
                        from ironclad.report.builder import build_report_context
                        from ironclad.report.markdown_renderer import render_markdown_str
                        hw, aw = result.win_probability()
                        scores = result.score_summary()
                        ctx = build_report_context(result, game_meta, cutoff_ts, n_draws=self.n_draws)
                        report_md = render_markdown_str(ctx)
                        top = edges_df.sort_values("ev", ascending=False)
                        post_model_output(
                            game_id=game_id,
                            report_md=report_md,
                            top_edges_df=top,
                            win_prob_home=hw,
                            win_prob_away=aw,
                            home_team=result.home_team,
                            away_team=result.away_team,
                            projected_home=round(scores["home_score_mean"]),
                            projected_away=round(scores["away_score_mean"]),
                            season=game_meta.get("season"),
                            week=game_meta.get("week"),
                        )
                    except Exception as exc:
                        logger.warning("Discord model output failed for %s: %s", game_id, exc)

            except Exception as exc:
                logger.warning("Game %s failed: %s", game_id, exc)
                skipped.append(f"{game_id} (error: {exc})")

        return {
            "season": season,
            "week": week,
            "weekly": weekly_result,
            "odds_rows": odds_rows,
            "games_total": len(games_df),
            "games_simulated": games_simulated,
            "edges_saved": edges_saved,
            "skipped": skipped,
        }
