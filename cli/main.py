"""ironclad CLI — unified NFL matchup forecasting system."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from ironclad.config import REPORTS_DIR, DEFAULT_N_DRAWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
@click.version_option(package_name="ironclad-v2")
def cli() -> None:
    """ironclad-v2: NFL matchup forecasting."""


# ── backfill ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--seasons", "-s", multiple=True, type=int,
              help="Seasons to backfill (can repeat: -s 2022 -s 2023)")
@click.option("--season", type=int, default=None,
              help="Single season shorthand")
@click.option("--no-pbp", is_flag=True, default=False,
              help="Skip play-by-play (faster, no stats)")
def backfill(seasons, season, no_pbp) -> None:
    """Ingest historical data and build silver tables."""
    all_seasons = list(seasons)
    if season:
        all_seasons.append(season)
    if not all_seasons:
        click.echo("ERROR: Provide at least one --season or --seasons value", err=True)
        sys.exit(1)

    click.echo(f"Backfilling seasons: {sorted(all_seasons)}")
    from ironclad.workflow.backfill import BackfillWorkflow
    BackfillWorkflow().run(sorted(all_seasons), include_pbp=not no_pbp)
    click.echo("Done.")


# ── weekly ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--season", required=True, type=int)
@click.option("--week", required=True, type=int)
def weekly(season, week) -> None:
    """Run weekly refresh: ingest current data and rebuild silver tables."""
    click.echo(f"Running weekly refresh: season={season} week={week}")
    from ironclad.workflow.weekly import WeeklyWorkflow
    WeeklyWorkflow().run(season, week)
    click.echo("Done.")


# ── report ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--game-id", default=None, help="Canonical game ID (e.g. 2023_18_KC_LAC)")
@click.option("--home", default=None, help="Home team abbreviation")
@click.option("--away", default=None, help="Away team abbreviation")
@click.option("--week", default=None, type=int, help="Week number")
@click.option("--season", default=None, type=int, help="Season year")
@click.option("--format", "fmt", default="markdown",
              type=click.Choice(["markdown", "both"]),
              help="Output format")
@click.option("--output-dir", default=str(REPORTS_DIR), show_default=True,
              help="Directory to write report files")
@click.option("--n-draws", default=DEFAULT_N_DRAWS, show_default=True, type=int,
              help="Number of Monte Carlo draws")
@click.option("--backfill-if-missing", is_flag=True, default=False,
              help="Auto-backfill season data if game not found")
def report(game_id, home, away, week, season, fmt, output_dir, n_draws, backfill_if_missing) -> None:
    """Generate a pregame matchup report."""
    # Resolve game_id from team names if not provided
    if not game_id:
        if not (home and away and week and season):
            click.echo("ERROR: Provide --game-id OR all of --home --away --week --season", err=True)
            sys.exit(1)
        game_id = _lookup_game_id(home, away, week, season)
        if not game_id:
            click.echo(f"ERROR: Could not find game {away}@{home} week {week} {season}", err=True)
            sys.exit(1)

    click.echo(f"Generating report for {game_id}  (n_draws={n_draws})")
    from ironclad.workflow.matchup import MatchupWorkflow
    try:
        path = MatchupWorkflow(n_draws=n_draws).run(
            game_id=game_id,
            output_dir=Path(output_dir),
            fmt=fmt,
            backfill_if_missing=backfill_if_missing,
        )
        click.echo(f"\nReport written to: {path}")
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)


def _lookup_game_id(home: str, away: str, week: int, season: int) -> str | None:
    try:
        from ironclad.store.connection import get_connection
        conn = get_connection()
        df = conn.execute("""
            SELECT game_id FROM silver.games
            WHERE season = ? AND week = ? AND home_team = ? AND away_team = ?
        """, [season, week, home.upper(), away.upper()]).df()
        return str(df.iloc[0]["game_id"]) if not df.empty else None
    except Exception:
        return None


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
def status() -> None:
    """Show data and model health summary."""
    try:
        from ironclad.store.connection import get_connection
        from ironclad.store.schema import create_all_tables
        conn = get_connection()
        create_all_tables(conn)

        click.echo("\n=== ironclad-v2 status ===\n")

        for table in [
            "bronze.schedules", "bronze.play_by_play", "bronze.rosters",
            "silver.games", "silver.team_game_stats", "silver.player_game_stats",
            "gold.team_game_features", "gold.player_game_features",
        ]:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                click.echo(f"  {table:<40} {n:>8,} rows")
            except Exception:
                click.echo(f"  {table:<40}  (not found)")

        click.echo()
    except Exception as exc:
        click.echo(f"Status check failed: {exc}", err=True)


if __name__ == "__main__":
    cli()
