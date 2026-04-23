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
    """Run weekly refresh: ingest → silver → gold features → target backfill."""
    click.echo(f"Running weekly refresh: season={season} week={week}")
    from ironclad.workflow.weekly import WeeklyWorkflow
    result = WeeklyWorkflow().run(season, week)
    click.echo(f"\nIngest:          {result.get('ingest', {})}")
    click.echo(f"Silver:          {result.get('silver', {})}")
    click.echo(f"Features built:  {result.get('features_built', 0)} games")
    click.echo(f"Targets:         {result.get('targets', {})}")
    click.echo("\nDone.")


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

# ── train ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--train-seasons", default="2016-2022",
              help="Season range for training, e.g. '2016-2022' or '2018,2019,2020'")
@click.option("--val-seasons", default="2023",
              help="Season range for validation/calibration")
@click.option("--model", default="all",
              type=click.Choice(["all", "game-outcome", "score-env", "player-usage", "player-efficiency"]),
              help="Which model(s) to train")
def train(train_seasons, val_seasons, model) -> None:
    """Train ML models on backfilled historical data."""
    train_list = _parse_seasons(train_seasons)
    val_list = _parse_seasons(val_seasons)
    if not train_list:
        click.echo("ERROR: Could not parse --train-seasons", err=True)
        sys.exit(1)

    click.echo(f"Training seasons: {train_list}  |  Validation: {val_list}")
    from ironclad.models.trainer import ModelTrainer
    trainer = ModelTrainer()

    if model == "all":
        metrics = trainer.train_all(train_list, val_list or None)
    elif model == "game-outcome":
        metrics = {"game_outcome": trainer.train_game_outcome(train_list, val_list or None)}
    elif model == "score-env":
        metrics = {"score_env": trainer.train_score_env(train_list)}
    elif model == "player-usage":
        metrics = {"player_usage": trainer.train_player_usage(train_list)}
    elif model == "player-efficiency":
        metrics = {"player_efficiency": trainer.train_player_efficiency(train_list)}

    click.echo("\nTraining complete. Metrics:")
    for name, m in metrics.items():
        click.echo(f"  {name}: {m}")


# ── evaluate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--val-seasons", default="2023",
              help="Seasons to evaluate against (comma-separated or range)")
def evaluate(val_seasons) -> None:
    """Evaluate trained models against held-out seasons."""
    val_list = _parse_seasons(val_seasons)
    if not val_list:
        click.echo("ERROR: Could not parse --val-seasons", err=True)
        sys.exit(1)

    click.echo(f"Evaluating on seasons: {val_list}")
    from ironclad.models.trainer import ModelTrainer
    trainer = ModelTrainer()
    metrics = trainer.evaluate(val_list)

    if metrics:
        click.echo("\nEvaluation metrics:")
        for k, v in metrics.items():
            click.echo(f"  {k}: {v}")
    else:
        click.echo("No metrics available (run ironclad train first)")


def _parse_seasons(spec: str) -> list[int]:
    """Parse '2016-2022' or '2018,2019,2020' into a list of ints."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        parts = spec.split("-")
        if len(parts) == 2:
            try:
                return list(range(int(parts[0]), int(parts[1]) + 1))
            except ValueError:
                pass
    try:
        return [int(s.strip()) for s in spec.split(",")]
    except ValueError:
        return []


# ── status ─────────────────────────────────────────────────────────────────────

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
            "bronze.snap_counts", "bronze.player_stats_weekly",
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
