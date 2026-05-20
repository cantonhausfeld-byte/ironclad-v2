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


def _validate_season(ctx, param, value):
    if value is not None and not (1999 <= value <= 2030):
        raise click.BadParameter(f"Season must be between 1999 and 2030, got {value}")
    return value


def _validate_seasons(ctx, param, value):
    for s in value:
        if not (1999 <= s <= 2030):
            raise click.BadParameter(f"Season must be between 1999 and 2030, got {s}")
    return value


def _validate_week(ctx, param, value):
    if value is not None and not (1 <= value <= 23):
        raise click.BadParameter(f"Week must be between 1 and 23, got {value}")
    return value


# ── backfill ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--seasons", "-s", multiple=True, type=int, callback=_validate_seasons,
              help="Seasons to backfill (can repeat: -s 2022 -s 2023)")
@click.option("--season", type=int, default=None, callback=_validate_season, is_eager=False,
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
@click.option("--season", required=True, type=int, callback=_validate_season, is_eager=False)
@click.option("--week", required=True, type=int, callback=_validate_week, is_eager=False)
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
@click.option("--week", default=None, type=int, callback=_validate_week, is_eager=False, help="Week number")
@click.option("--season", default=None, type=int, callback=_validate_season, is_eager=False, help="Season year")
@click.option("--format", "fmt", default="markdown",
              type=click.Choice(["markdown", "both"]),
              help="Output format")
@click.option("--output-dir", default=str(REPORTS_DIR), show_default=True,
              help="Directory to write report files")
@click.option("--n-draws", default=DEFAULT_N_DRAWS, show_default=True, type=int,
              help="Number of Monte Carlo draws")
@click.option("--backfill-if-missing", is_flag=True, default=False,
              help="Auto-backfill season data if game not found")
@click.option("--use-drive-sim", is_flag=True, default=False,
              help="Use drive-level Markov chain simulation instead of score-distribution GameDraw")
def report(game_id, home, away, week, season, fmt, output_dir, n_draws, backfill_if_missing, use_drive_sim) -> None:
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
        path = MatchupWorkflow(n_draws=n_draws, use_drive_sim=use_drive_sim).run(
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
@click.option("--train-seasons", default="2016-2024",
              help="Season range for training, e.g. '2016-2024' or '2018,2019,2020'")
@click.option("--val-seasons", default="2025",
              help="Season range for validation/calibration")
@click.option("--model", default="all",
              type=click.Choice(["all", "game-outcome", "game-outcome-lgbm", "score-env",
                                 "player-usage", "player-efficiency", "ensemble",
                                 "bias-corrector"]),
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
    elif model == "game-outcome-lgbm":
        metrics = {"game_outcome_lgbm": trainer.train_game_outcome(train_list, val_list or None, use_lgbm=True)}
    elif model == "score-env":
        metrics = {"score_env": trainer.train_score_env(train_list)}
    elif model == "player-usage":
        metrics = {"player_usage": trainer.train_player_usage(train_list)}
    elif model == "player-efficiency":
        metrics = {"player_efficiency": trainer.train_player_efficiency(train_list)}
    elif model == "ensemble":
        metrics = {"ensemble": trainer.train_ensemble(train_list, val_list or None)}
    elif model == "bias-corrector":
        metrics = {"bias_corrector": trainer.train_bias_corrector()}

    click.echo("\nTraining complete. Metrics:")
    for name, m in metrics.items():
        click.echo(f"  {name}: {m}")


# ── backtest ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--train-start", default=2016, show_default=True, type=int,
              help="First season of training data (expanding window starts here)")
@click.option("--test-seasons", default="2018-2024",
              help="Seasons to predict (range or comma-separated, e.g. '2018-2024')")
@click.option("--run-id", default=None,
              help="Optional custom run identifier")
@click.option("--model", "model_name", default="xgb",
              type=click.Choice(["xgb", "lgbm", "ensemble"]), show_default=True,
              help="Model class: xgb (XGBoost), lgbm (LightGBM+Optuna), or ensemble")
def backtest(train_start, test_seasons, run_id, model_name) -> None:
    """Walk-forward backtest: train on [train_start..T-1], predict season T for each T."""
    test_list = _parse_seasons(test_seasons)
    if not test_list:
        click.echo("ERROR: Could not parse --test-seasons", err=True)
        import sys; sys.exit(1)

    click.echo(f"Walk-forward backtest | train_start={train_start} | test={test_list} | model={model_name}")
    click.echo("Training a fresh model for each fold — this may take several minutes...")

    from ironclad.eval.backtester import Backtester
    if model_name == "lgbm":
        from ironclad.models.team.game_outcome_lgbm import GameOutcomeLGBM
        model_class = GameOutcomeLGBM
    elif model_name == "ensemble":
        from ironclad.models.team.ensemble import GameOutcomeEnsemble
        model_class = GameOutcomeEnsemble
    else:
        model_class = None  # Backtester defaults to GameOutcomeModel
    bt = Backtester(model_class=model_class)
    preds = bt.run(train_start=train_start, test_seasons=test_list, run_id=run_id)

    if preds.empty:
        click.echo("No predictions generated. Check that gold.team_game_features is populated.")
        return

    click.echo(f"\nBacktest complete: {len(preds)} game predictions stored.")
    click.echo("Run 'ironclad dashboard' to view results.")


# ── dashboard ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--run-id", default=None,
              help="Specific backtest run ID (default: most recent)")
def dashboard(run_id) -> None:
    """Display backtest metrics dashboard from the most recent backtest run."""
    from ironclad.eval.backtester import Backtester
    from ironclad.eval.dashboard import print_dashboard

    bt = Backtester()
    if run_id:
        preds = bt.load_run(run_id)
    else:
        preds = bt.load_latest()

    if preds.empty:
        click.echo("No backtest results found. Run: ironclad backtest")
        return

    print_dashboard(preds)


# ── evaluate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--val-seasons", default="2025",
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


# ── odds ──────────────────────────────────────────────────────────────────────

@cli.command()
def odds() -> None:
    """Fetch this week's game odds and player props from The Odds API.

    Stores game-level odds in bronze.odds and player props (with resolved
    player_ids) in bronze.player_props. Run before `ironclad edges` to
    avoid needing a manual --props-file.
    """
    from ironclad.ingest.odds import OddsIngestor
    from ironclad.store.connection import get_connection
    from ironclad.store.schema import create_all_tables
    from ironclad.store.writer import BronzeWriter

    conn = get_connection()
    create_all_tables(conn)
    writer = BronzeWriter(conn=conn)

    click.echo("Fetching odds and player props from The Odds API...")
    try:
        n = OddsIngestor(writer=writer, conn=conn).ingest()
        click.echo(f"Ingested {n} odds/prop rows into bronze.")
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)


# ── run ───────────────────────────────────────────────────────────────────────

@cli.command("run")
@click.option("--season", default=None, type=int,
              help="NFL season year (auto-detected from schedule if omitted)")
@click.option("--week", default=None, type=int,
              help="Week number (auto-detected from schedule if omitted)")
@click.option("--n-draws", default=DEFAULT_N_DRAWS, show_default=True, type=int)
@click.option("--kelly-fraction", default=0.25, show_default=True, type=float)
@click.option("--min-ev", default=0.0, show_default=True, type=float,
              help="Only save edges with EV >= this value")
@click.option("--no-odds", is_flag=True, default=False,
              help="Skip The Odds API fetch (use if ODDS_API_KEY not set)")
def run_weekly(season, week, n_draws, kelly_fraction, min_ev, no_odds) -> None:
    """Full weekly pipeline: data refresh → odds → simulate all games → save edges.

    Runs everything in one shot. Suitable for cron:

      0 10 * * 3  ironclad run   # every Wednesday at 10am
    """
    from ironclad.workflow.auto_run import AutoRunWorkflow

    try:
        result = AutoRunWorkflow(
            n_draws=n_draws,
            kelly_fraction=kelly_fraction,
            min_ev=min_ev,
            skip_odds=no_odds,
        ).run(season=season, week=week)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    s, w = result["season"], result["week"]
    click.echo(f"\nironclad auto-run — {s} Week {w}")
    click.echo("─" * 40)
    fb = result["weekly"].get("features_built", "?")
    click.echo(f"Data refresh:     ✓ {fb} game feature sets built")
    if not no_odds:
        click.echo(f"Odds/props fetch: ✓ {result['odds_rows']} rows ingested")
    gt, gs = result["games_total"], result["games_simulated"]
    skipped_count = gt - gs
    click.echo(
        f"Simulations:      ✓ {gs}/{gt} games"
        + (f" ({skipped_count} skipped)" if skipped_count else "")
    )
    click.echo(f"Edges saved:      ✓ {result['edges_saved']} → gold.betting_edges")
    if result["skipped"]:
        click.echo("Skipped:")
        for item in result["skipped"]:
            click.echo(f"  - {item}")
    click.echo("\nRun `ironclad serve` to view edges in the dashboard.")

    from ironclad.notifications.discord import post_run_summary
    from ironclad.store.connection import get_connection
    post_run_summary(result, conn=get_connection())


# ── edges ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--game-id", required=True, help="Canonical game ID (e.g. 2024_18_KC_LAC)")
@click.option("--props-file", default=None, type=click.Path(exists=True, dir_okay=False),
              help="CSV of prop lines (player_id,stat_type,line,over_odds,under_odds). "
                   "If omitted, props are loaded from bronze.player_props (run `ironclad odds` first).")
@click.option("--n-draws", default=DEFAULT_N_DRAWS, show_default=True, type=int)
@click.option("--kelly-fraction", default=0.25, show_default=True, type=float,
              help="Fractional Kelly multiplier (0.25 = quarter Kelly)")
@click.option("--min-ev", default=0.0, show_default=True, type=float,
              help="Filter: only show edges with EV >= this value (per $1 stake)")
@click.option("--backfill-if-missing", is_flag=True, default=False)
@click.option("--save", is_flag=True, default=False,
              help="Persist edges to gold.betting_edges for later settling")
@click.option("--use-drive-sim", is_flag=True, default=False,
              help="Use drive-level Markov chain simulation instead of score-distribution GameDraw")
def edges(game_id, props_file, n_draws, kelly_fraction, min_ev, backfill_if_missing, save, use_drive_sim) -> None:
    """Compute +EV player props from a Monte Carlo simulation against market lines."""
    from ironclad.betting.props import PropAnalyzer, load_prop_lines, load_prop_lines_from_db
    from ironclad.workflow.matchup import MatchupWorkflow

    if props_file:
        prop_lines = load_prop_lines(Path(props_file))
        if not prop_lines:
            click.echo("ERROR: No prop lines parsed from CSV", err=True)
            sys.exit(1)
        click.echo(f"Loaded {len(prop_lines)} prop lines from {props_file}")
    else:
        from ironclad.store.connection import get_connection as _gc
        from ironclad.store.schema import create_all_tables as _cat
        _conn = _gc()
        _cat(_conn)
        prop_lines = load_prop_lines_from_db(_conn, game_id)
        if not prop_lines:
            click.echo(
                f"No player props found in DB for {game_id}.\n"
                "Run `ironclad odds` to fetch from The Odds API, or provide --props-file.",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Loaded {len(prop_lines)} prop lines from bronze.player_props")

    click.echo(f"Running simulation for {game_id} (n_draws={n_draws})...")
    result, _, _, _ = MatchupWorkflow(n_draws=n_draws, use_drive_sim=use_drive_sim).simulate(
        game_id, backfill_if_missing=backfill_if_missing,
    )

    analyzer = PropAnalyzer(kelly_fraction=kelly_fraction)
    edges_df = analyzer.analyze(result, prop_lines)
    if edges_df.empty:
        click.echo("No edges produced (no matching player_ids in simulation).")
        return

    if min_ev > 0:
        edges_df = edges_df[edges_df["ev"] >= min_ev].reset_index(drop=True)

    if edges_df.empty:
        click.echo(f"No edges with EV >= {min_ev}")
        return

    click.echo(f"\nTop edges (kelly_fraction={kelly_fraction}):\n")
    cols = ["player_name", "position", "stat_type", "market_line", "side", "odds",
            "model_prob", "market_prob", "edge", "ev", "kelly"]
    show = edges_df[[c for c in cols if c in edges_df.columns]]
    click.echo(show.to_string(index=False))

    if save:
        from ironclad.eval.performance_tracker import save_edges
        from ironclad.store.connection import get_connection
        from ironclad.store.schema import create_all_tables
        conn = get_connection()
        create_all_tables(conn)
        ids = save_edges(conn, game_id, n_draws, edges_df)
        click.echo(f"\nSaved {len(ids)} edge(s) to gold.betting_edges.")
        click.echo(f"Edge IDs: {', '.join(ids)}")


# ── parlay ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--game-id", required=True, help="Canonical game ID (e.g. 2024_18_KC_LAC)")
@click.option("--legs-file", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV of parlay legs: player_id,stat_type,direction,line[,market_odds]")
@click.option("--n-draws", default=DEFAULT_N_DRAWS, show_default=True, type=int)
@click.option("--backfill-if-missing", is_flag=True, default=False)
def parlay(game_id, legs_file, n_draws, backfill_if_missing) -> None:
    """Price a same-game parlay using the Monte Carlo joint distribution."""
    from ironclad.betting.parlays import load_parlay_legs, parlay_probability, market_independence_prob
    from ironclad.betting.ev import prob_to_american, american_to_prob
    from ironclad.betting.props import PropAnalyzer
    from ironclad.workflow.matchup import MatchupWorkflow

    legs = load_parlay_legs(Path(legs_file))
    if not legs:
        click.echo("ERROR: No legs parsed from CSV", err=True)
        sys.exit(1)

    click.echo(f"Running simulation for {game_id} (n_draws={n_draws})...")
    result, _, _, _ = MatchupWorkflow(n_draws=n_draws).simulate(
        game_id, backfill_if_missing=backfill_if_missing,
    )

    # Per-leg individual model probabilities
    from ironclad.betting.props import PropLine
    from ironclad.simulation.results import SimulationResult
    player_df = result._player_df

    click.echo(f"\nParlay: {len(legs)}-leg SGP for {game_id}  (n={n_draws} draws)\n")
    click.echo(f"  {'Player':<24} {'stat':<12} {'dir':<6} {'line':>6}  {'model':>7}  {'market':>7}  {'edge':>7}")
    click.echo("  " + "-" * 72)

    for leg in legs:
        player_rows = player_df[player_df["player_id"] == leg.player_id] if not player_df.empty else player_df
        player_name = player_rows.iloc[0]["player_name"] if not player_rows.empty else leg.player_id

        # Individual model probability for this leg
        from ironclad.betting.parlays import parlay_probability as single_leg_prob
        leg_prob = single_leg_prob(result, [leg])

        market_str = f"{american_to_prob(leg.market_odds):.1%}" if leg.market_odds is not None else "  n/a "
        edge_str = (
            f"{leg_prob - american_to_prob(leg.market_odds):+.1%}"
            if leg.market_odds is not None else "  n/a "
        )
        line_str = f"{leg.line:.1f}" if leg.line is not None else "  -"
        click.echo(
            f"  {player_name:<24} {leg.stat_type:<12} {leg.direction:<6} {line_str:>6}  "
            f"{leg_prob:>6.1%}  {market_str:>7}  {edge_str:>7}"
        )

    joint_prob = parlay_probability(result, legs)
    fair_odds = prob_to_american(joint_prob) if 0 < joint_prob < 1 else None
    fair_str = f"+{fair_odds}" if fair_odds and fair_odds > 0 else str(fair_odds)

    click.echo()
    click.echo(f"  Joint probability (model):     {joint_prob:.2%}   fair odds: {fair_str}")

    mkt_prob = market_independence_prob(legs)
    if mkt_prob is not None and 0 < mkt_prob < 1:
        mkt_odds = prob_to_american(mkt_prob)
        mkt_str = f"+{mkt_odds}" if mkt_odds > 0 else str(mkt_odds)
        parlay_edge = joint_prob - mkt_prob
        edge_direction = "model UNDERprices" if parlay_edge < 0 else "market UNDERprices"
        click.echo(f"  Market implied (independent):  {mkt_prob:.2%}   market odds: {mkt_str}")
        click.echo(f"  Parlay edge:                  {parlay_edge:+.2%}  ({edge_direction} this parlay)")
    else:
        click.echo("  (Provide market_odds for all legs to see market comparison)")


# ── status ─────────────────────────────────────────────────────────────────────

# ── settle ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--edge-id", required=True,
              help="8-char edge ID printed by 'ironclad edges --save'")
@click.option("--result", required=True,
              type=click.Choice(["win", "loss", "push"]),
              help="Outcome of the bet")
@click.option("--units", default=1.0, show_default=True, type=float,
              help="Units wagered (e.g. 1.0 = one Kelly unit)")
def settle(edge_id, result, units) -> None:
    """Record the settled outcome of a placed bet."""
    from ironclad.eval.performance_tracker import settle_bet
    from ironclad.store.connection import get_connection
    from ironclad.store.schema import create_all_tables
    conn = get_connection()
    create_all_tables(conn)
    try:
        outcome = settle_bet(conn, edge_id, result, units)
    except ValueError as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)
    sign = "+" if outcome["profit_units"] >= 0 else ""
    click.echo(
        f"Settled: {outcome['result']}  |  "
        f"{outcome['stat_type']} {outcome['side']}  |  "
        f"profit: {sign}{outcome['profit_units']:.2f} units"
    )


# ── results ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--game-id", default=None, help="Filter to a specific game")
def results(game_id) -> None:
    """Show P&L summary from settled bets."""
    from ironclad.eval.performance_tracker import load_results, pnl_summary
    from ironclad.store.connection import get_connection
    from ironclad.store.schema import create_all_tables
    conn = get_connection()
    create_all_tables(conn)
    df = load_results(conn, game_id=game_id)
    if df.empty:
        click.echo("No settled bets found. Use 'ironclad edges --save' then 'ironclad settle'.")
        return
    summary = pnl_summary(df)
    click.echo("\nBetting Results")
    click.echo(f"  Total bets:    {summary['total_bets']}")
    click.echo(f"  Win / Loss:    {summary['wins']} / {summary['losses']}")
    click.echo(f"  Win rate:      {summary['win_rate']:.1%}")
    click.echo(f"  Total profit:  {summary['total_profit_units']:+.2f} units")
    click.echo(f"  Total wagered: {summary['total_wagered_units']:.2f} units")
    click.echo(f"  ROI:           {summary['roi']:+.1%}")
    click.echo()
    cols = ["result_id", "stat_type", "side", "market_line", "odds",
            "model_prob", "ev", "units_wagered", "result", "profit_units"]
    show = df[[c for c in cols if c in df.columns]]
    click.echo(show.to_string(index=False))


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


# ── validate ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--season", required=True, type=int)
@click.option("--strict", is_flag=True, default=False,
              help="Exit 1 if any check fails")
def validate(season: int, strict: bool) -> None:
    """Run data quality checks for a season."""
    from ironclad.store.connection import get_connection
    from ironclad.store.schema import create_all_tables
    from ironclad.store.data_quality import run_all_checks

    conn = get_connection()
    create_all_tables(conn)

    click.echo(f"\nData quality checks — season {season}\n")

    try:
        results = run_all_checks(conn, season)
    except Exception as exc:
        click.echo(click.style(f"ERROR running checks: {exc}", fg="red"), err=True)
        sys.exit(1)

    def _s(passed: bool) -> str:
        return click.style("✓", fg="green") if passed else click.style("✗", fg="red")

    # Row counts
    rc = results["row_counts"]
    click.echo(f"  {_s(rc['passed'])}  Row counts")
    for table, count in rc["counts"].items():
        flag = click.style("  ← 0 rows", fg="yellow") if count == 0 else ""
        click.echo(f"       {table:<42} {count:>8,}{flag}")

    # Null rates
    nr = results["null_rates"]
    click.echo(f"\n  {_s(nr['passed'])}  Null rates (critical columns)")
    for key, pct in nr["null_rates"].items():
        color = "red" if pct >= 1.0 else ("yellow" if pct >= 0.5 else "green")
        click.echo(f"       {key:<55}  null: {click.style(f'{pct:.1%}', fg=color)}")

    # EPA range
    er = results["epa_range"]
    click.echo(f"\n  {_s(er['passed'])}  EPA plausibility [-1.5, 1.5]")
    for table, count in er["out_of_range"].items():
        click.echo(f"       {table:<42} {count:>6} out-of-range rows")

    # Completeness
    score = results["completeness_score"]
    cp = results["completeness_passed"]
    click.echo(
        f"\n  {_s(cp)}  Gold feature completeness: "
        f"{click.style(f'{score:.1%}', fg='green' if cp else 'red')}  (target ≥ 70%)"
    )

    # Summary
    overall = results["passed"]
    summary = (
        click.style("ALL CHECKS PASSED", fg="green")
        if overall
        else click.style("SOME CHECKS FAILED", fg="red")
    )
    click.echo(f"\n  Summary: {summary}\n")

    if strict and not overall:
        sys.exit(1)


# ── serve ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--port", default=8501, show_default=True, type=int,
              help="Port to run the Streamlit dashboard on")
@click.option("--host", default="localhost", show_default=True,
              help="Host/address to bind to")
def serve(port, host) -> None:
    """Launch the Streamlit betting intelligence dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent.parent / "ironclad" / "dashboard" / "app.py"
    if not app_path.exists():
        click.echo(f"ERROR: Dashboard not found at {app_path}", err=True)
        sys.exit(1)

    click.echo(f"Starting ironclad dashboard at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop.")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", str(app_path),
                "--server.port", str(port),
                "--server.address", host,
                "--server.headless", "true",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")
    except FileNotFoundError:
        click.echo(
            "ERROR: streamlit not installed. Run: pip install 'ironclad-v2[dashboard]'",
            err=True,
        )
        sys.exit(1)


# ── api ───────────────────────────────────────────────────────────────────────

@cli.command("api")
@click.option("--port", default=8000, show_default=True, type=int,
              help="Port to bind the API server to")
@click.option("--host", default="localhost", show_default=True,
              help="Host/address to bind to")
def api_serve(port, host) -> None:
    """Launch the FastAPI REST API server.

    Endpoints: GET /api/v1/games, POST /api/v1/simulate,
    POST /api/v1/edges, GET /api/v1/results.
    Interactive docs at http://<host>:<port>/docs
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "ERROR: uvicorn not installed. Run: pip install 'ironclad-v2[api]'",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Starting ironclad API at http://{host}:{port}")
    click.echo(f"Docs: http://{host}:{port}/docs")
    click.echo("Press Ctrl+C to stop.")
    uvicorn.run("ironclad.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
