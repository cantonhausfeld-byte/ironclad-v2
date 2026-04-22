# ironclad-v2 — developer guide for Claude Code

## Project overview

ironclad-v2 is an NFL matchup forecasting system. It ingests free public data (nfl_data_py, Open-Meteo, The Odds API), builds a 3-layer feature store in DuckDB, runs a Monte Carlo simulation (5,000 draws), and emits a pregame matchup report (Markdown + HTML).

## Setup

```bash
pip install -e ".[dev]"
python -c "import ironclad"   # verify import
```

Requires `.env` with `ODDS_API_KEY=<key>` (free tier, optional — stubs work without it).

## Running tests

```bash
pytest tests/unit/          # fast, no network
pytest tests/integration/   # needs local DB data
```

All unit tests must pass with no network calls. Integration tests require a backfilled DuckDB.

## CLI commands

```bash
ironclad backfill --season 2023        # ingest + silver + gold features
ironclad train --train-seasons 2016-2022 --val-seasons 2023
ironclad report --game-id 2023_18_KC_LAC --backfill-if-missing
ironclad report --home BAL --away KC --week 1 --season 2024 --format both
ironclad status
ironclad evaluate --val-seasons 2023
```

## Architecture: 3-layer data store

```
bronze.*   raw ingest (append-only, _ingest_ts)
silver.*   cleaned/joined (one row per game/team/player)
gold.*     feature store (cutoff_ts enforced, model-ready)
```

All tables live in `data/ironclad.ddb`. `FeatureSnapshot(cutoff_ts, conn)` is the gatekeeper — every feature query filters `_ingest_ts <= cutoff_ts` so historical runs are reproducible.

## Critical invariants

1. **Bronze is append-only** — never DELETE or UPDATE bronze rows; use `_ingest_ts` for time travel.
2. **Knowledge cutoff** — `cutoff_ts = kickoff - 30min`. All features must use only data knowable before that timestamp. `FeatureSnapshot` enforces this.
3. **Bernoulli availability gate** — a Questionable player (availability=0.60) plays at 100% capacity in 60% of draws and zero in 40%. Do not use scalar dampening.
4. **Reconcile per draw** — player yards are scaled to team totals within each individual draw, not on expectations.
5. **Stub models ship first** — `GameOutcomeModel`, `ScoreEnvironmentModel`, `PlayerUsageModel`, `PlayerEfficiencyModel` all have stub implementations that fall back to odds-implied probs and league-average priors. Real XGBoost training is optional.

## Key files

| File | Purpose |
|---|---|
| `ironclad/store/schema.py` | All DDL — edit here if adding columns |
| `ironclad/store/silver.py` | Bronze → silver transforms |
| `ironclad/features/snapshot.py` | Knowledge cutoff enforcement |
| `ironclad/features/team_features.py` | Gold team feature builder |
| `ironclad/features/player_features.py` | Gold player feature builder |
| `ironclad/simulation/engine.py` | Monte Carlo orchestrator |
| `ironclad/simulation/reconciler.py` | Per-draw player→team reconciliation |
| `ironclad/report/builder.py` | Report context assembly |
| `ironclad/report/templates/` | Jinja2 templates (.md.j2, .html.j2) |
| `ironclad/models/trainer.py` | XGBoost training pipeline |
| `ironclad/models/registry.py` | Model artifact save/load |
| `cli/main.py` | Click CLI entry point |

## Adding a new feature column

1. Add the column to `schema.py` DDL.
2. Compute it in the appropriate silver transform (`silver.py`) or feature builder (`team_features.py`, `player_features.py`).
3. Add it to the model's feature list (`TEAM_FEATURES` in `game_outcome.py`, etc.) if it should be used for training.
4. Add a unit test in `tests/unit/`.

## Model training

Training reads from `gold.team_game_features` and `gold.player_game_features`. Targets are the `target_*` columns backfilled by `TargetBackfiller` after games complete. Walk-forward validation uses held-out seasons. Trained models are serialized to `data/models/` via pickle.

## Report formats

`ironclad report --format markdown` (default) — writes `.md`.  
`ironclad report --format both` — writes `.md` + `.html` + `_players.csv` + `_teams.csv`.

## Common pitfalls

- **Position is UNK in player_game_stats**: Happens when `bronze.rosters` is empty (no backfill). Run `ironclad backfill --season <year>` first.
- **Team features empty**: Gold features require silver to exist. Run `ironclad backfill` before `ironclad report`.
- **Schema mismatch**: If you add columns to `schema.py` but have an existing `.ddb` file, the `CREATE TABLE IF NOT EXISTS` won't add the new columns. Delete `data/ironclad.ddb` and re-backfill.
- **ModelRegistry FileNotFoundError**: No trained model found — the system falls back to stubs automatically.
