# ironclad-v2

NFL matchup forecasting from free public data. For each game, ironclad builds
point-in-time team and player features, runs a 5,000-draw Monte Carlo
simulation, and produces a pregame report plus player-prop and game-line edges
priced against the market.

- **Data:** nflverse (play-by-play, rosters, injuries, depth charts, NGS, FTN,
  snap counts), Open-Meteo / NASA POWER weather, and The Odds API.
- **Store:** DuckDB with bronze (raw, append-only), silver (cleaned) and gold
  (model-ready features, knowledge-cutoff enforced) layers.
- **Models:** XGBoost / LightGBM game outcome, score environment, player
  usage and efficiency. Every model has a stub fallback, so reports work
  before any training.
- **Output:** Markdown/HTML reports, CSV exports, Discord posts, a FastAPI
  server and a Streamlit dashboard.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for status and what's left before v1.0,
and [`CLAUDE.md`](CLAUDE.md) for architecture and invariants.

## Setup

Requires Python 3.11+.

```bash
pip install -e ".[dev]"          # add ,dashboard for the Streamlit UI
cp .env.example .env              # then fill in keys (see below)
```

| Variable | Required | Purpose |
|---|---|---|
| `ODDS_API_KEY` | for live odds/edges | [The Odds API](https://the-odds-api.com). The free tier is 500 requests/month. |
| `DISCORD_WEBHOOK_URL` | no | Posts reports, edges and scheduler alerts |
| `IRONCLAD_DATA_DIR` / `IRONCLAD_DB_PATH` | no | Defaults to `./data/ironclad.ddb` |
| `IRONCLAD_LOG_FORMAT` | no | `text` (default) or `json` |

## First run

```bash
# 1. Load history + the current season (takes a while the first time)
ironclad backfill -s 2016 -s 2017 -s 2018 -s 2019 -s 2020 -s 2021 \
                  -s 2022 -s 2023 -s 2024 -s 2025 -s 2026

# 2. Train (optional — stubs work without it)
ironclad train --train-seasons 2016-2024 --val-seasons 2025

# 3. One matchup report
ironclad report --home BAL --away KC --week 4 --season 2026 --format both

# 4. Health check
ironclad status
```

## Weekly operation

```bash
ironclad run                 # refresh data → odds → simulate every game → save edges
ironclad top-edges           # best edges from the latest run
ironclad settle-week --season 2026 --week 4   # grade last week's edges
ironclad results             # P&L summary
```

### Unattended (GitHub Actions)

`.github/workflows/weekly.yml` runs the pipeline on GitHub's servers:

- **Tuesday:** post-game backfill.
- **Wednesday:** pre-game run. Every game gets a report, posted to Discord.
- **Saturday:** injury update. Remaining games are re-simulated and re-reported with Friday's final designations.
- **February:** retrain.

The DuckDB store persists between runs in the Actions cache, and the first run (or one after a cache eviction)
rebuilds it in about 1.5–2 hours. Reports are also uploaded as run artifacts.

Setup: in the GitHub repo, go to **Settings → Secrets and variables → Actions** and add `ODDS_API_KEY` and
`DISCORD_WEBHOOK_URL`. Both are optional: without the key you still get reports but no odds or prop edges, and
without the webhook nothing is posted. You can run it by hand from the Actions tab (**Weekly pipeline → Run
workflow**) and choose `scheduled`, `pre-game`, `post-game` or `rebuild`.

To self-host instead, run `ironclad schedule` under systemd (`deploy/ironclad-scheduler.service`) or cron
(`deploy/ironclad.cron`).

## Evaluating the models

```bash
ironclad backtest --test-seasons 2019-2025      # walk-forward, stores predictions
ironclad dashboard                              # metrics for the latest backtest run
ironclad profitability                          # spread/total ROI + CLV vs the market
ironclad prop-backtest                          # prop distribution coverage
ironclad evaluate --val-seasons 2025 --breakdown-by-week
```

## Development

```bash
pytest tests/unit/           # fast, no network — must pass
pytest tests/integration/    # needs a backfilled DB
ruff check ironclad/ cli/ tests/
```

CI runs ruff and the unit tests on every push. Before changing features or
models, read the invariants in [`CLAUDE.md`](CLAUDE.md), especially the
knowledge cutoff and the per-draw reconciliation.
