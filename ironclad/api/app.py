"""FastAPI REST API for ironclad-v2."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from ironclad.betting.props import PropAnalyzer, load_prop_lines_from_db
from ironclad.eval.performance_tracker import load_results, pnl_summary
from ironclad.store.connection import get_connection
from ironclad.workflow.matchup import MatchupWorkflow

logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class GameInfo(BaseModel):
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    gameday: str
    season_type: str | None = None


class SimulateRequest(BaseModel):
    game_id: str
    n_draws: int = 500

    @field_validator("n_draws")
    @classmethod
    def n_draws_valid(cls, v: int) -> int:
        if not (100 <= v <= 50_000):
            raise ValueError("n_draws must be between 100 and 50,000")
        return v


class SimulationResponse(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    n_draws: int
    home_win_prob: float
    away_win_prob: float
    score_summary: dict[str, Any]
    team_stats: list[dict[str, Any]]
    player_stats: list[dict[str, Any]]


class EdgeItem(BaseModel):
    player_name: str | None
    team: str | None
    position: str | None
    stat_type: str
    market_line: float | None
    side: str
    odds: int | None
    model_prob: float | None
    market_prob: float | None
    edge: float | None
    ev: float | None
    kelly: float | None


class EdgesRequest(BaseModel):
    game_id: str
    kelly_fraction: float = 0.25
    min_ev: float = 0.0
    n_draws: int = 500

    @field_validator("n_draws")
    @classmethod
    def n_draws_valid(cls, v: int) -> int:
        if not (100 <= v <= 50_000):
            raise ValueError("n_draws must be between 100 and 50,000")
        return v

    @field_validator("kelly_fraction")
    @classmethod
    def kelly_valid(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("kelly_fraction must be in (0, 1]")
        return v

    @field_validator("min_ev")
    @classmethod
    def min_ev_valid(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("min_ev must be >= 0.0")
        return v


class EdgesResponse(BaseModel):
    game_id: str
    n_draws: int
    edges: list[EdgeItem]


class ResultsResponse(BaseModel):
    summary: dict[str, Any]
    bets: list[dict[str, Any]]


class StoredEdgeItem(BaseModel):
    game_id: str
    season: int | None = None
    week: int | None = None
    player_name: str | None = None
    stat_type: str
    side: str
    market_line: float | None = None
    ev: float | None = None
    kelly: float | None = None


class StatusResponse(BaseModel):
    tables: dict[str, int]
    models: dict[str, list[str]]
    scheduler: dict[str, Any]


def _ro_conn():
    """Read-only connection for API endpoints.

    DuckDB allows many concurrent readers as long as no writer holds an
    exclusive lock. Using read_only=True here lets batch jobs (backfill,
    weekly, train) open a read-write connection in a separate process without
    the API blocking them.
    """
    return get_connection(read_only=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import duckdb

    from ironclad.config import DB_PATH
    from ironclad.store.schema import create_all_tables
    # Bootstrap schemas using a short-lived read-write connection that is
    # closed before we start serving. Leaving it open would block read-only
    # request-thread connections (DuckDB refuses mixed-mode opens on the
    # same file) and prevent batch writers from acquiring the writer lock.
    rw_conn = duckdb.connect(str(DB_PATH), read_only=False)
    rw_conn.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    rw_conn.execute("CREATE SCHEMA IF NOT EXISTS silver")
    rw_conn.execute("CREATE SCHEMA IF NOT EXISTS gold")
    create_all_tables(rw_conn)
    rw_conn.close()
    logger.info("ironclad API started")
    yield
    logger.info("ironclad API stopped")


app = FastAPI(
    title="ironclad API",
    description="NFL matchup simulation and betting edge REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── GET /api/v1/games ─────────────────────────────────────────────────────────

@app.get("/api/v1/games", response_model=list[GameInfo])
def get_games(
    season: int = Query(..., description="NFL season year"),
    week: int = Query(..., description="Week number"),
):
    """List games for a given season and week."""
    conn = _ro_conn()
    try:
        df = conn.execute(
            """
            SELECT game_id, season, week, home_team, away_team,
                   CAST(gameday AS VARCHAR) AS gameday, season_type
            FROM silver.games
            WHERE season = ? AND week = ?
            ORDER BY gameday
            """,
            [season, week],
        ).df()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return df.to_dict("records")


# ── POST /api/v1/simulate ─────────────────────────────────────────────────────

@app.post("/api/v1/simulate", response_model=SimulationResponse)
def simulate(req: SimulateRequest):
    """Run Monte Carlo simulation for a game."""
    try:
        result, _, _, _ = MatchupWorkflow(n_draws=req.n_draws).simulate(req.game_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Simulation failed for %s", req.game_id)
        raise HTTPException(status_code=500, detail=str(exc))

    home_prob, away_prob = result.win_probability()
    team_df = result.team_summary()
    player_df = result.player_summary()

    return SimulationResponse(
        game_id=req.game_id,
        home_team=result.home_team,
        away_team=result.away_team,
        n_draws=req.n_draws,
        home_win_prob=home_prob,
        away_win_prob=away_prob,
        score_summary=result.score_summary(),
        team_stats=team_df.to_dict("records") if not team_df.empty else [],
        player_stats=player_df.to_dict("records") if not player_df.empty else [],
    )


# ── POST /api/v1/edges ────────────────────────────────────────────────────────

@app.post("/api/v1/edges", response_model=EdgesResponse)
def get_edges(req: EdgesRequest):
    """Compute +EV betting edges for a game using Monte Carlo simulation."""
    conn = _ro_conn()

    prop_lines = load_prop_lines_from_db(conn, req.game_id)
    if not prop_lines:
        raise HTTPException(
            status_code=404,
            detail=f"No prop lines found for {req.game_id}. Run `ironclad odds` first.",
        )

    try:
        result, _, _, _ = MatchupWorkflow(n_draws=req.n_draws).simulate(req.game_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Simulation failed for %s", req.game_id)
        raise HTTPException(status_code=500, detail=str(exc))

    edges_df = PropAnalyzer(kelly_fraction=req.kelly_fraction).analyze(result, prop_lines)
    if req.min_ev > 0 and not edges_df.empty:
        edges_df = edges_df[edges_df["ev"] >= req.min_ev].reset_index(drop=True)

    edges = []
    if not edges_df.empty:
        for _, row in edges_df.iterrows():
            edges.append(EdgeItem(
                player_name=row.get("player_name"),
                team=row.get("team"),
                position=row.get("position"),
                stat_type=str(row["stat_type"]),
                market_line=row.get("market_line"),
                side=str(row["side"]),
                odds=int(row["odds"]) if row.get("odds") is not None else None,
                model_prob=row.get("model_prob"),
                market_prob=row.get("market_prob"),
                edge=row.get("edge"),
                ev=row.get("ev"),
                kelly=row.get("kelly"),
            ))

    return EdgesResponse(game_id=req.game_id, n_draws=req.n_draws, edges=edges)


# ── GET /api/v1/results ───────────────────────────────────────────────────────

@app.get("/api/v1/results", response_model=ResultsResponse)
def get_results(
    game_id: str | None = Query(None, description="Filter to a specific game"),
    season: int | None = Query(None, description="Filter to a specific season (not yet indexed)"),
):
    """Return P&L summary and settled bet history."""
    conn = _ro_conn()
    try:
        df = load_results(conn, game_id=game_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    summary = pnl_summary(df) if not df.empty else {
        "total_bets": 0, "wins": 0, "losses": 0,
        "win_rate": 0.0, "total_profit_units": 0.0,
        "total_wagered_units": 0.0, "roi": 0.0,
    }
    bets = df.to_dict("records") if not df.empty else []

    return ResultsResponse(summary=summary, bets=bets)


# ── GET /api/v1/edges/stored ──────────────────────────────────────────────────

@app.get("/api/v1/edges/stored", response_model=list[StoredEdgeItem])
def get_stored_edges(
    game_id: str | None = Query(None),
    season: int | None = Query(None),
    week: int | None = Query(None),
    min_ev: float = Query(0.0),
    limit: int = Query(25, ge=1, le=500),
):
    """Read stored edges from gold.betting_edges (no simulation)."""
    conn = _ro_conn()
    where = ["e.ev >= ?"]
    params: list = [min_ev]
    if game_id:
        where.append("e.game_id = ?")
        params.append(game_id)
    if season is not None:
        where.append("g.season = ?")
        params.append(season)
    if week is not None:
        where.append("g.week = ?")
        params.append(week)

    sql = f"""
        SELECT e.game_id, g.season, g.week,
               e.player_name, e.stat_type, e.side, e.market_line, e.ev, e.kelly
        FROM gold.betting_edges e
        LEFT JOIN silver.games g ON g.game_id = e.game_id
        WHERE {' AND '.join(where)}
        ORDER BY e.ev DESC
        LIMIT ?
    """
    params.append(limit)
    try:
        df = conn.execute(sql, params).df()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return df.to_dict("records") if not df.empty else []


# ── GET /api/v1/status ────────────────────────────────────────────────────────

_STATUS_TABLES = [
    "bronze.schedules", "bronze.play_by_play", "bronze.rosters",
    "silver.games", "silver.team_game_stats", "silver.player_game_stats",
    "gold.team_game_features", "gold.player_game_features", "gold.betting_edges",
]


@app.get("/api/v1/status", response_model=StatusResponse)
def get_status():
    """Data-store, model-registry, and scheduler snapshot."""
    import json
    from pathlib import Path

    from ironclad.config import DATA_DIR
    from ironclad.models.registry import ModelRegistry

    conn = _ro_conn()
    tables: dict[str, int] = {}
    for t in _STATUS_TABLES:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            n = 0
        tables[t] = int(n)

    registry = ModelRegistry()
    models: dict[str, list[str]] = {}
    for name in ["game_outcome", "score_env", "player_usage", "player_efficiency"]:
        try:
            models[name] = registry.list_versions(name)
        except Exception:
            models[name] = []

    scheduler: dict[str, Any] = {}
    state_path = Path(DATA_DIR) / "scheduler_state.json"
    if state_path.exists():
        try:
            scheduler = json.loads(state_path.read_text())
        except Exception:
            scheduler = {}

    return StatusResponse(tables=tables, models=models, scheduler=scheduler)


# ── GET /api/v1/health ────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health():
    """Liveness check."""
    return {"status": "ok"}
