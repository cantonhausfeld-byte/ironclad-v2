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
    from ironclad.store.schema import create_all_tables
    # Bootstrap schemas using a brief read-write connection, then close it so
    # the batch process can acquire the writer lock unobstructed.
    rw_conn = get_connection(read_only=False)
    create_all_tables(rw_conn)
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
