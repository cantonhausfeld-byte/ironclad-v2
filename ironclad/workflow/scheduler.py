"""Weekly automation scheduler for ironclad.

Runs two distinct jobs on a weekly cycle:
  - PRE-GAME (Wednesday 10am): odds refresh, simulate, save edges
  - POST-GAME (Tuesday 6am):   backfill new PBP/stats, rebuild targets, validate

The scheduler uses a persistent state file to avoid running the same job twice
in the same week. It is designed to be long-running (started once via cron or
systemd) but can also be run in one-shot mode for CI / manual invocations.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).parent.parent.parent / "data" / "scheduler_state.json"

# Day-of-week constants (0 = Monday … 6 = Sunday)
_WEDNESDAY = 2
_TUESDAY = 1

# Hour (24h, local) at which each job fires
_PRE_GAME_HOUR = 10   # Wednesday 10 am
_POST_GAME_HOUR = 6   # Tuesday 6 am

# How often the scheduler wakes to check (seconds)
_POLL_INTERVAL_SECONDS = 60 * 15  # 15 minutes


# ── State persistence ─────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        if _STATE_FILE.exists():
            return json.loads(_STATE_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def _iso_week_key(dt: date) -> str:
    """Return 'YYYY-Www' for the NFL week containing *dt*."""
    # NFL week starts Thursday; map to ISO week for dedup purposes
    return dt.strftime("%Y-W%V")


# ── Job runners ────────────────────────────────────────────────────────────────

def run_pre_game(season: int | None = None, week: int | None = None,
                 n_draws: int = 5000, kelly_fraction: float = 0.25,
                 min_ev: float = 0.0) -> dict:
    """Wednesday refresh: fetch odds, simulate upcoming games, save edges."""
    from ironclad.workflow.auto_run import AutoRunWorkflow
    logger.info("=== PRE-GAME RUN: season=%s week=%s ===", season, week)
    result = AutoRunWorkflow(
        n_draws=n_draws,
        kelly_fraction=kelly_fraction,
        min_ev=min_ev,
    ).run(season=season, week=week)
    logger.info(
        "Pre-game complete: %d games simulated, %d edges saved",
        result["games_simulated"], result["edges_saved"],
    )
    return result


def run_post_game(season: int, n_draws: int = 5000) -> dict:
    """Tuesday backfill: ingest new PBP/stats, rebuild targets, validate."""
    from ironclad.store.connection import get_connection
    from ironclad.store.data_quality import log_quality_report
    from ironclad.store.schema import create_all_tables
    from ironclad.workflow.backfill import BackfillWorkflow

    logger.info("=== POST-GAME BACKFILL: season=%d ===", season)
    result = BackfillWorkflow().run([season])
    conn = get_connection()
    create_all_tables(conn)
    log_quality_report(conn, season)
    logger.info("Post-game backfill complete: %s", result)
    return result


# ── Detection helpers ──────────────────────────────────────────────────────────

def detect_nfl_season(today: date | None = None) -> int:
    """Return the current NFL season year.

    NFL seasons start in September and end in February.
    January–February belongs to the previous calendar year's season.
    """
    d = today or date.today()
    return d.year if d.month >= 3 else d.year - 1


def detect_upcoming_week(conn=None) -> tuple[int, int] | None:
    """Return (season, week) of the next upcoming game, or None if off-season."""
    try:
        from ironclad.store.connection import get_connection
        from ironclad.store.schema import create_all_tables
        c = conn or get_connection()
        create_all_tables(c)
        row = c.execute("""
            SELECT season, week FROM bronze.schedules
            WHERE gameday >= CURRENT_DATE AND season_type = 'REG'
            ORDER BY gameday ASC LIMIT 1
        """).fetchone()
        if row:
            return int(row[0]), int(row[1])
    except Exception as exc:
        logger.warning("Could not detect upcoming week: %s", exc)
    return None


def detect_completed_week(conn=None) -> tuple[int, int] | None:
    """Return (season, week) of the most recently completed week, or None."""
    try:
        from ironclad.store.connection import get_connection
        from ironclad.store.schema import create_all_tables
        c = conn or get_connection()
        create_all_tables(c)
        row = c.execute("""
            SELECT season, week FROM bronze.schedules
            WHERE gameday < CURRENT_DATE AND season_type = 'REG'
            ORDER BY gameday DESC LIMIT 1
        """).fetchone()
        if row:
            return int(row[0]), int(row[1])
    except Exception as exc:
        logger.warning("Could not detect completed week: %s", exc)
    return None


# ── Main scheduler loop ────────────────────────────────────────────────────────

class WeeklyScheduler:
    """Long-running scheduler.  Call ``run()`` to start; it blocks forever."""

    def __init__(
        self,
        n_draws: int = 5000,
        kelly_fraction: float = 0.25,
        min_ev: float = 0.0,
        poll_interval: int = _POLL_INTERVAL_SECONDS,
        dry_run: bool = False,
    ) -> None:
        self.n_draws = n_draws
        self.kelly_fraction = kelly_fraction
        self.min_ev = min_ev
        self.poll_interval = poll_interval
        self.dry_run = dry_run

    def run(self) -> None:
        logger.info("ironclad scheduler started (poll=%ds)", self.poll_interval)
        while True:
            try:
                self._tick()
            except Exception as exc:
                logger.error("Scheduler tick failed: %s", exc, exc_info=True)
            time.sleep(self.poll_interval)

    def run_once(self) -> dict:
        """Single-shot: check if any job is due right now and run it."""
        return self._tick()

    def _tick(self) -> dict:
        now = datetime.now()
        state = _load_state()
        result: dict = {}

        # ── Pre-game check (Wednesday, hour >= PRE_GAME_HOUR) ────────────────
        if now.weekday() == _WEDNESDAY and now.hour >= _PRE_GAME_HOUR:
            week_key = _iso_week_key(now.date())
            pre_key = f"pre_game_{week_key}"
            if state.get(pre_key) != "done":
                upcoming = detect_upcoming_week()
                if upcoming:
                    season, week = upcoming
                    logger.info("Pre-game job due: season=%d week=%d", season, week)
                    if not self.dry_run:
                        r = run_pre_game(
                            season=season,
                            week=week,
                            n_draws=self.n_draws,
                            kelly_fraction=self.kelly_fraction,
                            min_ev=self.min_ev,
                        )
                        result["pre_game"] = r
                    state[pre_key] = "done"
                    _save_state(state)
                else:
                    logger.info("Pre-game: no upcoming games found (off-season?)")

        # ── Post-game check (Tuesday, hour >= POST_GAME_HOUR) ────────────────
        if now.weekday() == _TUESDAY and now.hour >= _POST_GAME_HOUR:
            week_key = _iso_week_key(now.date())
            post_key = f"post_game_{week_key}"
            if state.get(post_key) != "done":
                completed = detect_completed_week()
                if completed:
                    season, _ = completed
                    logger.info("Post-game backfill due: season=%d", season)
                    if not self.dry_run:
                        r = run_post_game(season=season, n_draws=self.n_draws)
                        result["post_game"] = r
                    state[post_key] = "done"
                    _save_state(state)
                else:
                    logger.info("Post-game: no completed games found (pre-season?)")

        return result
