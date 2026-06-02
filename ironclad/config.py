"""Central configuration: paths, constants, and environment."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("IRONCLAD_DATA_DIR", ROOT_DIR / "data"))
DB_PATH = DATA_DIR / "ironclad.ddb"
REPORTS_DIR = ROOT_DIR / "reports"
STATIC_DIR = DATA_DIR / "static"
MODELS_DIR = DATA_DIR / "models"

for _d in (DATA_DIR / "bronze", DATA_DIR / "silver", DATA_DIR / "gold",
           MODELS_DIR, STATIC_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API keys ───────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
DISCORD_WEBHOOK_URL: str = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ── Simulation defaults ────────────────────────────────────────────────────────
DEFAULT_N_DRAWS = 5000
KNOWLEDGE_CUTOFF_MARGIN_MINUTES = 30  # minutes before kickoff

# ── League-average priors (used when rolling history is sparse) ────────────────
LEAGUE_HOME_WIN_PROB = 0.573
LEAGUE_AVG_TOTAL = 47.5
LEAGUE_AVG_HOME_MARGIN = 2.5

LEAGUE_PRIORS = {
    "off_epa_per_play": 0.0,
    "def_epa_per_play": 0.0,
    "pass_rate": 0.575,
    "total_plays": 64.0,
    "sack_rate": 0.065,
    "points_per_game": 23.0,
    "yards_per_play": 5.5,
    "success_rate": 0.44,
}

# ── Injury status → availability probability ───────────────────────────────────
AVAILABILITY_MAP: dict[str, float] = {
    "Out": 0.00,
    "Doubtful": 0.25,
    "Questionable": 0.60,
    "Limited": 0.85,
    "Full Participation": 1.00,
    "Full": 1.00,
    "Active": 1.00,
    "DNP": 0.05,
}
AVAILABILITY_DEFAULT = 1.00  # no report = assumed available

# ── Rolling window ─────────────────────────────────────────────────────────────
ROLLING_WINDOW = 4  # L4 games
MIN_GAMES_FOR_ROLLING = 1  # start using rolling after this many games

# ── Feature store version ──────────────────────────────────────────────────────
FEATURE_VERSION = "v1.0"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("IRONCLAD_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("IRONCLAD_LOG_FORMAT", "text").lower()  # "text" | "json"


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts":      datetime.now(tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure root logger. Call once at process startup (CLI entry point).

    level: "DEBUG" | "INFO" | "WARNING" | "ERROR" (default: IRONCLAD_LOG_LEVEL env var)
    fmt:   "text" | "json"                         (default: IRONCLAD_LOG_FORMAT env var)
    """
    effective_level = (level or LOG_LEVEL).upper()
    effective_fmt = (fmt or LOG_FORMAT).lower()

    root = logging.getLogger()
    root.setLevel(effective_level)

    # Remove any handlers added by earlier basicConfig calls
    root.handlers.clear()

    handler = logging.StreamHandler()
    if effective_fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)
