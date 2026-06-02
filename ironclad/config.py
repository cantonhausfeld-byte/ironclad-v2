"""Central configuration: paths, constants, and environment."""
import os
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
