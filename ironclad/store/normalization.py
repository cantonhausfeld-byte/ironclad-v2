"""Canonical team abbreviation normalization for ironclad-v2.

Maps historical and variant abbreviations to their current canonical form.
Apply at ingest boundaries so all downstream tables use consistent codes.
"""
from __future__ import annotations

import pandas as pd

# Maps any historical/variant abbreviation → current canonical abbreviation.
# Covers franchise relocations and known nfl_data_py/nflverse inconsistencies.
TEAM_MAP: dict[str, str] = {
    # Relocated franchises
    "OAK": "LV",    # Raiders: Oakland → Las Vegas (2020)
    "SD":  "LAC",   # Chargers: San Diego → Los Angeles (2017)
    "STL": "LAR",   # Rams: St. Louis → Los Angeles (2016)
    # nfl_data_py used "LA" for Rams during 2016–2018 transition
    "LA":  "LAR",
    # Jaguars abbreviation variants
    "JAC": "JAX",
    # Occasional typos / legacy codes in older data feeds
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "ARZ": "ARI",
    "SL":  "LAR",   # very old STL variant
}


# Maps The Odds API full team names → canonical abbreviations.
_FULL_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # Historical names
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
    "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LAR",
}


def full_name_to_abbr(name: str | None) -> str | None:
    """Map an Odds API full team name to its canonical abbreviation, or None if unknown."""
    if not name:
        return None
    return _FULL_NAME_TO_ABBR.get(name.strip())


def normalize_team(team: str | None) -> str | None:
    """Return canonical team abbreviation, or the input unchanged if unknown."""
    if not team or not isinstance(team, str):
        return team
    t = team.strip().upper()
    return TEAM_MAP.get(t, t)


def normalize_teams(series: pd.Series) -> pd.Series:
    """Vectorized version of normalize_team for a pandas Series."""
    return series.map(lambda t: normalize_team(t) if pd.notna(t) else t)
