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


def normalize_team(team: str | None) -> str | None:
    """Return canonical team abbreviation, or the input unchanged if unknown."""
    if not team or not isinstance(team, str):
        return team
    t = team.strip().upper()
    return TEAM_MAP.get(t, t)


def normalize_teams(series: pd.Series) -> pd.Series:
    """Vectorized version of normalize_team for a pandas Series."""
    return series.map(lambda t: normalize_team(t) if pd.notna(t) else t)
