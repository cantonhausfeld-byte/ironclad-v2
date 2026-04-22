"""Ingest odds from The Odds API (free tier: 500 req/month)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests
import pandas as pd

from ironclad.config import ODDS_API_KEY
from ironclad.ingest.base import BaseIngestor
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"


class OddsIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, game_ids: list[str] | None = None) -> int:
        if not ODDS_API_KEY:
            logger.warning("ODDS_API_KEY not set; skipping odds ingest")
            return 0

        url = f"{BASE_URL}/sports/{SPORT}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "spreads,totals,h2h",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json()

        rows = []
        for event in events:
            rows.extend(_parse_event(event))

        if not rows:
            return 0
        df = pd.DataFrame(rows)
        return self._writer.write_odds(df)


def _parse_event(event: dict) -> list[dict]:
    rows = []
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    # Use commence_time + teams to build a rough game_id for matching later
    game_id = f"ODDS_{event.get('id', '')}"
    ts = datetime.now(tz=timezone.utc)

    for bookmaker in event.get("bookmakers", []):
        source = bookmaker.get("key", "unknown")
        spread_home = spread_away = None
        spread_juice_home = spread_juice_away = None
        total_over = total_juice_over = total_juice_under = None
        ml_home = ml_away = None

        for market in bookmaker.get("markets", []):
            mk = market.get("key")
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            if mk == "spreads":
                h = outcomes.get(home, {})
                a = outcomes.get(away, {})
                spread_home = h.get("point")
                spread_juice_home = h.get("price")
                spread_away = a.get("point")
                spread_juice_away = a.get("price")
            elif mk == "totals":
                ov = outcomes.get("Over", {})
                un = outcomes.get("Under", {})
                total_over = ov.get("point")
                total_juice_over = ov.get("price")
                total_juice_under = un.get("price")
            elif mk == "h2h":
                ml_home = outcomes.get(home, {}).get("price")
                ml_away = outcomes.get(away, {}).get("price")

        rows.append({
            "game_id": game_id,
            "source": source,
            "retrieved_at": ts,
            "spread_home": spread_home,
            "spread_away": spread_away,
            "spread_juice_home": spread_juice_home,
            "spread_juice_away": spread_juice_away,
            "total_over": total_over,
            "total_juice_over": total_juice_over,
            "total_juice_under": total_juice_under,
            "moneyline_home": ml_home,
            "moneyline_away": ml_away,
        })
    return rows
