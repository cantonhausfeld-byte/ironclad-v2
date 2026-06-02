"""Ingest game odds and player props from The Odds API (free tier: 500 req/month)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from ironclad.config import ODDS_API_KEY
from ironclad.ingest.base import BaseIngestor
from ironclad.store.connection import get_connection
from ironclad.store.normalization import full_name_to_abbr
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# Preferred bookmaker for player props (minimises API request count).
_PREFERRED_BOOK = "draftkings"

# The Odds API player-prop market keys → our internal stat_type names.
_PROP_MARKET_MAP: dict[str, str] = {
    "player_pass_yds":       "pass_yards",
    "player_rush_yds":       "rush_yards",
    "player_reception_yds":  "rec_yards",
    "player_receptions":     "receptions",
    "player_anytime_td":     "anytime_td",
}
_PROP_MARKETS_CSV = ",".join(_PROP_MARKET_MAP)


class OddsIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None, conn=None) -> None:
        self._writer = writer or BronzeWriter()
        self._conn = conn or get_connection()

    def _ingest(self, game_ids: list[str] | None = None) -> int:
        if not ODDS_API_KEY:
            logger.warning("ODDS_API_KEY not set; skipping odds ingest")
            return 0

        # ── 1. Fetch game-level odds ──────────────────────────────────────────
        resp = requests.get(
            f"{BASE_URL}/sports/{SPORT}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "spreads,totals,h2h",
                "oddsFormat": "american",
            },
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()

        odds_rows: list[dict] = []
        prop_rows: list[dict] = []
        ts = datetime.now(tz=timezone.utc)

        for event in events:
            event_id = event.get("id", "")
            home_full = event.get("home_team", "")
            away_full = event.get("away_team", "")
            commence_time = event.get("commence_time", "")

            home_abbr = full_name_to_abbr(home_full)
            away_abbr = full_name_to_abbr(away_full)
            game_id = _resolve_game_id(
                self._conn, home_abbr, away_abbr, commence_time, event_id
            )

            odds_rows.extend(_parse_game_odds(event, game_id, home_full, away_full, ts))

            # ── 2. Fetch player props for this event ──────────────────────────
            if ODDS_API_KEY:
                props = _fetch_player_props(
                    event_id, game_id, home_abbr, away_abbr, self._conn, ts
                )
                prop_rows.extend(props)

        total = 0
        if odds_rows:
            total += self._writer.write_odds(pd.DataFrame(odds_rows))
        if prop_rows:
            total += self._writer.write_player_props(pd.DataFrame(prop_rows))
        return total


# ── Game odds parsing ─────────────────────────────────────────────────────────

def _parse_game_odds(
    event: dict,
    game_id: str,
    home_full: str,
    away_full: str,
    ts: datetime,
) -> list[dict]:
    rows = []
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
                h = outcomes.get(home_full, {})
                a = outcomes.get(away_full, {})
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
                ml_home = outcomes.get(home_full, {}).get("price")
                ml_away = outcomes.get(away_full, {}).get("price")

        rows.append({
            "game_id":          game_id,
            "source":           source,
            "retrieved_at":     ts,
            "spread_home":      spread_home,
            "spread_away":      spread_away,
            "spread_juice_home": spread_juice_home,
            "spread_juice_away": spread_juice_away,
            "total_over":       total_over,
            "total_juice_over": total_juice_over,
            "total_juice_under": total_juice_under,
            "moneyline_home":   ml_home,
            "moneyline_away":   ml_away,
        })
    return rows


# ── Player prop fetching ──────────────────────────────────────────────────────

def _fetch_player_props(
    event_id: str,
    game_id: str,
    home_abbr: str | None,
    away_abbr: str | None,
    conn,
    ts: datetime,
) -> list[dict]:
    try:
        resp = requests.get(
            f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": _PROP_MARKETS_CSV,
                "oddsFormat": "american",
                "bookmakers": _PREFERRED_BOOK,
            },
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Player prop fetch failed for event %s: %s", event_id, exc)
        return []

    data = resp.json()
    rows: list[dict] = []

    for bookmaker in data.get("bookmakers", []):
        bk = bookmaker.get("key", "unknown")
        for market in bookmaker.get("markets", []):
            stat_type = _PROP_MARKET_MAP.get(market.get("key", ""))
            if not stat_type:
                continue

            # Group outcomes by player name: {name: {line, over_odds, under_odds}}
            by_player: dict[str, dict] = {}
            for outcome in market.get("outcomes", []):
                pname = outcome.get("name", "")
                desc = (outcome.get("description") or "").lower()
                price = outcome.get("price")
                point = outcome.get("point")
                if pname not in by_player:
                    by_player[pname] = {"line": point, "over_odds": None, "under_odds": None}
                if stat_type == "anytime_td" or "over" in desc:
                    by_player[pname]["over_odds"] = price
                    if point is not None:
                        by_player[pname]["line"] = point
                elif "under" in desc:
                    by_player[pname]["under_odds"] = price

            for pname, prop in by_player.items():
                # Try to resolve team and player_id from rosters
                team_abbr, player_id = _resolve_player(conn, pname, home_abbr, away_abbr)
                rows.append({
                    "game_id":      game_id,
                    "event_id":     event_id,
                    "player_name":  pname,
                    "player_id":    player_id,
                    "team":         team_abbr,
                    "stat_type":    stat_type,
                    "line":         prop.get("line"),
                    "over_odds":    prop.get("over_odds"),
                    "under_odds":   prop.get("under_odds"),
                    "bookmaker":    bk,
                    "retrieved_at": ts,
                })
    return rows


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_game_id(
    conn,
    home_abbr: str | None,
    away_abbr: str | None,
    commence_time: str,
    event_id: str,
) -> str:
    """Return canonical game_id from silver.games, or fallback ODDS_{event_id}."""
    if not home_abbr or not away_abbr or not commence_time:
        return f"ODDS_{event_id}"
    gameday = commence_time[:10]  # "YYYY-MM-DD"
    try:
        row = conn.execute(
            "SELECT game_id FROM silver.games "
            "WHERE home_team = ? AND away_team = ? AND CAST(gameday AS VARCHAR) = ?",
            [home_abbr, away_abbr, gameday],
        ).fetchone()
        if row:
            return row[0]
        # Try ±1 day for timezone edge cases
        from datetime import date, timedelta
        d = date.fromisoformat(gameday)
        for delta in (-1, 1):
            alt = (d + timedelta(days=delta)).isoformat()
            row = conn.execute(
                "SELECT game_id FROM silver.games "
                "WHERE home_team = ? AND away_team = ? AND CAST(gameday AS VARCHAR) = ?",
                [home_abbr, away_abbr, alt],
            ).fetchone()
            if row:
                return row[0]
    except Exception as exc:
        logger.debug("game_id lookup failed: %s", exc)
    return f"ODDS_{event_id}"


def _resolve_player(
    conn,
    player_name: str,
    home_abbr: str | None,
    away_abbr: str | None,
) -> tuple[str | None, str | None]:
    """Return (team_abbr, player_id) from bronze.rosters, or (None, None) if not found."""
    try:
        teams = [t for t in [home_abbr, away_abbr] if t]
        if teams:
            placeholders = ",".join("?" * len(teams))
            row = conn.execute(
                f"SELECT team, player_id FROM bronze.rosters "
                f"WHERE player_name = ? AND team IN ({placeholders}) "
                f"ORDER BY _ingest_ts DESC LIMIT 1",
                [player_name, *teams],
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT team, player_id FROM bronze.rosters "
                "WHERE player_name = ? ORDER BY _ingest_ts DESC LIMIT 1",
                [player_name],
            ).fetchone()
        if row:
            return row[0], row[1]
    except Exception as exc:
        logger.debug("Player lookup failed for %r: %s", player_name, exc)
    return None, None
