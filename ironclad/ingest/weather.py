"""Ingest weather data from Open-Meteo (free, no API key required)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
import requests

from ironclad.ingest.base import BaseIngestor
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_HOURLY_VARS = "temperature_2m,precipitation,windspeed_10m,winddirection_10m"
_GAME_DURATION_HOURS = 4


class WeatherIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, games: pd.DataFrame) -> int:
        """Fetch weather for a DataFrame of games with columns: game_id, gameday, lat, lon.

        Optional columns: gametime_local or gametime (HH:MM:SS string).
        Uses hourly Open-Meteo data at kickoff time for accuracy.
        """
        rows = []
        for _, row in games.iterrows():
            try:
                gametime = row.get("gametime_local") or row.get("gametime")
                rec = _fetch_weather(
                    game_id=row["game_id"],
                    lat=row.get("lat"),
                    lon=row.get("lon"),
                    gameday=row["gameday"],
                    gametime_local=gametime,
                )
                if rec:
                    rows.append(rec)
            except Exception as exc:
                logger.warning("Weather fetch failed for %s: %s", row["game_id"], exc)

        if not rows:
            return 0
        df = pd.DataFrame(rows)
        return self._writer.write_weather(df)


def _fetch_weather(
    game_id: str,
    lat: float | None,
    lon: float | None,
    gameday,
    gametime_local: str | None = None,
) -> dict | None:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None

    if isinstance(gameday, str):
        gameday = date.fromisoformat(gameday[:10])
    elif hasattr(gameday, "date"):
        gameday = gameday.date()

    # Parse kickoff hour from gametime (default 13 = 1pm local if unknown)
    kickoff_hour = 13
    if gametime_local:
        try:
            kickoff_hour = int(str(gametime_local).split(":")[0])
        except (ValueError, IndexError):
            pass

    today = date.today()
    is_historical = gameday <= today
    url = OPEN_METEO_URL if is_historical else OPEN_METEO_FORECAST_URL

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": _HOURLY_VARS,
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "start_date": str(gameday),
        "end_date": str(gameday),
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("windspeed_10m", [])
    wind_dirs = hourly.get("winddirection_10m", [])
    precips = hourly.get("precipitation", [])

    # Pick the hour index that matches kickoff (API returns 0..23 in order)
    idx = min(kickoff_hour, len(temps) - 1) if temps else 0

    # Sum precipitation over the game window (kickoff + up to 4 hours)
    game_slice = slice(kickoff_hour, kickoff_hour + _GAME_DURATION_HOURS)
    precip_total = sum(p for p in precips[game_slice] if p is not None) if precips else None

    return {
        "game_id": game_id,
        "retrieved_at": datetime.now(tz=timezone.utc),
        "source": "open-meteo",
        "temp_f": temps[idx] if temps else None,
        "wind_mph": winds[idx] if winds else None,
        "wind_dir": wind_dirs[idx] if winds else None,
        "precip_in": precip_total,
        "humidity_pct": None,
        "conditions": None,
    }
