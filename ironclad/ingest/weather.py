"""Ingest weather data from Open-Meteo (free, no API key required)."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import requests
import pandas as pd

from ironclad.ingest.base import BaseIngestor
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, games: pd.DataFrame) -> int:
        """Fetch weather for a DataFrame of games with columns: game_id, gameday, lat, lon, gametime."""
        rows = []
        for _, row in games.iterrows():
            try:
                rec = _fetch_weather(
                    game_id=row["game_id"],
                    lat=row.get("lat"),
                    lon=row.get("lon"),
                    gameday=row["gameday"],
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
) -> dict | None:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None

    if isinstance(gameday, str):
        gameday = date.fromisoformat(gameday)
    elif hasattr(gameday, "date"):
        gameday = gameday.date()

    today = date.today()
    is_historical = gameday <= today

    url = OPEN_METEO_URL if is_historical else OPEN_METEO_FORECAST_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max",
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

    daily = data.get("daily", {})
    temps = daily.get("temperature_2m_max", [None])
    winds = daily.get("windspeed_10m_max", [None])
    precips = daily.get("precipitation_sum", [None])

    return {
        "game_id": game_id,
        "retrieved_at": datetime.now(tz=timezone.utc),
        "source": "open-meteo",
        "temp_f": temps[0] if temps else None,
        "wind_mph": winds[0] if winds else None,
        "wind_dir": None,
        "precip_in": precips[0] if precips else None,
        "humidity_pct": None,
        "conditions": None,
    }
