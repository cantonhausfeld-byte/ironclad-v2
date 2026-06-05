"""Ingest weather data from Open-Meteo (primary) with NASA POWER fallback."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from ironclad.ingest.base import BaseIngestor
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

_HOURLY_VARS = "temperature_2m,precipitation,windspeed_10m,winddirection_10m"
_GAME_DURATION_HOURS = 4
# Forecast window Open-Meteo supports (days from today); beyond this → future fallback
_FORECAST_HORIZON_DAYS = 16


class WeatherIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self, games: pd.DataFrame) -> int:
        """Fetch weather for a DataFrame of games with columns: game_id, gameday, lat, lon.

        Skips games already stored in bronze.weather. Falls back to NASA POWER
        on Open-Meteo 429 (rate limit) and to a same-date prior-year climate
        proxy on 400 (future date beyond forecast horizon).
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


# ── Primary: Open-Meteo ───────────────────────────────────────────────────────

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

    kickoff_hour = _parse_kickoff_hour(gametime_local)
    today = date.today()
    days_out = (gameday - today).days

    if days_out <= 0:
        # Historical — try archive, fall back to NASA POWER on 429
        return _try_open_meteo_archive(game_id, lat, lon, gameday, kickoff_hour) \
            or _fetch_weather_nasa(game_id, lat, lon, gameday, kickoff_hour)
    elif days_out <= _FORECAST_HORIZON_DAYS:
        # Near-future — use forecast endpoint
        return _try_open_meteo_forecast(game_id, lat, lon, gameday, kickoff_hour)
    else:
        # Far-future — use same calendar date from prior year as climate proxy
        return _fetch_weather_climate_proxy(game_id, lat, lon, gameday, kickoff_hour)


def _parse_kickoff_hour(gametime_local: str | None) -> int:
    if gametime_local:
        try:
            return int(str(gametime_local).split(":")[0])
        except (ValueError, IndexError):
            pass
    return 13  # default 1 pm local


def _open_meteo_params(lat, lon, gameday) -> dict:
    return {
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


def _parse_open_meteo(data: dict, game_id: str, kickoff_hour: int, source: str) -> dict | None:
    hourly = data.get("hourly", {})
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("windspeed_10m", [])
    wind_dirs = hourly.get("winddirection_10m", [])
    precips = hourly.get("precipitation", [])
    if not temps:
        return None
    idx = min(kickoff_hour, len(temps) - 1)
    game_slice = slice(kickoff_hour, kickoff_hour + _GAME_DURATION_HOURS)
    precip_total = sum(p for p in precips[game_slice] if p is not None) if precips else None
    return {
        "game_id": game_id,
        "retrieved_at": datetime.now(tz=timezone.utc),
        "source": source,
        "temp_f": temps[idx],
        "wind_mph": winds[idx] if winds else None,
        "wind_dir": wind_dirs[idx] if wind_dirs else None,
        "precip_in": precip_total,
        "humidity_pct": None,
        "conditions": None,
    }


def _try_open_meteo_archive(game_id, lat, lon, gameday, kickoff_hour) -> dict | None:
    try:
        resp = requests.get(
            OPEN_METEO_ARCHIVE_URL,
            params=_open_meteo_params(lat, lon, gameday),
            timeout=10,
        )
        resp.raise_for_status()
        return _parse_open_meteo(resp.json(), game_id, kickoff_hour, "open-meteo")
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            logger.debug("Open-Meteo archive rate-limited for %s; trying NASA POWER", game_id)
        else:
            logger.debug("Open-Meteo archive error for %s: %s", game_id, e)
        return None
    except Exception as e:
        logger.debug("Open-Meteo archive error for %s: %s", game_id, e)
        return None


def _try_open_meteo_forecast(game_id, lat, lon, gameday, kickoff_hour) -> dict | None:
    try:
        resp = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=_open_meteo_params(lat, lon, gameday),
            timeout=10,
        )
        resp.raise_for_status()
        return _parse_open_meteo(resp.json(), game_id, kickoff_hour, "open-meteo-forecast")
    except Exception as e:
        logger.debug("Open-Meteo forecast error for %s: %s", game_id, e)
        return None


# ── Fallback 1: NASA POWER (historical, no rate limit) ───────────────────────

def _fetch_weather_nasa(game_id, lat, lon, gameday, kickoff_hour) -> dict | None:
    """Fetch historical weather from NASA POWER (ERA5-based, no API key, no rate limit)."""
    try:
        date_str = gameday.strftime("%Y%m%d")
        resp = requests.get(
            NASA_POWER_URL,
            params={
                "parameters": "T2M,WS10M,PRECTOTCORR",
                "community": "RE",
                "latitude": lat,
                "longitude": lon,
                "start": date_str,
                "end": date_str,
                "format": "JSON",
            },
            timeout=20,
        )
        resp.raise_for_status()
        param = resp.json()["properties"]["parameter"]
        t_c = param.get("T2M", {}).get(date_str)
        ws = param.get("WS10M", {}).get(date_str)
        pr = param.get("PRECTOTCORR", {}).get(date_str)
        if t_c is None:
            return None
        return {
            "game_id": game_id,
            "retrieved_at": datetime.now(tz=timezone.utc),
            "source": "nasa-power",
            "temp_f": round(t_c * 1.8 + 32, 1) if t_c is not None else None,
            "wind_mph": round(ws * 2.237, 1) if ws is not None else None,
            "wind_dir": None,
            "precip_in": round(pr / 25.4, 3) if pr is not None else None,
            "humidity_pct": None,
            "conditions": None,
        }
    except Exception as e:
        logger.debug("NASA POWER error for %s: %s", game_id, e)
        return None


# ── Fallback 2: climate proxy (far-future dates) ─────────────────────────────

def _fetch_weather_climate_proxy(game_id, lat, lon, gameday, kickoff_hour) -> dict | None:
    """Estimate future-game weather using the same calendar date from recent prior years.

    Tries the previous 3 years in order and returns the first successful result,
    tagged source='open-meteo-climate-proxy' so downstream code knows it is an
    estimate rather than an actual observation.
    """
    for years_back in range(1, 4):
        try:
            proxy_day = gameday.replace(year=gameday.year - years_back)
        except ValueError:
            # Feb 29 edge case
            proxy_day = gameday.replace(year=gameday.year - years_back, day=28)
        rec = _try_open_meteo_archive(game_id, lat, lon, proxy_day, kickoff_hour) \
            or _fetch_weather_nasa(game_id, lat, lon, proxy_day, kickoff_hour)
        if rec:
            rec["source"] = "open-meteo-climate-proxy"
            rec["game_id"] = game_id  # keep original game_id, not proxy date's
            logger.debug(
                "Climate proxy for %s: using %s data (-%d yr)",
                game_id, proxy_day, years_back,
            )
            return rec
    return None
