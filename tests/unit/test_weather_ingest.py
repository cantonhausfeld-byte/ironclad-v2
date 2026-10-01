"""Weather ingest: batched writes, per-game failure isolation, Open-Meteo circuit breaker."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from ironclad.ingest import weather
from ironclad.ingest.weather import WeatherIngestor, _CircuitBreaker


def _games(n):
    return pd.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "gameday": ["2023-10-01"] * n,
        "lat": [39.0] * n,
        "lon": [-94.5] * n,
    })


class _Writer:
    def __init__(self):
        self.batches = []

    def write_weather(self, df):
        self.batches.append(df)
        return len(df)


def test_writes_in_batches_and_skips_failed_games(monkeypatch):
    monkeypatch.setattr(weather, "_WRITE_BATCH", 3)

    def fake_fetch(game_id, **_):
        if game_id == "g4":
            raise RuntimeError("boom")
        return {"game_id": game_id, "temp_f": 60.0}

    writer = _Writer()
    with patch.object(weather, "_fetch_weather", side_effect=fake_fetch):
        n = WeatherIngestor(writer).ingest(_games(8))

    assert n == 7
    assert [len(b) for b in writer.batches] == [3, 3, 1]
    stored = sorted(g for b in writer.batches for g in b["game_id"])
    assert stored == ["g0", "g1", "g2", "g3", "g5", "g6", "g7"]


def test_breaker_opens_after_threshold_and_resets_on_success():
    b = _CircuitBreaker(threshold=3)
    for _ in range(2):
        b.record(ok=False)
    assert not b.open
    b.record(ok=True)
    for _ in range(3):
        b.record(ok=False)
    assert b.open
    b.reset()
    assert not b.open


def test_open_breaker_skips_open_meteo_and_falls_back_to_nasa(monkeypatch):
    monkeypatch.setattr(weather, "_open_meteo_breaker", _CircuitBreaker(threshold=2))
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.HTTPError(response=MagicMock(status_code=429))
    nasa = MagicMock(return_value={"game_id": "g", "source": "nasa-power"})

    with (
        patch.object(weather.requests, "get", return_value=resp) as get,
        patch.object(weather, "_fetch_weather_nasa", nasa),
    ):
        for _ in range(5):
            rec = weather._fetch_weather("g", 39.0, -94.5, "2023-10-01", "13:00")
            assert rec["source"] == "nasa-power"

    assert get.call_count == 2  # breaker opened after two 429s
    assert nasa.call_count == 5
