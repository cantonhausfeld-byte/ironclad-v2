"""Walk-forward backtester fold construction."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from ironclad.eval.backtester import Backtester


def _frames(seasons, n=40, seed=0):
    rng = np.random.default_rng(seed + sum(seasons))
    X = pd.DataFrame({
        "game_id": [f"{seasons[0]}_{i}" for i in range(n)],
        "home_off_epa_per_play_l4": rng.normal(0, 0.1, n),
        "away_off_epa_per_play_l4": rng.normal(0, 0.1, n),
        "home_week": [(i % 17) + 1 for i in range(n)],
        "home_season": [seasons[0]] * n,
        "home_team": ["AAA"] * n,
        "away_team": ["BBB"] * n,
    })
    y = pd.DataFrame({
        "home_win": rng.integers(0, 2, n),
        "home_margin": rng.normal(0, 10, n),
        "total_score": rng.normal(44, 8, n),
    })
    return X, y


def test_calibration_season_is_held_out_of_model_fit(conn):
    loads: list[list[int]] = []
    fits: list[int] = []

    def fake_load(self, seasons):
        loads.append(list(seasons))
        return _frames(list(seasons))

    from ironclad.models.team.game_outcome import GameOutcomeModel
    real_fit = GameOutcomeModel.fit

    def spy_fit(self, X, y):
        fits.append(len(X))
        return real_fit(self, X, y)

    with (
        patch("ironclad.models.trainer.ModelTrainer._load_team_data", fake_load),
        patch.object(GameOutcomeModel, "fit", spy_fit),
    ):
        Backtester(conn)._run_fold("t", [2016, 2017, 2018], 2019)

    # Fit on 2016-2017, calibrate on held-out 2018, predict 2019
    assert loads[:3] == [[2016, 2017], [2018], [2019]]
    assert len(fits) == 1
