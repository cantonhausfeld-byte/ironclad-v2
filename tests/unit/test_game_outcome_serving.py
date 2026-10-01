"""Train/serve parity for the game outcome model.

Training and backtests feed the model one row per game with home_/away_
prefixed team features. Live simulation must do the same — predicting from the
home team's row alone silently drops the opponent from every *_diff feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ironclad.models.team.game_outcome import GameOutcomeModel, pivot_game_rows
from ironclad.simulation.engine import MonteCarloEngine


def _team(team: str, off_epa: float, game_id: str = "2026_04_AAA_BBB", **extra) -> pd.DataFrame:
    return pd.DataFrame([{
        "game_id": game_id, "team": team, "season": 2026, "week": 4,
        "off_epa_per_play_l4": off_epa, "def_epa_per_play_l4": 0.0,
        "rest_days": 7, **extra,
    }])


def _fitted_model(n: int = 400, seed: int = 0) -> GameOutcomeModel:
    """Model trained on games where the home side wins iff its EPA edge is positive."""
    rng = np.random.default_rng(seed)
    home_epa = rng.normal(0, 0.15, n)
    away_epa = rng.normal(0, 0.15, n)
    home = pd.concat(
        [_team("H", h, game_id=f"g{i}") for i, h in enumerate(home_epa)], ignore_index=True,
    )
    away = pd.concat(
        [_team("A", a, game_id=f"g{i}") for i, a in enumerate(away_epa)], ignore_index=True,
    )
    X = pivot_game_rows(home, away)
    margin = 60 * (home_epa - away_epa) + rng.normal(0, 3, n)
    y = pd.DataFrame({
        "home_win": (margin > 0).astype(int),
        "home_margin": margin,
        "total_score": 44 + rng.normal(0, 5, n),
    })
    model = GameOutcomeModel()
    model.fit(X, y)
    return model


def test_pivot_game_rows_prefixes_both_sides():
    X = pivot_game_rows(_team("H", 0.1), _team("A", -0.1))
    assert len(X) == 1
    assert X["game_id"].iloc[0] == "2026_04_AAA_BBB"
    assert X["home_off_epa_per_play_l4"].iloc[0] == 0.1
    assert X["away_off_epa_per_play_l4"].iloc[0] == -0.1
    assert X["home_opp_def_pass_epa_l4"].iloc[0] == 0.0


def test_prediction_depends_on_opponent():
    model = _fitted_model()
    home = _team("H", 0.10)
    vs_weak = model.predict(pivot_game_rows(home, _team("A", -0.25)))
    vs_strong = model.predict(pivot_game_rows(home, _team("A", 0.35)))
    assert vs_weak["home_win_prob"] > vs_strong["home_win_prob"] + 0.2
    assert vs_weak["home_margin_mean"] > vs_strong["home_margin_mean"]


def test_engine_feeds_outcome_model_both_teams():
    captured: dict[str, pd.DataFrame] = {}

    class Spy:
        def predict(self, X):
            captured["X"] = X
            return GameOutcomeModel().predict(X)

    engine = MonteCarloEngine(n_draws=10)
    engine._outcome_model = Spy()
    engine._bias_corrector = None
    engine._predict_outcome("H", "A", _team("H", 0.1), _team("A", -0.1))

    X = captured["X"]
    assert X["away_off_epa_per_play_l4"].iloc[0] == -0.1
    assert X["home_off_epa_per_play_l4"].iloc[0] == 0.1


def test_stub_reads_home_odds_from_pivoted_frame():
    home = _team("H", 0.0, home_win_prob_from_odds=0.7, implied_total_from_odds=51.0,
                 spread_from_odds=6.5)
    away = _team("A", 0.0, home_win_prob_from_odds=0.3, implied_total_from_odds=51.0,
                 spread_from_odds=-6.5)
    out = GameOutcomeModel().predict(pivot_game_rows(home, away))
    assert abs(out["home_win_prob"] - 0.7) < 1e-9
    assert out["total_mean"] == 51.0
    assert out["home_margin_mean"] == 6.5


def test_model_trained_before_a_feature_was_added_still_predicts(monkeypatch):
    from ironclad.models.team import game_outcome as go

    old_features = [f for f in go.TEAM_FEATURES if f != "elo_diff"]
    monkeypatch.setattr(go, "TEAM_FEATURES", old_features)
    monkeypatch.setattr(go, "CLF_FEATURES", old_features)
    model = _fitted_model()
    monkeypatch.undo()

    assert "elo_diff" in go.TEAM_FEATURES
    assert "elo_diff" not in model._clf.feature_names_in_
    out = model.predict(pivot_game_rows(_team("H", 0.1), _team("A", -0.1)))
    assert 0.0 < out["home_win_prob"] < 1.0


def test_margin_std_is_out_of_sample_not_memorised():
    """margin = 60 * EPA edge + N(0, 3): the simulation's margin std must be at
    least the true noise (3.0), not the smaller in-sample residual."""
    from ironclad.models.team.game_outcome import _build_diff_features, _model_matrix

    n = 600
    rng = np.random.default_rng(0)  # same draws as _fitted_model(seed=0)
    home_epa, away_epa = rng.normal(0, 0.15, n), rng.normal(0, 0.15, n)
    margin = 60 * (home_epa - away_epa) + rng.normal(0, 3, n)
    X = pivot_game_rows(
        pd.concat([_team("H", h, game_id=f"g{i}") for i, h in enumerate(home_epa)], ignore_index=True),
        pd.concat([_team("A", a, game_id=f"g{i}") for i, a in enumerate(away_epa)], ignore_index=True),
    )
    model = _fitted_model(n=n)
    Xm = _model_matrix(model._reg_margin, _build_diff_features(X), [])
    in_sample = float(np.std(margin - model._reg_margin.predict(Xm)))

    assert in_sample < 3.0 < model._margin_std < 5.0
