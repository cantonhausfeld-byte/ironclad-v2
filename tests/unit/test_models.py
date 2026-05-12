"""Tests for model stubs and registry."""
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from ironclad.models.team.game_outcome import GameOutcomeModel
from ironclad.models.team.score_env import ScoreEnvironmentModel
from ironclad.models.player.usage import PlayerUsageModel
from ironclad.models.player.efficiency import PlayerEfficiencyModel
from ironclad.models.registry import ModelRegistry
from ironclad.models.calibration import IsotonicCalibrator
import numpy as np


def _team_row(home_win_prob=0.58, total=47.5, spread=-3.0):
    return pd.DataFrame([{
        "home_win_prob_from_odds": home_win_prob,
        "implied_total_from_odds": total,
        "spread_from_odds": spread,
        "off_pass_rate_l4": 0.58,
        "def_sack_rate_l4": 0.06,
        "rest_days": 7,
        "is_dome": False, "temp_f": 65.0, "wind_mph": 8.0,
    }])


def _player_row(pos="WR", avail=1.0):
    return pd.DataFrame([{
        "position": pos, "availability": avail,
        "depth_team": 1,
        "target_share_l4": 0.18, "carry_share_l4": 0.0,
        "catch_rate_l4": 0.65, "yards_per_target_l4": 8.2,
        "yards_per_carry_l4": 4.2,
        "td_rate_per_target_l4": 0.05, "td_rate_per_carry_l4": 0.04,
        "opp_def_pass_epa_l4": -0.02, "opp_def_rush_epa_l4": -0.01,
        "team_implied_total": 23.5,
    }])


# ── GameOutcomeModel stub ─────────────────────────────────────────────────────

def test_game_outcome_stub_probs_sum_to_one():
    m = GameOutcomeModel()
    out = m.predict(_team_row())
    assert abs(out["home_win_prob"] + out["away_win_prob"] - 1.0) < 1e-6


def test_game_outcome_uses_odds_implied():
    m = GameOutcomeModel()
    out = m.predict(_team_row(home_win_prob=0.7))
    assert abs(out["home_win_prob"] - 0.7) < 0.01


def test_game_outcome_total_positive():
    m = GameOutcomeModel()
    out = m.predict(_team_row(total=52.0))
    assert out["total_mean"] > 0


# ── ScoreEnvironmentModel ─────────────────────────────────────────────────────

def test_score_env_valid_ranges():
    m = ScoreEnvironmentModel()
    out = m.predict(_team_row())
    assert 0.3 <= out["pass_rate_projected"] <= 0.85
    assert out["total_plays_projected"] > 0
    assert out["team_total_projected"] > 0


# ── PlayerUsageModel ──────────────────────────────────────────────────────────

def test_player_usage_out_zeroes():
    m = PlayerUsageModel()
    out = m.predict(_player_row("WR", avail=0.0))
    assert out["targets_projected"] == 0.0
    assert out["carries_projected"] == 0.0


def test_player_usage_wr_positive_targets():
    m = PlayerUsageModel()
    out = m.predict(_player_row("WR"))
    assert out["targets_projected"] > 0


def test_player_usage_rb_positive_carries():
    m = PlayerUsageModel()
    out = m.predict(_player_row("RB"))
    assert out["carries_projected"] > 0


# ── PlayerEfficiencyModel ─────────────────────────────────────────────────────

def test_player_efficiency_rates_in_range():
    m = PlayerEfficiencyModel()
    out = m.predict(_player_row("WR"))
    assert 0.0 <= out["catch_rate"] <= 1.0
    assert out["yards_per_target"] > 0
    assert 0.0 <= out["td_rate_per_target"] <= 0.5


# ── ModelRegistry ─────────────────────────────────────────────────────────────

def test_registry_save_load_roundtrip():
    m = GameOutcomeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(Path(tmpdir))
        registry.save(m, {"test_metric": 0.5})
        loaded = registry.load("game_outcome")
        assert loaded.name == "game_outcome"
        meta = registry.metadata("game_outcome")
        assert meta["metrics"]["test_metric"] == 0.5


def test_registry_list_versions():
    m = GameOutcomeModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(Path(tmpdir))
        registry.save(m)
        versions = registry.list_versions("game_outcome")
        assert len(versions) >= 1


# ── IsotonicCalibrator ────────────────────────────────────────────────────────

def test_calibrator_fit_and_transform():
    cal = IsotonicCalibrator()
    probs = np.array([0.3, 0.5, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    cal.fit(probs, labels)
    out = cal.transform(probs)
    assert len(out) == 4
    assert all(0.0 <= p <= 1.0 for p in out)


def test_calibrator_passthrough_when_unfitted():
    cal = IsotonicCalibrator()
    probs = np.array([0.4, 0.6])
    out = cal.transform(probs)
    assert (out == probs).all()


# ── Per-week breakdown ────────────────────────────────────────────────────────

def _make_eval_frames(n: int = 40, seed: int = 42):
    """Minimal X_val / y_val with home_week column for breakdown testing."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"home_week": [(i % 18) + 1 for i in range(n)]})
    y = pd.DataFrame({
        "home_win": [i % 2 for i in range(n)],
        "home_margin": rng.normal(0, 10, size=n).tolist(),
        "total_score": rng.normal(45, 8, size=n).tolist(),
    })
    return X, y


def test_breakdown_by_week_structure(conn):
    from ironclad.models.trainer import ModelTrainer
    model = GameOutcomeModel()
    X_train, y_train = _make_eval_frames(n=60, seed=0)
    model.fit(X_train, y_train)

    trainer = ModelTrainer(conn)
    X_val, y_val = _make_eval_frames(n=40, seed=99)
    metrics = trainer._eval_game_outcome(model, X_val, y_val, breakdown_by_week=True)

    assert "by_week" in metrics
    for row in metrics["by_week"]:
        assert {"week", "n_games", "log_loss", "margin_mae", "margin_bias"} <= row.keys()
        assert isinstance(row["week"], int)
        assert row["n_games"] >= 1
        assert row["margin_mae"] >= 0.0


def test_breakdown_by_week_absent_without_flag(conn):
    from ironclad.models.trainer import ModelTrainer
    model = GameOutcomeModel()
    X_train, y_train = _make_eval_frames(n=60, seed=0)
    model.fit(X_train, y_train)

    trainer = ModelTrainer(conn)
    X_val, y_val = _make_eval_frames(n=40, seed=99)
    metrics = trainer._eval_game_outcome(model, X_val, y_val, breakdown_by_week=False)

    assert "by_week" not in metrics
    assert "val_log_loss" in metrics


# ── Hyperparameter kwargs ─────────────────────────────────────────────────────

def test_game_outcome_model_accepts_hp_kwargs():
    m = GameOutcomeModel(max_depth=3, n_estimators=50, learning_rate=0.1)
    assert m._hp["max_depth"] == 3
    assert m._hp["n_estimators"] == 50
    assert m._hp["learning_rate"] == pytest.approx(0.1)


def test_game_outcome_model_hp_kwargs_used_in_fit():
    X, y = _make_eval_frames(n=60, seed=7)
    m = GameOutcomeModel(max_depth=2, n_estimators=20)
    m.fit(X, y)
    out = m.predict(X.iloc[:1])
    assert 0.0 < out["home_win_prob"] < 1.0


def test_game_outcome_default_kwargs_unchanged():
    m = GameOutcomeModel()
    assert m._hp["max_depth"] == 4
    assert m._hp["n_estimators"] == 300
    assert m._hp["subsample"] == pytest.approx(0.8)


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def test_walk_forward_returns_folds(conn):
    from ironclad.models.trainer import ModelTrainer
    from unittest.mock import patch

    trainer = ModelTrainer(conn)

    def fake_train(train_seasons, val_seasons=None, **kwargs):
        return {
            "train_rows": len(train_seasons) * 100,
            "val_log_loss": 0.67,
            "val_margin_mae": 10.5,
            "val_brier": 0.23,
        }

    with patch.object(trainer, "train_game_outcome", side_effect=fake_train):
        result = trainer.walk_forward(folds_start=2019, folds_end=2021)

    assert len(result["folds"]) == 3
    assert result["folds"][0]["eval_season"] == 2019
    assert result["folds"][2]["eval_season"] == 2021
    assert result["avg_log_loss"] == pytest.approx(0.67)
    assert result["avg_margin_mae"] == pytest.approx(10.5)
    for fold in result["folds"]:
        assert {"eval_season", "train_rows", "val_log_loss", "val_margin_mae"} <= fold.keys()
