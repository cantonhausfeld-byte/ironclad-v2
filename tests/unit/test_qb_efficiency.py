"""Tests for QB efficiency sub-model inside PlayerEfficiencyModel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ironclad.models.player.efficiency import PlayerEfficiencyModel, _EFFICIENCY_PRIORS


def _make_qb_df(n: int = 80, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a minimal DataFrame with QB rows shaped like gold.player_game_features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "position":              ["QB"] * n,
        "season":                rng.integers(2018, 2025, size=n).tolist(),
        "catch_rate_l4":         rng.uniform(0.58, 0.70, size=n),
        "yards_per_target_l4":   rng.uniform(6.5, 9.5, size=n),
        "td_rate_per_target_l4": rng.uniform(0.02, 0.08, size=n),
        "adot_l4":               rng.uniform(7.0, 13.0, size=n),
        "cpoe_l4":               rng.uniform(-5.0, 8.0, size=n),
        "aggressiveness_l4":     rng.uniform(10.0, 22.0, size=n),
        "opp_def_pass_epa_l4":   rng.uniform(-0.1, 0.1, size=n),
        "opp_def_sack_rate_l4":  rng.uniform(0.04, 0.10, size=n),
        "team_off_pass_rate_l4": rng.uniform(0.50, 0.65, size=n),
        "wind_mph":              rng.uniform(0.0, 20.0, size=n),
        "temp_f":                rng.uniform(30.0, 80.0, size=n),
        "is_dome":               rng.choice([0, 1], size=n).tolist(),
    })
    y = pd.DataFrame({
        "target_catch_rate":    rng.uniform(0.58, 0.72, size=n),
        "target_yds_per_target": rng.uniform(6.0, 10.0, size=n),
        "target_td_rate_tgt":   rng.uniform(0.02, 0.08, size=n),
        # Unused columns present in real training data
        "target_yds_per_carry": rng.uniform(3.5, 6.0, size=n),
        "target_td_rate_carry": rng.uniform(0.02, 0.10, size=n),
    })
    return X, y


def _make_wr_df(n: int = 60, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a minimal WR DataFrame."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "position":              ["WR"] * n,
        "season":                rng.integers(2018, 2025, size=n).tolist(),
        "catch_rate_l4":         rng.uniform(0.55, 0.75, size=n),
        "yards_per_target_l4":   rng.uniform(7.0, 11.0, size=n),
        "yards_per_carry_l4":    rng.uniform(3.0, 7.0, size=n),
        "yac_per_rec_l4":        rng.uniform(2.0, 6.0, size=n),
        "td_rate_per_target_l4": rng.uniform(0.04, 0.12, size=n),
        "td_rate_per_carry_l4":  rng.uniform(0.02, 0.08, size=n),
        "opp_def_pass_epa_l4":   rng.uniform(-0.1, 0.1, size=n),
        "opp_def_rush_epa_l4":   rng.uniform(-0.1, 0.1, size=n),
        "opp_def_sack_rate_l4":  rng.uniform(0.04, 0.10, size=n),
        "is_dome":               rng.choice([0, 1], size=n).tolist(),
        "temp_f":                rng.uniform(30.0, 80.0, size=n),
        "wind_mph":              rng.uniform(0.0, 20.0, size=n),
        "team_off_epa_l4":       rng.uniform(-0.1, 0.3, size=n),
    })
    y = pd.DataFrame({
        "target_catch_rate":    rng.uniform(0.55, 0.75, size=n),
        "target_yds_per_target": rng.uniform(7.0, 11.0, size=n),
        "target_yds_per_carry": rng.uniform(3.0, 7.0, size=n),
        "target_td_rate_tgt":   rng.uniform(0.04, 0.12, size=n),
        "target_td_rate_carry": rng.uniform(0.02, 0.08, size=n),
    })
    return X, y


# ── Test 1: fit() trains QB regressors ───────────────────────────────────────

def test_qb_efficiency_fit_trains_qb_regressors():
    model = PlayerEfficiencyModel()
    X_qb, y_qb = _make_qb_df(n=80)
    model.fit(X_qb, y_qb)
    assert model._qb_regs is not None
    for key in ("catch_rate", "yards_per_target", "td_rate_per_target"):
        assert key in model._qb_regs, f"QB regressor '{key}' not trained"


# ── Test 2: QB predict returns valid range ────────────────────────────────────

def test_qb_efficiency_predict_qb_in_valid_range():
    model = PlayerEfficiencyModel()
    X_qb, y_qb = _make_qb_df(n=80)
    model.fit(X_qb, y_qb)

    row = X_qb.iloc[[0]]
    result = model.predict(row)

    assert "catch_rate" in result
    assert "yards_per_target" in result
    assert 0.50 <= result["catch_rate"] <= 0.75, f"completion% out of range: {result['catch_rate']}"
    assert result["yards_per_target"] >= 4.0, f"YPA too low: {result['yards_per_target']}"
    assert result["yards_per_target"] <= 13.0


# ── Test 3: Non-QB uses standard path ────────────────────────────────────────

def test_qb_efficiency_non_qb_uses_standard_path():
    model = PlayerEfficiencyModel()
    X_qb, y_qb = _make_qb_df(n=80)
    X_wr, y_wr = _make_wr_df(n=60)

    X_all = pd.concat([X_qb, X_wr], ignore_index=True)
    y_all = pd.concat([y_qb, y_wr], ignore_index=True)
    model.fit(X_all, y_all)

    # WR should use the standard path (not _predict_qb)
    wr_row = X_wr.iloc[[0]]
    result = model.predict(wr_row)
    assert result["catch_rate"] is not None
    # WR catch rates can go higher than 0.75 (standard path allows up to 0.99)
    assert result["catch_rate"] <= 0.99


# ── Test 4: Unfitted model falls back to stub priors ─────────────────────────

def test_qb_efficiency_unfitted_falls_back_to_stub():
    model = PlayerEfficiencyModel()
    X_qb, _ = _make_qb_df(n=1)
    row = X_qb.iloc[[0]]

    result = model.predict(row)
    # Should return QB position prior (0.64) since no QB regressors trained
    qb_prior = _EFFICIENCY_PRIORS["QB"]["catch_rate"]
    assert abs(result["catch_rate"] - qb_prior) < 0.01, (
        f"Expected stub prior {qb_prior}, got {result['catch_rate']}"
    )


# ── Test 5: Tolerates null cpoe_l4 / aggressiveness_l4 ───────────────────────

def test_qb_efficiency_tolerates_null_cpoe():
    model = PlayerEfficiencyModel()
    X_qb, y_qb = _make_qb_df(n=80)
    model.fit(X_qb, y_qb)

    # Set NGS features to NaN on the prediction row
    row = X_qb.iloc[[0]].copy()
    row["cpoe_l4"] = float("nan")
    row["aggressiveness_l4"] = float("nan")

    # Should not raise; _to_xgb fills NaN with 0.0
    result = model.predict(row)
    assert "catch_rate" in result
    assert 0.50 <= result["catch_rate"] <= 0.75
