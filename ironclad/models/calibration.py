"""Calibration wrappers for probability outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class IsotonicCalibrator:
    """Post-hoc isotonic calibration for probability outputs."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        self._iso.fit(probs, labels)
        self._fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probs
        return self._iso.transform(probs)

    @property
    def fitted(self) -> bool:
        return self._fitted


class PlattCalibrator:
    """
    Platt scaling: fits sigmoid(A * logit(raw_prob) + B) via logistic regression.

    Only 2 parameters — robust for small calibration sets (100-500 samples).
    Use instead of IsotonicCalibrator when calibration data < ~1000 samples.
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        from scipy.special import logit

        logits = logit(np.clip(probs, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        self._lr.fit(logits, labels)
        self._fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probs
        from scipy.special import logit

        logits = logit(np.clip(probs, 1e-7, 1 - 1e-7)).reshape(-1, 1)
        return self._lr.predict_proba(logits)[:, 1]

    @property
    def fitted(self) -> bool:
        return self._fitted


class TemperatureScaler:
    """
    Temperature scaling: calibrated_prob = sigmoid(logit(raw_prob) / T).

    T > 1 compresses overconfident probabilities toward 0.5.
    T is optimized to minimize Brier score on the calibration set.
    """

    def __init__(self) -> None:
        self._T = 1.0
        self._fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        from scipy.optimize import minimize_scalar
        from scipy.special import expit, logit
        from sklearn.metrics import brier_score_loss

        clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = logit(clipped)

        def loss(T: float) -> float:
            return float(brier_score_loss(labels, expit(logits / T)))

        result = minimize_scalar(loss, bounds=(1.0, 20.0), method="bounded")
        self._T = float(result.x)
        self._fitted = True

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probs
        from scipy.special import expit, logit

        logits = logit(np.clip(probs, 1e-7, 1 - 1e-7))
        return expit(logits / self._T)

    @property
    def temperature(self) -> float:
        return self._T

    @property
    def fitted(self) -> bool:
        return self._fitted
