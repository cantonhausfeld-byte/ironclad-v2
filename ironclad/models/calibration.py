"""Calibration wrappers for probability outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


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
