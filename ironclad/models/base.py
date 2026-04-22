"""Base model interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseModel(ABC):
    name: str = "base"
    version: str = "0.0.0"

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> dict:
        ...

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        raise NotImplementedError("fit() not implemented for this model")

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.DataFrame) -> None:
        pass

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    def load(self, directory: Path) -> None:
        raise NotImplementedError
