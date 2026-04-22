"""Load static stadium metadata from CSV."""
from __future__ import annotations

import logging

import pandas as pd

from ironclad.config import STATIC_DIR
from ironclad.ingest.base import BaseIngestor
from ironclad.store.writer import BronzeWriter

logger = logging.getLogger(__name__)


class StadiumIngestor(BaseIngestor):
    def __init__(self, writer: BronzeWriter | None = None) -> None:
        self._writer = writer or BronzeWriter()

    def _ingest(self) -> int:
        path = STATIC_DIR / "stadiums.csv"
        if not path.exists():
            logger.warning("stadiums.csv not found at %s", path)
            return 0
        df = pd.read_csv(path)
        df["is_dome"] = df["is_dome"].astype(bool)
        df["altitude_ft"] = df["altitude_ft"].fillna(0).astype(int)
        n = self._writer.write_stadiums(df)
        logger.info("Wrote %d stadium rows", n)
        return n
