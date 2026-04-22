"""Base ingestor with retry logic."""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    MAX_RETRIES = 4
    RETRY_DELAYS = [2, 4, 8, 16]

    def ingest(self, *args, **kwargs):
        for attempt, delay in enumerate(self.RETRY_DELAYS, 1):
            try:
                return self._ingest(*args, **kwargs)
            except Exception as exc:
                if attempt == self.MAX_RETRIES:
                    raise
                logger.warning("Attempt %d failed (%s); retrying in %ds", attempt, exc, delay)
                time.sleep(delay)

    @abstractmethod
    def _ingest(self, *args, **kwargs):
        ...
