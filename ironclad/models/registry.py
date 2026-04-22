"""Model artifact registry: save, load, and version management."""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

from ironclad.config import MODELS_DIR
from ironclad.models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, base_dir: Path = MODELS_DIR) -> None:
        self.base_dir = base_dir

    def save(self, model: BaseModel, metrics: dict | None = None) -> Path:
        model_dir = self.base_dir / model.name / model.version
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        meta = {
            "name": model.name,
            "version": model.version,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "metrics": metrics or {},
        }
        (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved %s v%s to %s", model.name, model.version, model_dir)
        return model_path

    def load(self, name: str, version: str = "latest") -> BaseModel:
        model_dir = self._resolve_dir(name, version)
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No model artifact at {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded %s v%s", name, version)
        return model

    def metadata(self, name: str, version: str = "latest") -> dict:
        model_dir = self._resolve_dir(name, version)
        meta_path = model_dir / "metadata.json"
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text())

    def list_versions(self, name: str) -> list[str]:
        name_dir = self.base_dir / name
        if not name_dir.exists():
            return []
        return sorted(d.name for d in name_dir.iterdir() if d.is_dir())

    def _resolve_dir(self, name: str, version: str) -> Path:
        if version == "latest":
            versions = self.list_versions(name)
            if not versions:
                raise FileNotFoundError(f"No saved versions for model '{name}'")
            version = versions[-1]
        return self.base_dir / name / version
