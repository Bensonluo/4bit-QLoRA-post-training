"""Dataset registry with lineage tracking."""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.data_flywheel.schemas import DatasetItem, LineageRecord, PreferencePair


def _compute_hash(items: list[dict[str, Any]]) -> str:
    """Compute a deterministic hash of a list of items."""
    content = json.dumps(items, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class DatasetRegistry(ABC):
    """Abstract dataset registry."""

    @abstractmethod
    def register(
        self,
        name: str,
        items: list[DatasetItem] | list[PreferencePair],
        lineage: LineageRecord,
    ) -> str:
        """Register a dataset and return its version id."""
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        name: str,
        version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load a dataset version as raw dicts."""
        raise NotImplementedError

    @abstractmethod
    def get_lineage(self, name: str, version: str) -> LineageRecord:
        """Get lineage record for a dataset version."""
        raise NotImplementedError


class LocalDatasetRegistry(DatasetRegistry):
    """Local filesystem-backed dataset registry."""

    def __init__(self, base_dir: str = "./data/registry") -> None:
        """Initialize local registry."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, name: str) -> Path:
        """Get dataset directory."""
        return self.base_dir / name

    def register(
        self,
        name: str,
        items: list[DatasetItem] | list[PreferencePair],
        lineage: LineageRecord,
    ) -> str:
        """Register dataset locally."""
        ds_dir = self._dataset_dir(name)
        ds_dir.mkdir(parents=True, exist_ok=True)

        version = lineage.lineage_id
        data_path = ds_dir / f"{version}.jsonl"
        manifest_path = ds_dir / "manifest.json"

        records = [item.to_dict() for item in items]
        with open(data_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        output_hash = _compute_hash(records)
        lineage.output_hash = output_hash

        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        manifest[version] = {
            "lineage": lineage.to_dict(),
            "path": str(data_path.relative_to(self.base_dir)),
            "num_items": len(items),
            "output_hash": output_hash,
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return version

    def load(
        self,
        name: str,
        version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load dataset version."""
        ds_dir = self._dataset_dir(name)
        manifest_path = ds_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset not found: {name}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if version is None:
            version = list(manifest.keys())[-1]

        if version not in manifest:
            raise ValueError(f"Version {version} not found for dataset {name}")

        data_path = self.base_dir / manifest[version]["path"]
        records: list[dict[str, Any]] = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_lineage(self, name: str, version: str) -> LineageRecord:
        """Get lineage record."""
        ds_dir = self._dataset_dir(name)
        manifest_path = ds_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if version not in manifest:
            raise ValueError(f"Version {version} not found for dataset {name}")

        return LineageRecord.from_dict(manifest[version]["lineage"])

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a dataset."""
        manifest_path = self._dataset_dir(name) / "manifest.json"
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return list(manifest.keys())


def new_lineage_id() -> str:
    """Generate a new lineage/version id."""
    return f"v_{uuid.uuid4().hex[:12]}"
