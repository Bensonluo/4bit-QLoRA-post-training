"""Dataset registry with lineage tracking."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.data_flywheel.schemas import DatasetItem, LineageRecord, PreferencePair

_SPLITS = ("train", "validation", "test")


def _canonical_bytes(value: Any) -> bytes:
    """Canonical JSON makes a split artifact independent of dictionary insertion order."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _validate_split_records(records: Any, split: str) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError(f"Split {split} must be a nonempty list of records")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or any(
            not isinstance(record.get(field), str) for field in ("instruction", "input", "output")
        ):
            raise ValueError(
                f"Split {split} row {index + 1} requires instruction/input/output strings"
            )
        if not record["instruction"].strip() or not record["output"].strip():
            raise ValueError(
                f"Split {split} row {index + 1} requires nonempty instruction and output"
            )


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

    def _split_root(self, name: str) -> Path:
        """Keep new split artifacts in a separate namespace from legacy versions."""
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or any(ord(char) < 32 for char in name)
        ):
            raise ValueError("Dataset name must be a single nonempty path component")
        root = self.base_dir / name / "splits"
        try:
            root.resolve().relative_to(self.base_dir.resolve())
        except ValueError:
            raise ValueError("Dataset path escapes the registry") from None
        if (self.base_dir / name).is_symlink() or root.is_symlink():
            raise ValueError("Dataset split directories must not be symbolic links")
        return root

    def register_splits(
        self,
        name: str,
        splits: dict[str, list[dict[str, Any]]],
        *,
        metadata: dict[str, Any],
    ) -> str:
        """Publish immutable train/validation/test files with caller-supplied provenance.

        Metadata should carry source/recipe digests and processing lineage. This
        layer preserves those facts and row references without interpreting their
        business meaning or choosing the partition algorithm.
        """
        root = self._split_root(name)
        if not isinstance(splits, dict) or set(splits) != set(_SPLITS):
            raise ValueError("Exactly train, validation and test splits are required")
        if not isinstance(metadata, dict):
            raise ValueError("Split metadata must be a JSON object")
        payloads: dict[str, bytes] = {}
        entries: dict[str, dict[str, Any]] = {}
        try:
            for split in _SPLITS:
                records = splits[split]
                _validate_split_records(records, split)
                payload = b"".join(_canonical_bytes(record) + b"\n" for record in records)
                payloads[split] = payload
                entries[split] = {
                    "path": f"{split}.jsonl",
                    "row_count": len(records),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            identity = {"format_version": 1, "splits": entries, "metadata": metadata}
            version = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
            manifest_bytes = _canonical_bytes({**identity, "version": version}) + b"\n"
        except (TypeError, OverflowError) as exc:
            raise ValueError(
                "Split records and metadata must contain JSON-serializable values"
            ) from exc
        root.mkdir(parents=True, exist_ok=True)
        destination = root / version
        if destination.exists():
            self.get_split_manifest(name, version)
            return version
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
        try:
            for split, payload in payloads.items():
                (staging / f"{split}.jsonl").write_bytes(payload)
            (staging / "manifest.json").write_bytes(manifest_bytes)
            try:
                staging.rename(destination)
            except OSError:
                # Another writer may publish the same complete content first.
                # Never overwrite it or accept an incomplete/corrupt winner.
                if not destination.is_dir():
                    raise
                self.get_split_manifest(name, version)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return version

    def _read_split_package(
        self, name: str, version: str
    ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
        if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
            raise ValueError("Split version must be its complete SHA256 identity")
        directory = self._split_root(name) / version
        if directory.is_symlink():
            raise ValueError("Dataset split version must not be a symbolic link")
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Dataset split manifest must not be a symbolic link")
        try:
            manifest = json.loads(manifest_path.read_bytes())
            if (
                not isinstance(manifest, dict)
                or set(manifest) != {"format_version", "version", "splits", "metadata"}
                or manifest["format_version"] != 1
                or manifest["version"] != version
                or not isinstance(manifest["metadata"], dict)
                or not isinstance(manifest["splits"], dict)
                or set(manifest["splits"]) != set(_SPLITS)
            ):
                raise ValueError("Invalid split manifest")
            identity = {key: value for key, value in manifest.items() if key != "version"}
            if hashlib.sha256(_canonical_bytes(identity)).hexdigest() != version:
                raise ValueError("Split manifest content does not match its immutable version")
            loaded: dict[str, list[dict[str, Any]]] = {}
            for split in _SPLITS:
                entry = manifest["splits"][split]
                if (
                    not isinstance(entry, dict)
                    or set(entry) != {"path", "row_count", "sha256"}
                    or entry["path"] != f"{split}.jsonl"
                    or type(entry["row_count"]) is not int
                    or entry["row_count"] < 1
                ):
                    raise ValueError(f"Invalid {split} manifest entry")
                path = directory / entry["path"]
                if path.is_symlink():
                    raise ValueError("Dataset split files must not be symbolic links")
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Split {split} content hash mismatch; artifact was modified")
                records = [json.loads(line) for line in payload.splitlines()]
                _validate_split_records(records, split)
                if len(records) != entry["row_count"]:
                    raise ValueError(f"Split {split} row count mismatch")
                loaded[split] = records
        except (TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid split artifact") from exc
        return manifest, loaded

    def get_split_manifest(self, name: str, version: str) -> dict[str, Any]:
        """Read a split manifest after verifying all files and their full hashes."""
        manifest, _ = self._read_split_package(name, version)
        return manifest

    def load_split(self, name: str, version: str, split: str) -> list[dict[str, Any]]:
        """Load one partition only after the immutable package passes verification."""
        if split not in _SPLITS:
            raise ValueError("Split must be train, validation or test")
        _, records = self._read_split_package(name, version)
        return records[split]


def new_lineage_id() -> str:
    """Generate a new lineage/version id."""
    return f"v_{uuid.uuid4().hex[:12]}"
