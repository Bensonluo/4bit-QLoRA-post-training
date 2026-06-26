"""Data models for the data flywheel."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


@dataclass
class DatasetItem:
    """A single training/example item."""

    id: str
    prompt: str
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "synthetic"
    lineage_id: str = ""
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with ISO datetime."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetItem:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            response=data.get("response"),
            metadata=data.get("metadata", {}),
            source=data.get("source", "synthetic"),
            lineage_id=data.get("lineage_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class PreferencePair:
    """A preference pair for DPO/GRPO training."""

    id: str
    prompt: str
    chosen: str
    rejected: str
    generation_policy: str = ""
    judge_model: str | None = None
    reward_chosen: float = 0.0
    reward_rejected: float = 0.0
    lineage_id: str = ""
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with ISO datetime."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferencePair:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            generation_policy=data.get("generation_policy", ""),
            judge_model=data.get("judge_model"),
            reward_chosen=data.get("reward_chosen", 0.0),
            reward_rejected=data.get("reward_rejected", 0.0),
            lineage_id=data.get("lineage_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class LineageRecord:
    """Lineage metadata for a dataset or transformation."""

    lineage_id: str
    operation: str
    input_hash: str
    output_hash: str
    parent_lineage_ids: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    timestamp: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with ISO datetime."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageRecord:
        """Create from dictionary."""
        return cls(
            lineage_id=data["lineage_id"],
            operation=data["operation"],
            input_hash=data["input_hash"],
            output_hash=data["output_hash"],
            parent_lineage_ids=data.get("parent_lineage_ids", []),
            config=data.get("config", {}),
            run_id=data.get("run_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
