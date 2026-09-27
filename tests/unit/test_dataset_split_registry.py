"""Immutable data partitions remain reproducible, complete and separate from legacy data."""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.data_flywheel.schemas import DatasetItem, LineageRecord


@pytest.fixture()
def registry(tmp_path):
    return LocalDatasetRegistry(str(tmp_path / "registry"))


@pytest.fixture()
def splits():
    return {
        split: [
            {
                "instruction": "判断类别",
                "input": f"描述 {index}",
                "output": "质量",
                "source_refs": [{"source_digest": "source-sha", "row_id": f"r{index}"}],
            }
        ]
        for index, split in enumerate(("train", "validation", "test"))
    }


META = {"source_digest": "source-sha", "recipe_digest": "recipe-sha", "split_seed": 42}


def test_partition_round_trip_has_full_hashes_provenance_and_stable_version(registry, splits):
    version = registry.register_splits("售后", splits, metadata=META)
    assert len(version) == 64
    manifest = registry.get_split_manifest("售后", version)
    assert manifest["metadata"] == META
    before = {}
    for split, entry in manifest["splits"].items():
        path = registry.base_dir / "售后" / "splits" / version / entry["path"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["row_count"] == len(splits[split])
        assert registry.load_split("售后", version, split) == splits[split]
        before[path] = path.stat().st_mtime_ns
    reordered = {key: list(value) for key, value in reversed(list(splits.items()))}
    assert (
        registry.register_splits("售后", reordered, metadata=dict(reversed(list(META.items()))))
        == version
    )
    assert all(path.stat().st_mtime_ns == modified for path, modified in before.items())


def test_new_content_or_provenance_creates_version_without_mutating_old(registry, splits):
    original = copy.deepcopy(splits)
    old = registry.register_splits("data", splits, metadata=META)
    splits["train"][0]["output"] = "物流"
    new = registry.register_splits("data", splits, metadata=META)
    another = registry.register_splits("data", splits, metadata={**META, "recipe_digest": "new"})
    assert len({old, new, another}) == 3
    assert registry.load_split("data", old, "train") == original["train"]


@pytest.mark.parametrize(
    "change", ["missing", "extra", "empty", "missing_output", "nontext", "blank_output"]
)
def test_invalid_partition_contract_never_publishes(registry, splits, change):
    if change == "missing":
        del splits["test"]
    elif change == "extra":
        splits["dev"] = splits["validation"]
    elif change == "empty":
        splits["test"] = []
    elif change == "missing_output":
        del splits["test"][0]["output"]
    elif change == "nontext":
        splits["train"][0]["input"] = {"text": "unsupported"}
    else:
        splits["validation"][0]["output"] = " "
    with pytest.raises(ValueError):
        registry.register_splits("data", splits, metadata=META)
    assert not list(registry.base_dir.rglob("manifest.json"))


@pytest.mark.parametrize("name", ["../outside", "/tmp/escape", "a/b", "a\\b", "..", "", "x\ny"])
def test_dataset_name_cannot_escape_registry(registry, splits, name):
    with pytest.raises(ValueError):
        registry.register_splits(name, splits, metadata=META)


def test_symlink_escape_is_rejected(registry, splits, tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    (registry.base_dir / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        registry.register_splits("linked", splits, metadata=META)
    assert list(target.iterdir()) == []


def test_tampered_data_blocks_read_and_idempotent_registration(registry, splits):
    version = registry.register_splits("data", splits, metadata=META)
    path = registry.base_dir / "data" / "splits" / version / "test.jsonl"
    path.write_text('{"instruction":"changed","input":"","output":"wrong"}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.load_split("data", version, "train")
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.register_splits("data", splits, metadata=META)
    assert "wrong" in path.read_text()


def test_tampered_manifest_and_missing_partition_are_rejected(registry, splits):
    version = registry.register_splits("data", splits, metadata=META)
    directory = registry.base_dir / "data" / "splits" / version
    path = directory / "manifest.json"
    original = path.read_bytes()
    manifest = json.loads(original)
    manifest["metadata"]["recipe_digest"] = "tampered"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="immutable version"):
        registry.get_split_manifest("data", version)
    path.write_bytes(original)
    (directory / "validation.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        registry.get_split_manifest("data", version)


def test_failed_staging_write_does_not_publish_partial_version(registry, splits, monkeypatch):
    write = Path.write_bytes

    def fail_midway(path, content):
        if path.name == "validation.jsonl":
            raise OSError("simulated interrupted disk write")
        return write(path, content)

    monkeypatch.setattr(Path, "write_bytes", fail_midway)
    with pytest.raises(OSError, match="simulated"):
        registry.register_splits("data", splits, metadata=META)
    assert list((registry.base_dir / "data" / "splits").iterdir()) == []


def test_concurrent_publish_returns_one_complete_version(registry, splits, monkeypatch):
    barrier = Barrier(2)
    rename = Path.rename

    def together(path, target):
        barrier.wait(timeout=5)
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(registry.register_splits, "data", splits, metadata=META) for _ in range(2)
        ]
        versions = [future.result(timeout=10) for future in futures]
    assert versions[0] == versions[1]
    root = registry.base_dir / "data" / "splits"
    assert [path.name for path in root.iterdir()] == [versions[0]]
    assert registry.load_split("data", versions[0], "test") == splits["test"]


def test_legacy_registration_and_latest_are_independent(registry, splits):
    lineage = LineageRecord("legacy-v1", "import", "source", "")
    item = DatasetItem("row1", "旧数据", "旧答案")
    legacy = registry.register("data", [item], lineage)
    split_version = registry.register_splits("data", splits, metadata=META)
    assert registry.list_versions("data") == [legacy]
    assert registry.load("data")[0]["prompt"] == "旧数据"
    assert registry.get_lineage("data", legacy).lineage_id == "legacy-v1"
    assert registry.load_split("data", split_version, "train") == splits["train"]
