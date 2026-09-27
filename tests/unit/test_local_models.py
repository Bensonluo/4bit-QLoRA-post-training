"""Discovery checks local file completeness without loading or hashing model weights."""

import json
from pathlib import Path

from src.workbench.local_models import discover_local_models


def model(path, *, sharded=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"fixture"}')
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    if sharded:
        (path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "first": "model-00001-of-00002.safetensors",
                        "other": "model-00002-of-00002.safetensors",
                    },
                }
            )
        )
        (path / "model-00001-of-00002.safetensors").write_bytes(b"part-one")
        (path / "model-00002-of-00002.safetensors").write_bytes(b"part-two")
    else:
        (path / "model.safetensors").write_bytes(b"fixture-weight")
    return path


def test_hf_snapshot_and_namespace_cache_are_discovered_without_weight_reads(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    first = model(hub / "models--org--model" / "snapshots" / "revision", sharded=True)
    second = model(hub / "namespace" / "model")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "MODELSCOPE_CACHE"):
        monkeypatch.delenv(key, raising=False)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in {".safetensors", ".bin"}, "discovery must not read model weights"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    rows = discover_local_models()
    assert {row["model_path"] for row in rows} == {str(first), str(second)}
    assert {row["name"] for row in rows} == {"org/model", "namespace/model"}
    assert all(row["status"] == "available" and row["weight_bytes"] > 0 for row in rows)
    assert discover_local_models() == rows


def test_hf_cached_blob_symlinks_are_accepted_but_external_link_is_not_read(tmp_path, monkeypatch):
    repository = tmp_path / "hub" / "models--org--model"
    snapshot = model(repository / "snapshots" / "revision")
    blobs = repository / "blobs"
    blobs.mkdir()
    for path in list(snapshot.iterdir()):
        blob = blobs / path.name.replace(".", "-")
        path.replace(blob)
        path.symlink_to(Path("../../blobs") / blob.name)
    assert discover_local_models([snapshot])[0]["status"] == "available"
    secret = tmp_path / "secret.json"
    secret.write_text("SHOULD NEVER BE READ")
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(secret)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path.resolve() != secret
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    result = discover_local_models([snapshot])[0]
    assert result["status"] == "incomplete"
    assert any("越出" in issue for issue in result["issues"])
    assert "SHOULD NEVER BE READ" not in json.dumps(result)


def test_missing_shard_and_traversal_index_are_incomplete(tmp_path):
    path = model(tmp_path / "model", sharded=True)
    (path / "model-00002-of-00002.safetensors").unlink()
    row = discover_local_models([path])[0]
    assert row["status"] == "incomplete"
    assert any("00002" in issue for issue in row["issues"])
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "../outside.bin"}})
    )
    row = discover_local_models([path])[0]
    assert row["status"] == "incomplete"
    assert row["weight_bytes"] == 0
    assert any("目录内" in issue for issue in row["issues"])


def test_orphan_shards_tokenizer_config_only_and_invalid_config_are_not_available(tmp_path):
    path = model(tmp_path / "model")
    (path / "model.safetensors").rename(path / "model-00001-of-00002.safetensors")
    (path / "tokenizer.json").unlink()
    (path / "config.json").write_text("invalid")
    row = discover_local_models([path])[0]
    assert row["status"] == "incomplete"
    assert any("JSON" in issue for issue in row["issues"])
    assert any("tokenizer 数据" in issue for issue in row["issues"])
    assert any("孤立分片" in issue for issue in row["issues"])


def test_explicit_roots_are_bounded_deduplicated_and_do_not_follow_external_directories(tmp_path):
    root = tmp_path / "models"
    candidate = model(root / "local")
    model(root / "too" / "deep" / "model")
    external = model(tmp_path / "outside")
    (root / "external-link").symlink_to(external, target_is_directory=True)
    alias = tmp_path / "alias"
    alias.symlink_to(candidate, target_is_directory=True)
    rows = discover_local_models([root, candidate, alias])
    assert len(rows) == 1
    assert rows[0]["model_path"] == str(candidate)
    assert rows[0]["status"] == "available"
    assert discover_local_models([]) == []


def test_hf_home_and_cwd_models_are_known_default_locations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "MODELSCOPE_CACHE"):
        monkeypatch.delenv(key, raising=False)
    first = model(tmp_path / "hf" / "hub" / "models--org--one" / "snapshots" / "rev")
    second = model(tmp_path / "models" / "business")
    assert {row["model_path"] for row in discover_local_models()} == {str(first), str(second)}


def test_zero_length_or_broken_symlink_shard_is_not_available(tmp_path):
    path = model(tmp_path / "model", sharded=True)
    shard = path / "model-00002-of-00002.safetensors"
    shard.write_bytes(b"")
    assert discover_local_models(path)[0]["status"] == "incomplete"
    shard.unlink()
    shard.symlink_to(path / "missing")
    assert discover_local_models(path)[0]["status"] == "incomplete"
