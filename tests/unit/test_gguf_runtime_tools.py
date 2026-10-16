"""The native bridge receipt binds current sources and actual executable content."""

import json
import subprocess
from pathlib import Path

import pytest

from src.utils import gguf_runtime_tools as runtime


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    sources = tmp_path / "native"
    sources.mkdir()
    for name in ("CMakeLists.txt", "main.cpp"):
        (sources / name).write_text(f"{name} original source")
    monkeypatch.setattr(runtime, "BRIDGE_SOURCE_DIR", sources)
    binary = tmp_path / "tunesmith-gguf-eval"
    binary.write_bytes(b"original static bridge")
    binary.chmod(0o755)
    state = {
        "calls": [],
        "version": {"status": "ready", "protocol": runtime.PROTOCOL, "revision": runtime.REVISION},
        "returncode": 0,
    }

    def run(command, **kwargs):
        assert command == [str(binary.resolve()), "--version"]
        assert kwargs == {"capture_output": True, "text": True, "check": False, "timeout": 10}
        state["calls"].append(command)
        return subprocess.CompletedProcess(
            command, state["returncode"], json.dumps(state["version"]), ""
        )

    monkeypatch.setattr(runtime.subprocess, "run", run)
    runtime.write_evaluation_runtime_manifest(
        binary,
        bridge_sources=runtime.bridge_source_identity(),
        build_flags=dict(runtime.STATIC_CPU_FLAGS),
    )
    state["calls"].clear()
    return binary, sources, state


def test_complete_receipt_accepts_current_bridge_without_loading_model(bridge):
    binary, _, state = bridge
    identity = runtime.evaluation_runtime_identity(binary)
    assert identity["binary_path"] == str(binary.resolve())
    assert identity["revision"] == runtime.REVISION
    assert identity["protocol"] == runtime.PROTOCOL
    assert identity["linkage"] == "static_llama_cpp_cpu"
    assert identity["bridge_sources"] == runtime.bridge_source_identity()
    assert len(identity["sha256"]) == 64
    assert state["calls"] == [[str(binary.resolve()), "--version"]]


@pytest.mark.parametrize("changed", ["binary", "main.cpp", "CMakeLists.txt", "receipt", "symlink"])
def test_changed_or_unbound_bridge_refused_before_executing_it(bridge, changed):
    binary, sources, state = bridge
    sidecar = Path(str(binary) + ".json")
    if changed == "binary":
        binary.write_bytes(b"replaced executable")
    elif changed in {"main.cpp", "CMakeLists.txt"}:
        (sources / changed).write_text("source changed after build")
    elif changed == "receipt":
        sidecar.unlink()
    else:
        original = binary.with_name("original")
        binary.rename(original)
        binary.symlink_to(original)
    with pytest.raises(ValueError):
        runtime.evaluation_runtime_identity(binary)
    assert state["calls"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "a" * 40),
        ("protocol", "old_protocol"),
        ("linkage", "shared"),
        ("build_flags", {"BUILD_SHARED_LIBS": "ON"}),
        ("bridge_sources", {}),
    ],
)
def test_incomplete_or_dynamic_build_receipt_cannot_claim_static_fixed_runtime(
    bridge, field, value
):
    binary, _, state = bridge
    sidecar = Path(str(binary) + ".json")
    manifest = json.loads(sidecar.read_text())
    manifest[field] = value
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="不匹配"):
        runtime.evaluation_runtime_identity(binary)
    assert not state["calls"]


@pytest.mark.parametrize("failure", ["exit", "protocol", "revision", "status"])
def test_version_self_check_must_match_even_with_matching_content_receipt(bridge, failure):
    binary, _, state = bridge
    if failure == "exit":
        state["returncode"] = 9
    else:
        state["version"][failure] = "wrong"
    with pytest.raises(ValueError, match="协议或固定版本"):
        runtime.evaluation_runtime_identity(binary)


def test_source_changes_during_build_cannot_be_attested(bridge):
    binary, sources, _ = bridge
    expected = runtime.bridge_source_identity()
    (sources / "main.cpp").write_text("new source while compiler was running")
    with pytest.raises(ValueError, match="源码或静态 CPU"):
        runtime.write_evaluation_runtime_manifest(
            binary, bridge_sources=expected, build_flags=dict(runtime.STATIC_CPU_FLAGS)
        )


@pytest.mark.parametrize("operation", ["identity", "write_receipt"])
@pytest.mark.parametrize("changed", ["binary", "source"])
def test_change_during_self_check_never_accepts_or_replaces_receipt(
    bridge, monkeypatch, operation, changed
):
    binary, sources, _ = bridge
    expected = runtime.bridge_source_identity()
    sidecar = Path(str(binary) + ".json")
    original_receipt = sidecar.read_bytes()

    def mutate(_):
        if changed == "binary":
            binary.write_bytes(b"binary changed during version check")
        else:
            (sources / "main.cpp").write_text("source changed during version check")

    monkeypatch.setattr(runtime, "_version", mutate)
    with pytest.raises(ValueError, match="期间发生变化"):
        if operation == "identity":
            runtime.evaluation_runtime_identity(binary)
        else:
            runtime.write_evaluation_runtime_manifest(
                binary, bridge_sources=expected, build_flags=dict(runtime.STATIC_CPU_FLAGS)
            )
    assert sidecar.read_bytes() == original_receipt


def test_oversized_receipt_is_refused_before_execution(bridge):
    binary, _, state = bridge
    Path(str(binary) + ".json").write_text(" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="小型文件"):
        runtime.evaluation_runtime_identity(binary)
    assert not state["calls"]
