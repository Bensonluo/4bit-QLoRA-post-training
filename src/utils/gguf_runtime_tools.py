"""Bind the local CPU evaluation bridge to the current native implementation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

REVISION = "436f6f89e1e581249900b37a5b8a12a36a6d0912"
PROTOCOL = "tunesmith_gguf_eval_v1"
BRIDGE_SOURCE_DIR = Path(__file__).resolve().parents[2] / "native" / "gguf_eval"
STATIC_CPU_FLAGS = {
    "BUILD_SHARED_LIBS": "OFF",
    "GGML_CPU": "ON",
    **{
        f"GGML_{name}": "OFF"
        for name in (
            "BACKEND_DL",
            "METAL",
            "CUDA",
            "HIP",
            "MUSA",
            "VULKAN",
            "SYCL",
            "OPENCL",
            "CANN",
            "RPC",
            "WEBGPU",
            "ZDNN",
            "VIRTGPU",
            "VIRTGPU_BACKEND",
            "OPENVINO",
            "ET",
            "ET_SYSEMU",
            "HEXAGON",
            "ZENDNN",
            "OPENMP_FETCH",
        )
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bridge_source_identity() -> dict[str, str]:
    """Hash both the bridge implementation and its enforced static CPU build rules."""
    return {
        f"native/gguf_eval/{name}": _sha256(BRIDGE_SOURCE_DIR / name)
        for name in ("CMakeLists.txt", "main.cpp")
    }


def _version(binary: Path) -> None:
    try:
        result = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"GGUF 评测桥无法自检，请重新准备本地工具：{exc}") from exc
    try:
        value = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("GGUF 评测桥未返回有效版本 JSON，请重新准备工具。") from exc
    if result.returncode != 0 or value != {
        "status": "ready",
        "protocol": PROTOCOL,
        "revision": REVISION,
    }:
        raise ValueError("GGUF 评测桥协议或固定版本不匹配，请重新准备工具。")


def _binary_path(binary: Path) -> Path:
    binary = Path(binary).expanduser()
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("缺少可执行的独立 GGUF 评测桥，请重新运行工具准备命令。")
    return binary.resolve()


def write_evaluation_runtime_manifest(
    binary: Path, *, bridge_sources: dict[str, str], build_flags: dict[str, str]
) -> None:
    """Attest only after setup checked the actual static CPU CMake cache and built it."""
    binary = _binary_path(binary)
    if build_flags != STATIC_CPU_FLAGS or bridge_source_identity() != bridge_sources:
        raise ValueError("GGUF 评测桥源码或静态 CPU 构建设置已变化，未写入准备证据。")
    digest = _sha256(binary)
    _version(binary)
    if digest != _sha256(binary) or bridge_source_identity() != bridge_sources:
        raise ValueError("GGUF 评测桥在自检期间发生变化，未写入准备证据。")
    manifest = {
        "format_version": 1,
        "revision": REVISION,
        "protocol": PROTOCOL,
        "linkage": "static_llama_cpp_cpu",
        "build_flags": build_flags,
        "bridge_sources": bridge_sources,
        "sha256": digest,
    }
    sidecar = binary.with_name(binary.name + ".json")
    if sidecar.is_symlink():
        raise ValueError("评测桥证据路径是符号链接，请保留现有内容并另选工具目录。")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".bridge-manifest-", dir=binary.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, sidecar)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluation_runtime_identity(binary: Path) -> dict[str, Any]:
    """Reject stale sources, changed binaries, missing receipts and incompatible bridges.

    llama/ggml are statically linked by the hashed CMake rules. The receipt binds
    the matching checked CPU-only cache, so no mutable llama/ggml shared library
    sits outside this binary identity. System C/C++/CPU framework libraries remain
    platform dependencies, as with the Python runtime itself.
    """
    binary = _binary_path(binary)
    sidecar = binary.with_name(binary.name + ".json")
    try:
        if sidecar.is_symlink() or sidecar.stat().st_size > 64 * 1024:
            raise ValueError("GGUF 评测桥证据不是有效的独立小型文件。")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        sources = bridge_source_identity()
        digest = _sha256(binary)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("GGUF 评测桥缺少有效源码/二进制准备证据，请重新准备工具。") from exc
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != expected
        for key, expected in {
            "format_version": 1,
            "revision": REVISION,
            "protocol": PROTOCOL,
            "linkage": "static_llama_cpp_cpu",
            "build_flags": STATIC_CPU_FLAGS,
            "bridge_sources": sources,
            "sha256": digest,
        }.items()
    ):
        raise ValueError("GGUF 评测桥与当前源码、二进制或固定 CPU 构建证据不匹配，请重新准备工具。")
    _version(binary)
    if digest != _sha256(binary) or bridge_source_identity() != sources:
        raise ValueError("GGUF 评测桥在身份核验期间发生变化，请重试。")
    return {**manifest, "binary_path": str(binary)}
