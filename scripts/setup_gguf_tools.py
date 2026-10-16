#!/usr/bin/env python3
"""Prepare pinned official GGUF conversion tools without modifying Python environments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

REVISION = "436f6f89e1e581249900b37a5b8a12a36a6d0912"
REPOSITORY = "https://github.com/ggml-org/llama.cpp"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep direct `python scripts/setup_gguf_tools.py` usable without an editable install.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def default_destination() -> Path:
    if sys.platform == "darwin":
        cache = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        cache = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache / "tunesmith" / "llama.cpp" / REVISION[:12]


def _run(arguments: list[str], phase: str, *, codes: tuple[int, ...] = (0,)) -> str:
    print(phase, file=sys.stderr, flush=True)
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise ValueError(
            f"找不到工具 {arguments[0]}；请安装 Git、CMake 和 C/C++ 编译器后重试。"
        ) from exc
    output = result.stdout + result.stderr
    if output:
        print(output.rstrip(), file=sys.stderr, flush=True)
    if result.returncode not in codes:
        raise ValueError(f"{phase}失败（退出码 {result.returncode}）：{output.strip()}")
    return output.strip()


def _verify_checkout(directory: Path) -> None:
    prefix = ["git", "-C", str(directory)]
    top = _run([*prefix, "rev-parse", "--show-toplevel"], "检查工具目录")
    if Path(top).resolve() != directory:
        raise ValueError("目标不是独立 llama.cpp checkout；不会覆盖外来目录。")
    origin = _run([*prefix, "remote", "get-url", "origin"], "检查官方来源")
    head = _run([*prefix, "rev-parse", "HEAD"], "检查固定版本")
    if origin not in {REPOSITORY, REPOSITORY + ".git"} or head != REVISION:
        raise ValueError("现有目录不是本次固定版本的官方 llama.cpp；不会切换版本或覆盖。")
    dirty = _run(
        [*prefix, "status", "--porcelain=v1", "--untracked-files=all"], "检查 checkout 完整性"
    )
    if dirty:
        raise ValueError("现有 checkout 已修改或含外来文件；请保留它并另选 --destination。")
    for name in (
        "convert_hf_to_gguf.py",
        "CMakeLists.txt",
        "conversion/__init__.py",
        "gguf-py/gguf/__init__.py",
    ):
        if not (directory / name).is_file():
            raise ValueError(f"官方 checkout 不完整，缺少 {name}；未宣称工具准备成功。")


def _quantizer(directory: Path) -> Path:
    name = "llama-quantize.exe" if sys.platform == "win32" else "llama-quantize"
    for candidate in (
        directory / "build" / "bin" / name,
        directory / "build" / "bin" / "Release" / name,
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise ValueError("没有可执行的 llama-quantize；请重新运行准备命令，不能仅凭目录存在宣称成功。")


def _evaluation_runtime(directory: Path, jobs: int, *, check_only: bool) -> dict[str, Any]:
    from src.utils.gguf_runtime_tools import (
        BRIDGE_SOURCE_DIR,
        STATIC_CPU_FLAGS,
        bridge_source_identity,
        evaluation_runtime_identity,
        write_evaluation_runtime_manifest,
    )

    build = directory / "build-tunesmith-eval"
    name = "tunesmith-gguf-eval.exe" if sys.platform == "win32" else "tunesmith-gguf-eval"
    binary = build / "bin" / name
    if any(
        path.is_symlink() for path in (build, binary.parent, binary, Path(str(binary) + ".json"))
    ):
        raise ValueError("评测桥构建/证据路径含符号链接，请另选独立工具目录。")
    if check_only:
        return evaluation_runtime_identity(binary)
    try:
        return evaluation_runtime_identity(binary)
    except ValueError:
        # A missing/stale receipt must cause a real rebuild, even if mtimes make
        # CMake believe old object files are current after a source restoration.
        pass
    sources = bridge_source_identity()
    _run(
        [
            "cmake",
            "-S",
            str(BRIDGE_SOURCE_DIR),
            "-B",
            str(build),
            f"-DLLAMA_CPP_DIR={directory}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        "配置固定 CPU 静态评测桥",
    )
    _run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--clean-first",
            "--target",
            "tunesmith-gguf-eval",
            "--parallel",
            str(jobs),
        ],
        "编译本地 JSON 行评测桥（无服务/无模型下载）",
    )
    cache = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    flags = {key: cache.get(key, "") for key in STATIC_CPU_FLAGS}
    if flags != STATIC_CPU_FLAGS or Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != (
        BRIDGE_SOURCE_DIR.resolve()
    ):
        raise ValueError("评测桥未使用当前源码的固定静态 CPU 构建配置，未宣称准备成功。")
    _verify_checkout(directory)
    write_evaluation_runtime_manifest(binary, bridge_sources=sources, build_flags=flags)
    return evaluation_runtime_identity(binary)


def prepare(destination: Path, jobs: int, *, check_only: bool = False) -> dict[str, Any]:
    if type(jobs) is not int or jobs < 1:
        raise ValueError("--jobs 必须是正整数。")
    destination = destination.expanduser()
    if destination.is_symlink():
        raise ValueError("目标是符号链接；请另选独立 --destination，不覆盖其他工具。")
    destination = destination.resolve()
    reused = destination.exists()
    if reused:
        if not destination.is_dir():
            raise ValueError("目标已经存在且不是目录；不会覆盖。")
        _verify_checkout(destination)
    else:
        if check_only:
            raise ValueError("工具目录尚不存在；--check-only 不会下载或创建目录。")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".tunesmith-llama-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            _run(["git", "init", str(stage)], "初始化临时工具 checkout")
            _run(["git", "-C", str(stage), "remote", "add", "origin", REPOSITORY], "绑定官方仓库")
            _run(
                ["git", "-C", str(stage), "fetch", "--depth", "1", "origin", REVISION],
                "下载固定版本官方工具（不下载模型）",
            )
            _run(["git", "-C", str(stage), "checkout", "--detach", REVISION], "检出固定版本")
            _verify_checkout(stage)
            # copytree refuses an existing destination; CMake runs only after
            # publication so its absolute source/build paths remain valid.
            shutil.copytree(stage, destination)
    converter = destination / "convert_hf_to_gguf.py"
    try:
        _run([sys.executable, "-B", str(converter), "--help"], "检查当前 Python 的转换依赖")
    except ValueError as exc:
        requirements = destination / "requirements" / "requirements-convert_hf_to_gguf.txt"
        raise ValueError(
            f"当前 Python {sys.executable} 无法运行 GGUF 转换脚本。请使用具备转换依赖的 Python 重试；"
            f"依赖清单：{requirements}。本命令不会自动 pip 安装或修改项目环境。原始错误：{exc}"
        ) from exc
    if not check_only:
        _run(
            [
                "cmake",
                "-S",
                str(destination),
                "-B",
                str(destination / "build"),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_BUILD_SERVER=OFF",
                "-DLLAMA_BUILD_TOOLS=ON",
                "-DGGML_METAL=OFF",
                "-DGGML_CUDA=OFF",
            ],
            "配置 CPU 量化工具编译",
        )
        _run(
            [
                "cmake",
                "--build",
                str(destination / "build"),
                "--config",
                "Release",
                "--target",
                "llama-quantize",
                "--parallel",
                str(jobs),
            ],
            "编译 llama-quantize（不编译推理 CLI）",
        )
    binary = _quantizer(destination)
    # This pinned upstream program intentionally exits 1 after printing help.
    # Permit that documented behavior only when the actual usage is present.
    help_text = _run([str(binary), "--help"], "检查量化器可运行", codes=(0, 1))
    if "usage:" not in help_text.lower() or "Q4_K_M" not in help_text:
        raise ValueError("量化器没有返回预期帮助内容；未宣称工具准备成功。")
    runtime = _evaluation_runtime(destination, jobs, check_only=check_only)
    _verify_checkout(destination)
    options = ["--format", "gguf", "--llama-cpp-dir", str(destination)]
    return {
        "status": "ready",
        "revision": REVISION,
        "repository": REPOSITORY,
        "llama_cpp_dir": str(destination),
        "quantizer": str(binary),
        "evaluation_runtime_path": runtime["binary_path"],
        "gguf_runtime_path": runtime["binary_path"],
        "evaluation_runtime_identity": runtime,
        "python_executable": sys.executable,
        "reused_checkout": reused,
        "check_only": check_only,
        "train_export_args": options,
        "train_export_options": shlex.join(options),
        "scope_note": "仅准备并自测官方转换/量化工具与固定 CPU 评测桥；未下载模型、未训练或评估模型。",
    }


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=default_destination(), help="独立工具缓存目录"
    )
    parser.add_argument(
        "--jobs", type=_positive_integer, default=min(os.cpu_count() or 1, 4), help="编译并行数"
    )
    parser.add_argument("--check-only", action="store_true", help="只检查已有工具，不下载或编译")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.destination, args.jobs, check_only=args.check_only)
    except (ValueError, OSError) as exc:
        print(f"GGUF 工具准备失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
