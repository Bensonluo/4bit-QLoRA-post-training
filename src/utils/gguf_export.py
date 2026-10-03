"""Local, staged Hugging Face → GGUF conversion without implicit downloads."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

GGUF_QUANTIZATIONS = ("f16", "q8_0", "q4_k_m", "q5_k_m", "q6_k")
_FILE_TYPES = {"f16": 1, "q8_0": 7, "q4_k_m": 15, "q5_k_m": 17, "q6_k": 18}
_TENSOR_TYPES = {"f16": 1, "q8_0": 8, "q4_k_m": 12, "q5_k_m": 13, "q6_k": 14}


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "torch", "transformers", "sentencepiece", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def gguf_toolchain(
    directory: str | Path | None, quantization: str, *, verify_content: bool = False
) -> dict[str, Any]:
    """Inspect a complete checkout and executable; never install during a preflight."""
    if quantization not in GGUF_QUANTIZATIONS:
        raise ValueError(
            f"不支持的 GGUF 精度：{quantization}。可选：{', '.join(GGUF_QUANTIZATIONS)}"
        )
    reference = directory or os.environ.get("TUNESMITH_LLAMA_CPP")
    if not reference:
        raise ValueError(
            "尚未配置 GGUF 工具。运行 python scripts/setup_gguf_tools.py 准备本地工具，"
            "再用 --llama-cpp-dir 指定输出目录；不会下载模型或修改训练环境。"
        )
    root = Path(reference).expanduser().resolve()
    converter = root / "convert_hf_to_gguf.py"
    if not converter.is_file() or not (root / "gguf-py" / "gguf").is_dir():
        raise ValueError(
            f"{root} 不是完整的 llama.cpp 工具目录；需要 convert_hf_to_gguf.py 和 gguf-py。"
        )
    candidates = [
        root / folder / name
        for folder in ("build/bin", "build/bin/Release", "bin")
        for name in ("llama-quantize", "llama-quantize.exe")
    ]
    quantizer = next(
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)), None
    )
    if quantization != "f16" and quantizer is None:
        raise ValueError(
            "未找到可执行的 llama-quantize。请运行 python scripts/setup_gguf_tools.py "
            "准备工具，或在该目录执行 cmake --build build --target llama-quantize。"
        )
    inputs = {converter}
    for package in (root / "conversion", root / "gguf-py" / "gguf"):
        inputs.update(package.rglob("*.py"))
    if quantization != "f16" and quantizer is not None:
        inputs.add(quantizer)
        for suffix in ("*.dylib", "*.so*", "*.dll"):
            inputs.update((root / "build").rglob(suffix))
            inputs.update(path for path in quantizer.parent.glob(suffix) if path.is_file())
    files: dict[str, Any] = {}
    for path in sorted(inputs):
        stat = path.stat()
        entry: dict[str, Any] = {"size": stat.st_size}
        if verify_content:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
        else:
            entry["mtime_ns"] = stat.st_mtime_ns
        files[path.relative_to(root).as_posix()] = entry
    revision = None
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
    return {
        "directory": str(root),
        "converter": str(converter),
        "quantizer": str(quantizer) if quantization != "f16" and quantizer else None,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "python_packages": _package_versions(),
        "revision": revision,
        "files": files,
    }


def _run(command: list[str], phase: str) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        details = (
            ((exc.stderr or "") + "\n" + (exc.stdout or ""))[-3000:]
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise ValueError(f"GGUF {phase}失败，未发布模型。{details.strip()}") from exc
    if result.stderr:
        print(result.stderr[-3000:], file=sys.stderr)


def _source_templates(value: Any) -> dict[str, str]:
    """Use the actual loaded tokenizer contract, including Jinja-file precedence."""
    templates = {}
    if isinstance(value, str):
        templates["default"] = value
    elif isinstance(value, dict):
        templates.update({key: text for key, text in value.items() if isinstance(text, str)})
    elif isinstance(value, list):
        templates.update(
            {
                item["name"]: item["template"]
                for item in value
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and isinstance(item.get("template"), str)
            }
        )
    return templates


def verify_gguf_tokenizer(source: Path, metadata: dict[str, Any]) -> None:
    """Refuse a converter that silently drops the saved tokenizer/template contract."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    vocabulary_size = config.get("vocab_size", len(tokenizer))
    if metadata.get("token_count") != vocabulary_size or vocabulary_size < len(tokenizer):
        raise ValueError("GGUF 词表数量与源 tokenizer 不同，不能确认转换保留了输入契约。")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        expected = getattr(tokenizer, name, None)
        actual = metadata.get(name)
        unused_bos = (
            name == "bos_token_id" and expected is None and metadata.get("add_bos_token") is False
        )
        if actual is not None and expected != actual and not unused_bos:
            raise ValueError(f"GGUF {name} 与源 tokenizer 不同，未发布转换产物。")
        if expected is not None and actual is None:
            raise ValueError(f"GGUF 缺少源 tokenizer 的 {name} 标记，未发布转换产物。")
    for name in ("add_bos_token", "add_eos_token"):
        expected = getattr(tokenizer, name, None)
        actual = metadata.get(name)
        if type(expected) is bool and actual is not expected:
            raise ValueError(f"GGUF {name} 与源 tokenizer 不同或未声明，未发布转换产物。")
    templates = _source_templates(tokenizer.chat_template)
    actual_templates = metadata.get("chat_template") or {}
    if any(actual_templates.get(name) != text for name, text in templates.items()):
        raise ValueError("GGUF 聊天模板与已保存的源模板不同或被遗漏，未发布转换产物。")


def convert_to_gguf(
    model_path: str | Path,
    output_path: str | Path,
    *,
    quantization: str = "q4_k_m",
    llama_cpp_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convert, optionally quantize, verify, then publish a new file without overwrite."""
    from src.utils.gguf_metadata import read_gguf_metadata

    source = Path(model_path).expanduser().resolve()
    target = Path(output_path).expanduser().absolute()
    if not (source / "config.json").is_file() or not (source / "tokenizer_config.json").is_file():
        raise ValueError("GGUF 转换需要完整的本地 Hugging Face 模型与 tokenizer。")
    if target.exists() or target.is_symlink():
        raise ValueError(f"GGUF 输出已存在，不会覆盖：{target}")
    inventory = gguf_toolchain(llama_cpp_dir, quantization)
    identity = gguf_toolchain(llama_cpp_dir, quantization, verify_content=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".gguf-convert-", dir=target.parent) as temporary:
        intermediate = Path(temporary) / "model-f16.gguf"
        command = [
            identity["python"],
            "-B",
            "-X",
            f"pycache_prefix={Path(temporary) / 'pycache'}",
            identity["converter"],
            str(source),
            "--outfile",
            str(intermediate),
            "--outtype",
            "f16",
        ]
        commands = [command]
        _run(command, "格式转换")
        final = intermediate
        if quantization != "f16":
            final = Path(temporary) / f"model-{quantization}.gguf"
            command = [identity["quantizer"], str(intermediate), str(final), quantization.upper()]
            commands.append(command)
            _run(command, "量化")
        metadata = read_gguf_metadata(final)
        if metadata["file_type"] != _FILE_TYPES[quantization]:
            raise ValueError("GGUF 实际精度与所选精度不一致，未发布模型。")
        if _TENSOR_TYPES[quantization] not in metadata["tensor_types"]:
            raise ValueError("GGUF 权重没有包含所选精度的张量，不能把未量化模型声明为量化成功。")
        verify_gguf_tokenizer(source, metadata)
        if inventory != gguf_toolchain(llama_cpp_dir, quantization) or identity != gguf_toolchain(
            llama_cpp_dir, quantization, verify_content=True
        ):
            raise ValueError("转换期间 GGUF 工具发生变化，未发布模型。")
        # link() fails atomically if another process has claimed this output name.
        try:
            os.link(final, target)
        except OSError as exc:
            raise ValueError(f"GGUF 输出路径发生变化，未覆盖现有内容：{target}") from exc
    return {
        "gguf_file": str(target),
        "quantization": quantization,
        "metadata": metadata,
        "toolchain": identity,
        "commands": commands,
        "business_evaluation_status": "not_run",
    }
