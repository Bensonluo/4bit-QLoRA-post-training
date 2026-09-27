"""Bounded, metadata-only discovery of existing local model candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path

_MAX_DIRECTORIES = 4096
_MAX_JSON_BYTES = 8 * 1024 * 1024
_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
)


def _default_roots():
    home = Path.home()
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(home / ".cache"))).expanduser()
    paths = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        str(Path(os.environ["HF_HOME"]).expanduser() / "hub")
        if os.environ.get("HF_HOME")
        else None,
        str(cache / "huggingface" / "hub"),
        os.environ.get("MODELSCOPE_CACHE"),
        str(cache / "modelscope" / "hub"),
        str(cache / "modelscope" / "hub" / "models"),
        str(Path.cwd() / "models"),
    ]
    return [
        (Path(path).expanduser().resolve(), "cache" if index < 7 else "local")
        for index, path in enumerate(paths)
        if path
    ]


def _inside(path: Path, boundary: Path) -> bool:
    return path == boundary or boundary in path.parents


def _children(path: Path, boundary: Path, remaining: list[int]):
    try:
        entries = sorted(path.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    children = []
    for entry in entries:
        if remaining[0] <= 0:
            break
        remaining[0] -= 1
        if entry.name.startswith(".") or entry.name in {"blobs", "refs", "logs", "checkpoints"}:
            continue
        try:
            resolved = entry.resolve()
            if entry.is_dir() and _inside(resolved, boundary):
                children.append(entry)
        except (OSError, RuntimeError):
            continue
    return children


def _looks_like_model(path: Path) -> bool:
    return any(
        (path / name).exists() or (path / name).is_symlink()
        for name in (
            "config.json",
            "tokenizer_config.json",
            "model.safetensors",
            "pytorch_model.bin",
            *_INDEX_NAMES,
        )
    )


def _candidates(root: Path, source: str):
    remaining = [_MAX_DIRECTORIES]
    if _looks_like_model(root):
        yield root, source
    if root.name.startswith("models--"):
        for revision in _children(root / "snapshots", root, remaining):
            yield revision, "huggingface"
        return
    for child in _children(root, root, remaining):
        if child.name.startswith("models--"):
            for revision in _children(child / "snapshots", root, remaining):
                yield revision, "huggingface"
            continue
        if _looks_like_model(child):
            yield child, source
        for model in _children(child, root, remaining):
            if _looks_like_model(model):
                yield model, "modelscope" if source == "cache" else source


def _inspect(directory: Path, source: str) -> dict:
    path = directory.resolve()
    hf_repo = path.parent.parent if path.parent.name == "snapshots" else None
    if hf_repo is not None and not hf_repo.name.startswith("models--"):
        hf_repo = None
    allowed = [path]
    if hf_repo is not None:
        allowed.append(hf_repo / "blobs")
        source = "huggingface"
    issues, weight_paths = [], set()

    def local_file(name: str) -> Path | None:
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts:
            issues.append(f"{name}: 文件引用不在模型目录内。")
            return None
        item = path / candidate
        try:
            resolved = item.resolve()
            if not any(_inside(resolved, boundary) for boundary in allowed):
                issues.append(f"{name}: 链接越出模型目录或合法缓存 blobs，未读取。")
                return None
            if not item.is_file() or item.stat().st_size == 0:
                issues.append(f"{name}: 文件缺失、下载未完成或内容为空。")
                return None
            return resolved
        except (OSError, RuntimeError):
            issues.append(f"{name}: 本地文件无法读取。")
            return None

    def json_object(name):
        item = local_file(name)
        if item is None:
            return None
        try:
            if item.stat().st_size > _MAX_JSON_BYTES:
                raise ValueError("metadata too large")
            value = json.loads(item.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("metadata not an object")
            return value
        except (OSError, UnicodeError, ValueError):
            issues.append(f"{name}: 不是可读取的本地 JSON 配置。")
            return None

    json_object("config.json")
    json_object("tokenizer_config.json")
    tokenizer_present = [
        name for name in _TOKENIZER_FILES if (path / name).exists() or (path / name).is_symlink()
    ]
    if tokenizer_present:
        for name in tokenizer_present:
            local_file(name)
    elif (path / "vocab.json").exists() or (path / "merges.txt").exists():
        local_file("vocab.json")
        local_file("merges.txt")
    else:
        issues.append("缺少 tokenizer 数据文件；仅 tokenizer_config.json 不足以本地分词。")
    weight_found = False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (path / name).exists() or (path / name).is_symlink():
            weight_found = True
            item = local_file(name)
            if item is not None:
                weight_paths.add(item)
    for name in _INDEX_NAMES:
        if not (path / name).exists() and not (path / name).is_symlink():
            continue
        weight_found = True
        index = json_object(name)
        if index is None:
            continue
        mapping = index.get("weight_map")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(not isinstance(value, str) or not value for value in mapping.values())
        ):
            issues.append(f"{name}: 权重索引缺少有效 weight_map。")
            continue
        for shard in sorted(set(mapping.values())):
            if Path(shard).suffix not in {".safetensors", ".bin"}:
                issues.append(f"{name}: 权重分片类型不受支持。")
                continue
            item = local_file(shard)
            if item is not None:
                weight_paths.add(item)
    if not weight_found:
        issues.append("缺少完整模型权重或分片索引；适配器及孤立分片不能当作基座。")
    name = (
        "/".join(hf_repo.name.removeprefix("models--").split("--"))
        if hf_repo
        else (f"{path.parent.name}/{path.name}" if source == "modelscope" else path.name)
    )
    try:
        weight_bytes = sum(item.stat().st_size for item in weight_paths)
    except OSError:
        weight_bytes = 0
        issues.append("扫描时权重文件已变化，请重新扫描。")
    return {
        "name": name,
        "model_path": str(path),
        "source": source,
        "weight_bytes": weight_bytes,
        "status": "incomplete" if issues else "available",
        "issues": list(dict.fromkeys(issues)),
    }


def discover_local_models(roots=None) -> list[dict]:
    """Scan known caches and cwd/models, or only explicit roots, at bounded depth.

    `available` means local files are complete candidates, not that architecture,
    tokenizer execution, device compatibility or memory requirements were verified.
    Weight files are only stat'ed; their contents are not loaded or hashed.
    """
    if roots is None:
        configured = _default_roots()
    else:
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if not isinstance(roots, (list, tuple)) or any(
            not isinstance(root, (str, Path)) for root in roots
        ):
            raise ValueError("模型扫描根目录必须是本地路径列表。")
        configured = [(Path(root).expanduser().resolve(), "explicit") for root in roots]
    found, scanned = {}, set()
    for root, source in configured:
        if root in scanned or not root.is_dir():
            continue
        scanned.add(root)
        for directory, kind in _candidates(root, source):
            try:
                actual = directory.resolve()
                if actual not in found:
                    found[actual] = _inspect(directory, kind)
            except (OSError, RuntimeError):
                continue
    return sorted(found.values(), key=lambda row: (row["name"].casefold(), row["model_path"]))
