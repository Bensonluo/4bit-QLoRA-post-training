"""Preflight stable local MLflow file stores before display-only browsing.

MLflow creates missing directories when opening its file store; other backends
may write even when asked only to search. This check never opens a database.
It cannot prevent another process from changing the store after the check.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname


class TrackingStoreUnavailable(ValueError):
    """The tracking store cannot safely be opened for display-only browsing."""


def _local_path(uri: str) -> Path:
    try:
        if not isinstance(uri, str) or not uri:
            raise ValueError
        if uri.startswith("file:"):
            parts = urlsplit(uri)
            if parts.scheme != "file" or parts.netloc not in ("", "localhost"):
                raise ValueError
            if parts.query or parts.fragment:
                raise ValueError
            path = Path(url2pathname(parts.path))
        else:
            path = Path(uri)
        if not path.is_absolute():
            raise ValueError
        return path
    except (OSError, ValueError, UnicodeError):
        raise TrackingStoreUnavailable(
            "展示环境无法安全读取此实验存储，请在本地工作台打开。"
        ) from None


def _directory_exists(path: Path) -> bool:
    """Distinguish an absent path from a broken symlink or inaccessible store."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        raise TrackingStoreUnavailable("展示环境无法读取实验存储，请在本地工作台打开。") from None
    try:
        if stat.S_ISDIR(path.stat().st_mode):
            return True
    except (OSError, ValueError):
        raise TrackingStoreUnavailable("展示环境无法读取实验存储，请在本地工作台打开。") from None
    raise TrackingStoreUnavailable("展示环境的实验存储不完整，请在本地工作台打开。")


def inspect_tracking_uri(
    uri: str, *, registry: bool = False
) -> Literal["missing", "ready", "missing_registry"]:
    """Check an existing file store without creating anything or opening SQLite.

    A missing root is an empty installation. Tracking uses ``.trash`` while
    registry uses ``models``; separate registries need not have ``.trash``.
    A missing models directory is reported without initializing it.
    """
    root = _local_path(uri)
    if not _directory_exists(root):
        return "missing"
    if registry:
        if not _directory_exists(root / "models"):
            return "missing_registry"
    elif not _directory_exists(root / ".trash"):
        raise TrackingStoreUnavailable("展示环境的实验存储不完整，请在本地工作台打开。")
    return "ready"
