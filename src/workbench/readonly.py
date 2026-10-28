"""Explicit read-only SQLite connections for browsing persisted workbench data."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast


class ReadOnlyWalUnsupported(sqlite3.OperationalError):
    """The SQLite WAL mode may write shared-memory files even for readers."""


def open_database(path: Path, *, read_only: bool, **kwargs: Any) -> sqlite3.Connection:
    """Open existing data in SQLite read-only mode without creating a store."""
    if read_only:
        if path.is_file():
            with path.open("rb") as stream:
                header = stream.read(20)
            wal_header = header[:16] == b"SQLite format 3\x00" and 2 in header[18:20]
            wal_sidecars = any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm"))
            if wal_header or wal_sidecars:
                # A WAL reader can create or change the database's -shm file.
                # immutable=1 avoids that write but silently ignores WAL commits.
                raise ReadOnlyWalUnsupported("展示模式暂不能无写入地读取 WAL 数据库或残留日志。")
        return cast(
            sqlite3.Connection,
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, **kwargs),
        )
    return cast(sqlite3.Connection, sqlite3.connect(path, **kwargs))


def require_writable(read_only: bool) -> None:
    if read_only:
        raise PermissionError("展示模式只允许浏览，不能修改工作台数据。")
