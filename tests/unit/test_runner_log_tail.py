"""Recent log reads stay bounded without losing or mis-decoding lines."""

import builtins
from pathlib import Path

import pytest

from src.tracking.runner import TrainingRunner


@pytest.mark.parametrize(
    ("contents", "tail", "expected"),
    [
        ("第一行\n第二行\n第三行\n", 2, "第二行\n第三行\n"),
        ("第一行\n第二行\n第三行", 2, "第二行\n第三行"),
        ("first\r\nsecond\r\nthird\r\n", 2, "second\nthird\n"),
        ("first\rsecond\rthird", 2, "second\nthird"),
        ("first\n\n", 1, "\n"),
        ("one line", 3, "one line"),
        ("", 2, ""),
        ("first\nsecond\n", 0, "first\nsecond\n"),
        ("first\nsecond\nthird\n", -1, "second\nthird\n"),
    ],
)
def test_tail_preserves_previous_text_semantics(tmp_path, contents, tail, expected):
    log = tmp_path / "train.log"
    log.write_bytes(contents.encode("utf-8"))
    runner = TrainingRunner(project_root=str(tmp_path))
    runner._run_meta["r"] = {"log_path": str(log)}
    assert runner.read_recent_logs("r", tail=tail) == expected


@pytest.mark.parametrize("separator", ["\n", "\r", "\r\n"])
def test_tail_reads_near_end_of_large_log_only(tmp_path, monkeypatch, separator):
    log = tmp_path / "train.log"
    log.write_bytes(
        ("日志内容" * 120 + separator).encode() * 3000 + ("最后一行" + separator).encode()
    )
    runner = TrainingRunner(project_root=str(tmp_path))
    runner._run_meta["r"] = {"log_path": str(log)}
    original_open = Path.open
    original_builtin_open = builtins.open
    read_sizes = []

    class CountingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def seek(self, *args):
            return self.stream.seek(*args)

        def tell(self):
            return self.stream.tell()

        def read(self, size=-1):
            data = self.stream.read(size)
            read_sizes.append(len(data))
            return data

        def readlines(self):
            lines = self.stream.readlines()
            read_sizes.append(sum(len(line.encode("utf-8")) for line in lines))
            return lines

    def count_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return CountingReader(stream) if path == log and mode == "rb" else stream

    def count_builtin(path, mode="r", *args, **kwargs):
        stream = original_builtin_open(path, mode, *args, **kwargs)
        return CountingReader(stream) if Path(path) == log else stream

    monkeypatch.setattr(Path, "open", count_open)
    monkeypatch.setattr(builtins, "open", count_builtin)
    result = runner.read_recent_logs("r", tail=2)
    assert result == ("日志内容" * 120 + "\n最后一行\n")
    assert read_sizes
    assert sum(read_sizes) < 32_768
    assert log.stat().st_size > 4_000_000


def test_tail_reads_complete_long_utf8_line_across_blocks(tmp_path):
    log = tmp_path / "train.log"
    content = "早期\n" + "长" * 20_000 + "\n结束\n"
    log.write_bytes(content.encode("utf-8"))
    runner = TrainingRunner(project_root=str(tmp_path))
    runner._run_meta["r"] = {"log_path": str(log)}
    assert runner.read_recent_logs("r", tail=2) == "长" * 20_000 + "\n结束\n"


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_crlf_crossing_block_boundaries_matches_text_readlines(tmp_path, offset):
    log = tmp_path / "train.log"
    content = "起始\r\n" + "x" * (8192 + offset) + "\r\n" + "结束\r\n" * 3
    log.write_bytes(content.encode("utf-8"))
    runner = TrainingRunner(project_root=str(tmp_path))
    runner._run_meta["r"] = {"log_path": str(log)}
    with open(log) as stream:
        expected = "".join(stream.readlines()[-4:])
    assert runner.read_recent_logs("r", tail=4) == expected
