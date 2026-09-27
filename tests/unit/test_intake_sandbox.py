"""Real OS isolation checks; unavailable backends are skipped, never emulated."""

import platform

import pytest

from src.workbench.sandbox import SandboxLimits, TransformCase, TransformSandbox, check_source

SOURCE = r"""import re
def transform(rows, config):
    output = []
    for row in rows:
        match = re.fullmatch(r"编号:(\d+)", row["raw"])
        if not match:
            raise ValueError("编号格式不符合已确认规则")
        output.append({**row, "number": match.group(1)})
    return output
"""


@pytest.fixture()
def sandbox():
    runner = TransformSandbox(backend="macos" if platform.system() == "Darwin" else "docker")
    if not runner.detect():
        pytest.skip(runner.unavailable_reason)
    return runner


def test_contract_check_is_not_claimed_as_isolation():
    check_source(SOURCE)
    with pytest.raises(ValueError, match="仅允许"):
        check_source("import os\ndef transform(rows, config): return rows")
    with pytest.raises(ValueError, match="唯一"):
        check_source("def transform(rows): return rows")


def test_unavailable_backend_never_executes_host_code(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-exist"
    runner = TransformSandbox()
    monkeypatch.setattr(runner, "detect", lambda: False)
    result = runner.run(
        f'def transform(rows, config):\n open({str(marker)!r}, "w").write("bad")\n return rows', []
    )
    assert result.status == "unavailable"
    assert not marker.exists()


def test_freeze_requires_business_expectations_and_counterexamples():
    runner = TransformSandbox()
    with pytest.raises(ValueError, match="反例"):
        runner.validate(SOURCE, [TransformCase("业务", [], [])])
    with pytest.raises(ValueError, match="预期"):
        runner.validate(
            SOURCE,
            [TransformCase("业务", []), TransformCase("反例", [], [], kind="counterexample")],
        )


def test_real_transform_and_frozen_validation_report(sandbox):
    rows = [{"_row_id": "r001", "raw": "编号:0012"}]
    expected = [{"_row_id": "r001", "raw": "编号:0012", "number": "0012"}]
    result = sandbox.run(SOURCE, rows)
    assert result.status == "passed", result.error
    assert result.rows == expected
    report = sandbox.validate(
        SOURCE,
        [
            TransformCase("真实含前导零编号", rows, expected),
            TransformCase(
                "非编号不能臆造转换", [{"raw": "未知"}], kind="counterexample", expect_error=True
            ),
        ],
    )
    assert report.status == "passed", report.cases
    assert report.source_digest == result.source_digest
    assert len(report.cases_digest) == 64
    assert all(case["passed"] for case in report.cases)


def test_real_sandbox_denies_reading_unmounted_host_file(sandbox, tmp_path):
    canary = tmp_path / "private-canary.txt"
    canary.write_text("synthetic-private-content")
    result = sandbox.run(
        'def transform(rows, config):\n return [{"leak": open(config["path"]).read()}]',
        [],
        {"path": str(canary)},
    )
    assert result.status == "failed"
    assert result.failure_kind == "transform", result.error
    assert "PermissionError" in result.error or "FileNotFoundError" in result.error
    assert result.rows is None


def test_real_sandbox_denies_host_write(sandbox, tmp_path):
    target = tmp_path / "outside-write.txt"
    result = sandbox.run(
        'def transform(rows, config):\n open(config["path"], "w").write("bad")\n return rows',
        [],
        {"path": str(target)},
    )
    assert result.status == "failed", result
    assert not target.exists()


def test_real_sandbox_blocks_network_even_after_python_import_bypass(sandbox):
    source = """import json
def transform(rows, config):
    socket = json.__builtins__["__import__"]("socket")
    connection = socket.socket()
    connection.connect(("127.0.0.1", 9))
    return rows
"""
    result = sandbox.run(source, [])
    assert result.status == "failed"
    assert result.failure_kind == "transform", result.error
    if sandbox.backend == "macos":
        assert "PermissionError" in result.error, result.error


def test_real_sandbox_does_not_inherit_agent_credentials(sandbox, monkeypatch):
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "synthetic-key-not-to-forward")
    source = """import json
def transform(rows, config):
    os = json.__builtins__["__import__"]("os")
    return [{"key": os.environ.get("TUNESMITH_AGENT_API_KEY")}]
"""
    result = sandbox.run(source, [])
    assert result.status == "passed", result.error
    assert result.rows == [{"key": None}]


def test_real_sandbox_timeout_terminates_code(sandbox):
    sandbox.limits = SandboxLimits(timeout_seconds=0.2, cpu_seconds=3)
    result = sandbox.run("def transform(rows, config):\n while True: pass", [])
    assert result.status == "failed"
    assert result.failure_kind == "timeout", result.error


def test_real_sandbox_output_schema_and_size_enforced(sandbox):
    result = sandbox.run("def transform(rows, config): return [42]", [])
    assert result.status == "failed"
    assert result.failure_kind == "transform"
    sandbox.limits = SandboxLimits(max_output_bytes=2048)
    result = sandbox.run('def transform(rows, config): return [{"text": "x" * 4096}]', [])
    assert result.status == "failed"
    assert result.rows is None


def test_real_sandbox_memory_supervision(sandbox):
    sandbox.limits = SandboxLimits(memory_mb=48)
    source = """import json
def transform(rows, config):
    time = json.__builtins__["__import__"]("time")
    value = bytearray(64 * 1024 * 1024)
    time.sleep(2)
    return [{"size": len(value)}]
"""
    result = sandbox.run(source, [])
    assert result.status == "failed"
    assert result.failure_kind in {"memory_limit", "transform", "execution"}
