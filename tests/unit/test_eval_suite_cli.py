"""Freezing and selecting an immutable suite through real CLI commands."""

import json
import subprocess
import sys
from pathlib import Path

from src.workbench.intake_service import IntakeService
from tests.unit.test_data_materialize import _full

CLI = Path(__file__).resolve().parents[2] / "scripts/data_intake.py"


def invoke(service, suite_root, *args):
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--store",
            str(service.root),
            "--suite-root",
            str(suite_root),
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_freezes_original_questions_and_materializes_with_same_suite(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    original_version = session.dataset.version
    root = tmp_path / "suites"
    result = invoke(
        service, root, "suite-freeze", session.session_id, "--revision", session.revision
    )
    assert result.returncode == 0, result.stderr
    ref = json.loads(result.stdout)
    assert len(ref["suite_id"]) == 64
    assert set(ref["case_counts"]) == {"validation", "test"}
    # 与其他业务子命令同口径：stdout 纯 JSON，stderr 追加题集人话摘要。
    assert "这套固定题集含开发题" in result.stderr
    assert "原评分题不能修改" in result.stderr
    assert "不代表业务效果达标" in result.stderr
    assert service.load(session.session_id).dataset.version == original_version
    shown = invoke(service, root, "suite-show", ref["suite_id"])
    assert shown.returncode == 0, shown.stderr
    assert ref["suite_id"] in shown.stdout
    assert "原开发/最终测试评分题固定" in shown.stderr
    assert "锚定数据版本" in shown.stderr
    materialized = invoke(
        service,
        root,
        "materialize",
        session.session_id,
        "--revision",
        session.revision,
        "--suite-id",
        ref["suite_id"],
    )
    assert materialized.returncode == 0, materialized.stderr
    updated = json.loads(materialized.stdout)
    assert updated["dataset"]["evaluation_suite"]["suite_id"] == ref["suite_id"]
    assert updated["dataset"]["evaluation_suite"]["cases_digest"] == ref["cases_digest"]


def test_cli_suite_freeze_rejects_stale_revision(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    result = invoke(
        service,
        tmp_path / "suites",
        "suite-freeze",
        session.session_id,
        "--revision",
        session.revision + 1,
    )
    assert result.returncode == 2
    assert "任务已更新" in result.stderr
