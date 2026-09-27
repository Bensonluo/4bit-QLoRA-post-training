"""Full-data CLI uses the same persisted validation and confirmation workflow."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.workbench.intake_service import IntakeService, next_action
from tests.unit.test_full_data import FULL, approved

CLI = Path(__file__).resolve().parents[2] / "scripts/data_intake.py"


@pytest.fixture()
def service(tmp_path):
    return IntakeService(tmp_path / "intake")


def invoke(service, *args):
    return subprocess.run(
        [sys.executable, str(CLI), "--store", str(service.root), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_full_validate_and_confirm_share_service_state(service, tmp_path):
    session = approved(service)
    path = tmp_path / "full.csv"
    path.write_bytes(FULL)
    result = invoke(
        service,
        "full-validate",
        session.session_id,
        "--revision",
        session.revision,
        "--input",
        path,
        "--encoding",
        "utf-8",
        "--delimiter",
        ",",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["full_data"]["preview"]["counts"]["ready"] == 3
    assert "review_full_data" in result.stderr
    assert service.load(session.session_id).source == session.source
    confirmed = invoke(
        service, "full-confirm", session.session_id, "--revision", payload["revision"]
    )
    assert confirmed.returncode == 0, confirmed.stderr
    assert next_action(service.load(session.session_id)) == "awaiting_dataset_split"
    assert "awaiting_dataset_split" in confirmed.stderr
    stale = invoke(service, "full-confirm", session.session_id, "--revision", payload["revision"])
    assert stale.returncode == 2
    assert "已变化" in stale.stderr


def test_cli_reuses_initial_full_and_does_not_promote_sample_implicitly(service):
    session = approved(service, scope="full")
    result = invoke(service, "full-validate", session.session_id, "--revision", session.revision)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["full_data"]["source"]["digest"] == session.source.digest
    sample = approved(service)
    result = invoke(service, "full-validate", sample.session_id, "--revision", sample.revision)
    assert result.returncode == 2
    assert "不能自动当作全量" in result.stderr


def test_cli_blocked_report_is_saved_and_confirmation_refused(service, tmp_path):
    session = approved(service)
    path = tmp_path / "missing.csv"
    path.write_text("编号,客户描述\n1,破损\n")
    result = invoke(
        service,
        "full-validate",
        session.session_id,
        "--revision",
        session.revision,
        "--input",
        path,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "needs_full_data_revision" in result.stderr
    assert report["full_data"]["schema_drift"]["missing_required"] == ["类别"]
    result = invoke(service, "full-confirm", session.session_id, "--revision", report["revision"])
    assert result.returncode == 2
    assert "未解决" in result.stderr


@pytest.mark.parametrize("groups", [["编号"], []])
def test_cli_materialize_emits_real_files_and_config(service, tmp_path, groups):
    session = approved(service, group_columns=groups)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    args = [
        "materialize",
        session.session_id,
        "--revision",
        session.revision,
        "--name",
        "test-intake",
        "--registry-root",
        tmp_path / "registry",
        "--validation-fraction",
        "0.2",
        "--test-fraction",
        "0.2",
        "--seed",
        "17",
    ]
    if not groups:
        blocked = invoke(service, *args)
        assert blocked.returncode == 2
        assert "独立业务对象" in blocked.stderr
        args.append("--independent-rows-confirmed")
    result = invoke(service, *args)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "ready_for_training_preflight" in result.stderr
    assert payload["dataset"]["statistics"]["row_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    for split in ("train", "validation", "test"):
        assert Path(payload["dataset"]["paths"][split]).is_file()
    assert payload["dataset"]["data_config"]["validation_split"] == 0
