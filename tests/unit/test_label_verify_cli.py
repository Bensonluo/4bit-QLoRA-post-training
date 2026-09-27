"""label-verify / label-verify-submit CLI:人读输出的统计局限说明与判定下界。

CLI 进程内调用（monkeypatch sys.argv，参考 test_probe_cli 的调用模式），
数据用 _full 确认后的全量预览，全部落在 tmp_path，不碰 outputs/。
"""

import json
import sys

import pytest

from src.workbench.intake_service import IntakeService
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def store(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    return service, session


def _run_verify(monkeypatch, service, session, *extra):
    from scripts import data_intake

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(service.root),
            "label-verify",
            session.session_id,
            "--revision",
            str(session.revision),
            *map(str, extra),
        ],
    )
    return data_intake.main()


def _run_submit(monkeypatch, service, session, verification_id, answers: dict):
    from scripts import data_intake

    argv = [
        "data_intake.py",
        "--store",
        str(service.root),
        "label-verify-submit",
        session.session_id,
        "--verification-id",
        verification_id,
    ]
    for row_id, answer in answers.items():
        argv += ["--answer", f"{row_id}={answer}"]
    monkeypatch.setattr(sys, "argv", argv)
    return data_intake.main()


def _answers_for(pending: dict, wrong_row_id: str | None = None) -> dict:
    """构造提交答案：wrong_row_id 之外的行都给正确答案。"""
    return {
        item["row_id"]: ("故意答错的答案" if item["row_id"] == wrong_row_id else "yes")
        for item in pending["items"]
    }


def test_cli_label_verify_previews_sample_lower_bound(store, monkeypatch, capsys):
    """抽题时 stderr 如实预告下界：即使全部一致，小样本的真实一致率下界也远低于 100%。"""
    service, session = store
    assert _run_verify(monkeypatch, service, session) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["sample_size"] == 5
    assert "即使全部一致" in err, "抽题预告必须说明全对时下界也不到 100%"
    assert "下界" in err, "统计局限说明必须给出可依赖的下界口径"
    attached = service.load(session.session_id).label_verification
    assert attached["verification_id"] == payload["verification_id"]
    assert attached["status"] == "pending"
