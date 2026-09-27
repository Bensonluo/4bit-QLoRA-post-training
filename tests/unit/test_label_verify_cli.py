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
    """构造提交答案：wrong_row_id 之外的行都给正确答案（全量数据的标签都是 yes）。"""
    return {
        row_id: ("故意答错的答案" if row_id == wrong_row_id else "yes")
        for row_id in pending["row_ids"]
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


def test_cli_label_verify_discloses_shortfall_when_labelled_rows_fewer(store, monkeypatch, capsys):
    """已标注行不足所选条数时，stderr 如实说明缩水，不冒充按请求数抽题。"""
    service, session = store
    assert _run_verify(monkeypatch, service, session, "--size", 50) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["sample_size"] == 9, "FULL 全量只有 9 条已标注行"
    assert "已标注行只有 9 条" in err and "不足你选择的 50 条" in err
    assert "本轮 9 条即使全部一致" in err, "证据说明按缩水后的实际样本量口径"


def test_cli_submit_verdict_line_shows_lower_bound_when_verified(store, monkeypatch, capsys):
    """判定行补上下界（通过态）：5/5 一致的下界远低于 100%，观测一致率不许被当成真实水平。"""
    service, session = store
    assert _run_verify(monkeypatch, service, session) == 0
    pending = json.loads(capsys.readouterr().out)
    capsys.readouterr()

    from src.workbench.intake_service import wilson_lower_bound

    assert (
        _run_submit(
            monkeypatch, service, session, pending["verification_id"], _answers_for(pending)
        )
        == 0
    )
    _, err = capsys.readouterr()
    assert "verified" in err
    assert "5/5 一致" in err
    expected = f"{wilson_lower_bound(5, 5):.0%}"
    assert f"95% 置信下界约 {expected}" in err, "通过判定也必须亮出下界数字"
    assert "下界约 100%" not in err, "小样本全对的下界不许冒充 100%"


def test_cli_submit_verdict_line_shows_lower_bound_when_failed(store, monkeypatch, capsys):
    """判定行补上下界（未通过态）：存在不一致时判定行同样给出下界数字。"""
    service, session = store
    assert _run_verify(monkeypatch, service, session) == 0
    pending = json.loads(capsys.readouterr().out)
    capsys.readouterr()

    from src.workbench.intake_service import wilson_lower_bound

    wrong_row = pending["row_ids"][0]
    assert (
        _run_submit(
            monkeypatch,
            service,
            session,
            pending["verification_id"],
            _answers_for(pending, wrong_row_id=wrong_row),
        )
        == 0
    )
    _, err = capsys.readouterr()
    assert "insufficient_agreement" in err
    assert "4/5 一致" in err
    expected = f"{wilson_lower_bound(4, 5):.0%}"
    assert f"95% 置信下界约 {expected}" in err
    assert "训练不会开始" in err


def test_cli_submit_hint_is_copyable_verbatim(store, monkeypatch, capsys):
    """submit_hint 照抄就能用：提示里的参数与解析器一致，不含会被拒的 --revision。"""
    service, session = store
    assert _run_verify(monkeypatch, service, session) == 0
    payload = json.loads(capsys.readouterr().out)
    hint = payload["submit_hint"]
    assert hint.startswith("data_intake.py label-verify-submit")
    assert "--revision" not in hint, "提交路径不收 --revision，提示不得把用户引向报错"
    assert "--verification-id" in hint and "--answer" in hint
