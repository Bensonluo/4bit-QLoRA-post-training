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


def test_cli_label_verify_export_csv_writes_blind_question_list(
    store, tmp_path, monkeypatch, capsys
):
    """--export-csv 抽题后落盘：题目清单含行ID与题目输入，绝不含数据答案。

    与 learnability-probe 的候选导出同款（BOM 表头、Excel 直开、父目录自动
    创建）；差别是盲标清单不许泄露标签——导出文件泄露答案，核验就失效。
    """
    service, session = store
    csv_path = tmp_path / "exports" / "questions.csv"
    assert _run_verify(monkeypatch, service, session, "--export-csv", csv_path) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert csv_path.exists()
    text = csv_path.read_bytes().decode("utf-8-sig")
    assert text.startswith("行ID")  # 表头：BOM 之后第一行就是列名
    assert "行ID,题目输入（仅根据此列作答）,你的盲标答案（填写后提交）" in text
    inputs = {
        row.row_id: row.input for row in service.load(session.session_id).full_data.preview.rows
    }
    for row_id in payload["row_ids"]:
        assert row_id in text
        assert inputs[row_id] in text, "题目列必须是隐藏答案后的真实输入"
    assert "yes" not in text, "数据标签 yes 不得出现在题目清单里，否则盲标失效"
    listed = [line for line in text.splitlines() if line.strip()]
    assert len(listed) == payload["sample_size"] + 1, "每条抽样题一行，外加表头"
    assert "题目清单已导出" in err


def test_cli_label_verify_export_csv_not_written_when_sampling_rejected(
    store, tmp_path, monkeypatch, capsys
):
    """抽题被拒（版本过期）时不写文件，也不装作导出成功。"""
    service, session = store
    csv_path = tmp_path / "questions.csv"
    assert (
        _run_verify(
            monkeypatch,
            service,
            session,
            "--revision",
            session.revision + 1,
            "--export-csv",
            csv_path,
        )
        == 2
    )
    _, err = capsys.readouterr()
    assert not csv_path.exists()
    assert "任务已更新" in err
