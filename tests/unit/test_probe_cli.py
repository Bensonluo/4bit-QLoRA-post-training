"""learnability-probe CLI:候选清单人话输出、存盘回读一致与 CSV 导出。

CLI 进程内调用（monkeypatch sys.argv，参考 test_intake_preflight_cli 的调用
模式），runtime 用 mock factory 替换（参考 test_learnability_probe 的
_factory 模式），不加载真实模型；探针记录全部落在 tmp_path，不碰 outputs/。
"""

import json
import sys

import pytest

from src.workbench.business_evaluation import Generation
from src.workbench.intake_service import IntakeService
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def store(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    return service, session


def _factory(answer):
    def make(model):
        class Runtime:
            def generate(self, prompt, protocol):
                return Generation(answer, truncated=False)

            def close(self):
                pass

        return Runtime()

    return make


def _mock_runtime(monkeypatch, answer):
    monkeypatch.setattr("src.workbench.learnability_probe._default_runtime", _factory(answer))


def _run_probe(monkeypatch, service, evaluation_root, session, *extra):
    from scripts import data_intake

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(service.root),
            "--evaluation-root",
            str(evaluation_root),
            "learnability-probe",
            session.session_id,
            "--revision",
            str(session.revision),
            "--model-path",
            "/tmp/base",
            *extra,
        ],
    )
    return data_intake.main()


def test_cli_probe_prints_candidates_in_plain_language(store, tmp_path, monkeypatch, capsys):
    """有分歧时 stderr 逐行列出候选；基座与盲标双信号矛盾的强证据行带 ⚠。"""
    service, session = store
    # 盲标抽样覆盖全部标注行:每个开发集行都有"用户盲标也不认同"的强信号
    pending = service.start_label_verification(session.session_id, session.revision, sample_size=50)
    answers = {item["row_id"]: "用户也不认同的答案" for item in pending["items"]}
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)

    evaluation_root = tmp_path / "eval"
    _mock_runtime(monkeypatch, "绝不正确的答案")
    assert _run_probe(monkeypatch, service, evaluation_root, session) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    candidates = payload["label_error_candidates"]
    assert candidates
    assert "没有发现值得优先核对的行" not in err
    assert "标签问题候选" in err and "候选不等于错误" in err
    for candidate in candidates:
        assert f"行 {candidate['row_id']}" in err
        assert candidate["data_label"] in err
        assert candidate["base_zero_shot"] in err
    # 盲标覆盖全部标注行:每个候选都应是强证据行,逐行带 ⚠
    assert err.count("⚠") == len(candidates)


def test_cli_probe_without_candidates_says_so(store, tmp_path, monkeypatch, capsys):
    """零样本全部命中时没有候选,stderr 明确说明,不装模作样列空表。"""
    service, session = store
    evaluation_root = tmp_path / "eval"
    _mock_runtime(monkeypatch, "yes")  # 全量数据标签都是 yes,即多数类
    assert _run_probe(monkeypatch, service, evaluation_root, session) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["label_error_candidates"] == []
    assert "没有发现值得优先核对的行" in err
    assert "⚠" not in err


def test_cli_probe_show_roundtrips_candidates_and_note(store, tmp_path, monkeypatch, capsys):
    """存盘→回读:候选字段与 note 原样可得,stderr 同样逐行列出候选。"""
    from scripts import data_intake

    service, session = store
    evaluation_root = tmp_path / "eval"
    pending = service.start_label_verification(session.session_id, session.revision, sample_size=50)
    answers = {item["row_id"]: "用户也不认同的答案" for item in pending["items"]}
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)

    _mock_runtime(monkeypatch, "绝不正确的答案")
    assert _run_probe(monkeypatch, service, evaluation_root, session) == 0
    saved_payload = json.loads(capsys.readouterr().out)
    capsys.readouterr()

    # show 不重新加载模型:此刻把默认运行时换成会炸的工厂,回读仍须成功
    def _boom(model):
        raise AssertionError("learnability-probe-show 不得重新加载模型")

    monkeypatch.setattr("src.workbench.learnability_probe._default_runtime", _boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(service.root),
            "--evaluation-root",
            str(evaluation_root),
            "learnability-probe-show",
            session.session_id,
        ],
    )
    assert data_intake.main() == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["kind"] == "learnability_probe"
    # 候选字段与人话说明存盘后原样回读,一字不差
    assert payload["label_error_candidates"] == saved_payload["label_error_candidates"]
    assert payload["candidates_note"] == saved_payload["candidates_note"]
    assert payload["note"] == saved_payload["note"]
    assert payload["label_error_candidates"]
    for candidate in payload["label_error_candidates"]:
        assert f"行 {candidate['row_id']}" in err
    assert err.count("⚠") == len(payload["label_error_candidates"])
