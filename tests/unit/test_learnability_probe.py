"""可学性探针:零样本 vs 多数类基线的最便宜证据,附样本量限制声明。"""

import json

import pytest

from src.workbench.business_evaluation import Generation
from src.workbench.intake_service import IntakeService
from src.workbench.learnability_probe import probe_learnability, save_probe
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
                assert protocol.max_new_tokens > 0
                return Generation(answer, truncated=False)

            def close(self):
                pass

        return Runtime()

    return make


def test_probe_reports_accuracy_against_majority_baseline(store, tmp_path):
    _, session = store
    labels = [row["output"] for row in _labels(session)]
    majority = max(set(labels), key=labels.count)

    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory(majority), sample_size=4
    )
    assert result["zero_shot_accuracy"] == 1.0
    assert result["majority_baseline"] > 0
    assert result["difference"] >= 0
    assert result["sample_size"] == min(4, result["dev_total"])
    assert "不能预测微调效果" in result["note"]
    assert all(item["match"] for item in result["observations"])
    path = save_probe(tmp_path / "probes", result)
    assert path.exists()

    # 全错的零样本:与基线的差为负,如实呈现
    wrong = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("绝不正确的答案"), sample_size=4
    )
    assert wrong["zero_shot_accuracy"] == 0.0
    assert wrong["difference"] == -wrong["majority_baseline"]
    assert wrong["observations"][0]["expected"] in labels


def _labels(session):
    import json

    path = session.dataset.paths["validation"]
    return [json.loads(line) for line in open(path) if line.strip()]


def test_probe_validates_inputs(store):
    _, session = store
    with pytest.raises(ValueError, match="抽样数量"):
        probe_learnability(session, "/tmp/base", runtime_factory=_factory("x"), sample_size=0)
    bare = session.model_copy(deep=True)
    bare.dataset = None
    with pytest.raises(ValueError, match="独立分区"):
        probe_learnability(bare, "/tmp/base", runtime_factory=_factory("x"))


def test_label_error_candidates_cross_signal_ranking(store, tmp_path):
    """基座与用户盲标双信号都矛盾的行排前;截断不作为分歧证据。"""
    _, session = store
    labels = [row["output"] for row in _labels(session)]
    majority = max(set(labels), key=labels.count)

    # 先造一个用户盲标分歧信号:提交与数据标签不同的答案(同库服务)
    service = store[0]
    current = service.load(session.session_id)
    pending = service.start_label_verification(current.session_id, current.revision, sample_size=1)
    answers = {item["row_id"]: "用户也不认同的答案" for item in pending["items"]}
    service.submit_label_verification(current.session_id, pending["verification_id"], answers)
    current = service.load(session.session_id)
    assert current.label_verification["verdict"] == "insufficient_agreement"

    calls = {"n": 0}

    def make(model):
        class Runtime:
            def generate(self, prompt, protocol):
                calls["n"] += 1
                answer = "基座不认同的输出" if calls["n"] % 2 else majority
                return Generation(answer, truncated=False)

            def close(self):
                pass

        return Runtime()

    result = probe_learnability(
        current, "/tmp/base", runtime_factory=lambda model: make(model), sample_size=2
    )
    candidates = result["label_error_candidates"]
    assert isinstance(candidates, list)
    assert "候选不等于错误" in result["candidates_note"]
    strong = [c for c in candidates if c["user_blind_answer"]]
    if strong:
        assert candidates[0]["user_blind_answer"], "强证据排前"
        assert "强证据" in candidates[0]["evidence"]
        # 强证据行必须带溯源提示:建议对照原始来源行,不要只凭模型输出改数据
        assert "原始来源行" in candidates[0]["evidence"]
        weak = [c for c in candidates if not c["user_blind_answer"]]
        assert all("原始来源行" not in c["evidence"] for c in weak), "弱信号不冒充溯源建议"
    assert "原始来源行" in result["candidates_note"]


def test_describe_candidates_plain_lines_and_empty_message():
    """人话清单:强证据带 ⚠ 逐行可核对;没有候选时给明确说法而不是沉默。"""
    from src.workbench.learnability_probe import describe_candidates

    assert describe_candidates([]) == ["没有发现值得优先核对的行。"]

    lines = describe_candidates(
        [
            {
                "row_id": "r1",
                "data_label": "质量",
                "base_zero_shot": "物流",
                "user_blind_answer": "服务",
                "evidence": "强证据,优先人工核对",
            },
            {
                "row_id": "r2",
                "data_label": "物流",
                "base_zero_shot": "质量",
                "user_blind_answer": None,
                "evidence": "弱信号供参考",
            },
        ]
    )
    assert len(lines) == 3
    assert "候选 2 行" in lines[0] and "强证据 1 行" in lines[0]
    assert "候选不等于错误" in lines[0]
    assert lines[1].startswith("⚠") and "r1" in lines[1] and "服务" in lines[1]
    assert "原始来源行" in lines[1], "强证据行在 CLI 清单里也提示对照原始来源行"
    assert lines[2].startswith("·") and "r2" in lines[2] and "⚠" not in lines[2]
    assert "原始来源行" not in lines[2]


def test_candidates_csv_export_is_excel_friendly():
    from src.workbench.learnability_probe import candidates_to_csv

    data = candidates_to_csv(
        [
            {
                "row_id": "r1",
                "data_label": "质量",
                "base_zero_shot": "物流",
                "user_blind_answer": None,
                "evidence": "弱信号",
            },
        ]
    )
    text = data.decode("utf-8-sig")
    assert "行ID" in text and "r1" in text and "弱信号" in text
    assert data.startswith(b"\xef\xbb\xbf")  # BOM: Excel 直接打开不乱码


def test_candidates_csv_export_covers_all_rows_beyond_one_page():
    """页面候选分页只影响展示;CSV 导出必须包含全部候选,不止当前页。"""
    from src.workbench.learnability_probe import candidates_to_csv

    candidates = [
        {
            "row_id": f"r{index:06d}",
            "data_label": "质量",
            "base_zero_shot": "物流",
            "user_blind_answer": None,
            "evidence": "弱信号",
        }
        for index in range(1, 26)
    ]
    text = candidates_to_csv(candidates).decode("utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 26  # 表头 + 全部 25 行,不是只有第一页的 20 行
    assert "r000025" in text


def test_saved_probe_can_be_loaded_back_per_dataset_version(store, tmp_path):
    """探针结果存盘后必须能按数据版本回读——重看结论不需要重新加载模型。"""
    from src.workbench.learnability_probe import load_latest_probe

    _, session = store
    root = tmp_path / "probes"
    assert load_latest_probe(root, session.dataset.version) is None  # 从未运行过

    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("质量"), sample_size=2
    )
    save_probe(root, result)
    loaded = load_latest_probe(root, session.dataset.version)
    assert loaded is not None
    assert loaded["zero_shot_accuracy"] == result["zero_shot_accuracy"]
    assert loaded["observations"] == result["observations"]

    # 数据版本不同(重物化后)不回读旧版本结果;损坏文件跳过不阻塞
    assert load_latest_probe(root, "other-version") is None
    (root / f"{session.dataset.version}-broken.json").write_text("{not json", encoding="utf-8")
    again = load_latest_probe(root, session.dataset.version)
    assert again is not None and again["kind"] == "learnability_probe"


def test_cli_probe_show_reads_saved_result_without_rerunning(store, tmp_path, monkeypatch, capsys):
    """CLI learnability-probe-show 回读已存盘结果;没跑过探针时如实说明。"""
    import sys

    from scripts import data_intake

    service, session = store
    evaluation_root = tmp_path / "eval"  # 探针记录在 evaluation_root.parent / "probes"
    argv = [
        "data_intake.py",
        "--store",
        str(service.root),
        "--evaluation-root",
        str(evaluation_root),
        "learnability-probe-show",
        session.session_id,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert data_intake.main() == 2  # 尚未运行过探针:明确告知,不伪造
    capsys.readouterr()

    result = probe_learnability(
        session, "/tmp/base", runtime_factory=_factory("质量"), sample_size=2
    )
    save_probe(evaluation_root.parent / "probes", result)
    monkeypatch.setattr(sys, "argv", argv)
    assert data_intake.main() == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["zero_shot_accuracy"] == result["zero_shot_accuracy"]
    assert "最近一次已保存的探针结果" in err
