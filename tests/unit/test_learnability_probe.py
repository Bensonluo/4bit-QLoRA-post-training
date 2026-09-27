"""可学性探针:零样本 vs 多数类基线的最便宜证据,附样本量限制声明。"""

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
