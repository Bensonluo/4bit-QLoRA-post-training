"""可学性探针:零样本 vs 多数类基线的最便宜证据,附样本量限制声明。"""

import pytest

from src.workbench.intake_service import IntakeService
from src.workbench.learnability_probe import probe_learnability, save_probe
from src.workbench.business_evaluation import Generation
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
