"""盲标核验:监督信号的业务含义经用户独立复现,否则训练不开始。"""

import pytest

from src.workbench.intake_service import IntakeService
from src.workbench.training_runs import TrainingRunService
from tests.unit.test_data_materialize import _full


@pytest.fixture()
def store(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    return service, session


def _answers_from(service, session, mutate=None):
    pending = service.start_label_verification(session.session_id, session.revision)
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    answers = {item["row_id"]: targets[item["row_id"]] for item in pending["items"]}
    if mutate is not None:
        mutate(answers, pending)
    return pending, answers


def test_start_hides_targets_and_samples_deterministically(store):
    service, session = store
    pending = service.start_label_verification(session.session_id, session.revision)
    assert pending["sample_size"] <= 5
    assert all(set(item) == {"row_id", "input"} for item in pending["items"])
    assert all("target" not in item and "答案" not in item["input"] for item in pending["items"])
    again = service.start_label_verification(session.session_id, session.revision)
    assert [i["row_id"] for i in again["items"]] == [i["row_id"] for i in pending["items"]]


def test_full_agreement_verifies_and_partial_reports_every_mismatch(store):
    service, session = store
    pending, answers = _answers_from(service, session)
    first = pending["items"][0]["row_id"]
    answers[first] = "明显不同的答案"
    result = service.submit_label_verification(
        session.session_id, pending["verification_id"], answers
    )
    assert result["verdict"] == "insufficient_agreement"
    assert result["matched"] == result["sample_size"] - 1
    bad = next(item for item in result["items"] if not item["match"])
    assert bad["row_id"] == first
    assert bad["data_label"] and bad["submitted_answer"] == "明显不同的答案"

    pending, answers = _answers_from(service, session)
    verified = service.submit_label_verification(
        session.session_id, pending["verification_id"], answers
    )
    assert verified["verdict"] == "verified"
    assert verified["agreement"] == 1.0
    refreshed = service.load(session.session_id)
    assert refreshed.label_verification["verdict"] == "verified"


def test_answer_set_and_resubmission_are_strict(store):
    service, session = store
    pending, answers = _answers_from(service, session)
    with pytest.raises(ValueError, match="不能多答或漏答"):
        service.submit_label_verification(session.session_id, pending["verification_id"], {})
    with pytest.raises(ValueError, match="不要留空"):
        service.submit_label_verification(
            session.session_id,
            pending["verification_id"],
            dict.fromkeys(answers, ""),
        )
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)
    with pytest.raises(ValueError, match="已提交过结论"):
        service.submit_label_verification(session.session_id, pending["verification_id"], answers)


def test_requires_confirmed_full_data_and_current_revision(store):
    service, session = store
    service.start_label_verification(session.session_id, session.revision)  # 已确认全量可直接开始
    changed = service.answer(session.session_id, "补充说明")
    with pytest.raises(ValueError, match="任务已更新"):
        service.start_label_verification(changed.session_id, session.revision)


def _assert_prepare_blocked(training, session, model_dir):
    record = training.prepare(session, model_dir, max_length=32)
    assert record["status"] == "blocked"
    assert "盲标核验" in record["issues"][-1]["message"]


def test_training_prepare_blocked_until_verified(store, tmp_path):
    service, session = store
    session = service.materialize_dataset(session.session_id, session.revision)
    training = TrainingRunService(tmp_path / "runs")
    _assert_prepare_blocked(training, session, tmp_path)

    pending, answers = _answers_from(service, session)
    first = pending["items"][0]["row_id"]
    answers[first] = "错的"
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)
    unverified = service.load(session.session_id)
    assert unverified.label_verification["verdict"] == "insufficient_agreement"
    _assert_prepare_blocked(training, unverified, tmp_path)

    pending, answers = _answers_from(service, session)
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)
    verified = service.load(session.session_id)
    record = training.prepare(verified, tmp_path, max_length=32)
    assert record["status"] in {"prepared", "blocked"}
    assert all("盲标核验" not in issue["message"] for issue in record["issues"])


def test_recipe_change_invalidates_previous_verification(store, tmp_path):
    service, session = store
    session = service.materialize_dataset(session.session_id, session.revision)
    pending, answers = _answers_from(service, session)
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)
    assert service.load(session.session_id).label_verification["verdict"] == "verified"

    # 修改配方(新增一个 replace 转换)并重新确认全量:监督语义变化,旧核验失效。
    from copy import deepcopy

    from src.workbench.intake_models import Transform
    from tests.unit.test_data_materialize import FULL

    analysis = deepcopy(session.analysis)
    analysis.recipe.inputs[0].transforms.append(Transform(operation="replace", old="x", new="y"))
    updated = service.apply_analysis(session, analysis)
    updated = service.confirm(updated.session_id, updated.revision)
    updated = service.validate_full_data(updated.session_id, updated.revision, "full.csv", FULL)
    updated = service.confirm_full_data(updated.session_id, updated.revision)
    refreshed = service.load(updated.session_id)
    assert refreshed.label_verification.get("stale") is True  # 修订后失效可见,而非静默消失
    training = TrainingRunService(tmp_path / "runs")
    rematerialized = service.materialize_dataset(refreshed.session_id, refreshed.revision)
    _assert_prepare_blocked(training, rematerialized, tmp_path)


def _contrast_store(tmp_path):
    from tests.unit.test_data_intake import CSV, analysis

    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户首次描述预测类别", "工单.csv", CSV)
    session = service.apply_analysis(session, analysis())
    return service, session


def test_contrast_check_pairs_answers_and_records_mismatch(tmp_path):
    """对比核验:答案配对正确才 verified,配错留档并给出纠正提示。"""
    service, session = _contrast_store(tmp_path)
    pending = service.start_contrast_check(session.session_id, session.revision)
    assert len(pending["items"]) == 2
    assert len(set(pending["options"])) == 2
    assert all("target" not in item and item["input"] for item in pending["items"])
    targets = {row.row_id: row.target for row in session.preview.rows}

    # 故意配反
    wrong = {
        pending["items"][0]["row_id"]: targets[pending["items"][1]["row_id"]],
        pending["items"][1]["row_id"]: targets[pending["items"][0]["row_id"]],
    }
    result = service.submit_contrast_check(session.session_id, pending["check_id"], wrong)
    assert result["verdict"] == "mismatch"
    assert all(not item["match"] for item in result["items"])
    assert service.contrast_check_status(session.session_id)["verdict"] == "mismatch"

    # 正确配对(两轮换题:种子含轮数,二连对才 needs_second_round=False)
    second_round = service.start_contrast_check(session.session_id, session.revision)
    if {i["row_id"] for i in second_round["items"]} == {i["row_id"] for i in pending["items"]}:
        pass  # 数据行太少时两轮可能同题;连胜语义不受影响
    right = {item["row_id"]: targets[item["row_id"]] for item in second_round["items"]}
    result = service.submit_contrast_check(session.session_id, second_round["check_id"], right)
    assert result["verdict"] == "verified"
    status = service.contrast_check_status(session.session_id)
    assert status["verdict"] == "verified"
    assert status["streak"] >= 1

    # 一轮对之后一轮错:连胜归零,需要重新二连对
    pending = service.start_contrast_check(session.session_id, session.revision)
    wrong = {
        pending["items"][0]["row_id"]: targets[pending["items"][1]["row_id"]],
        pending["items"][1]["row_id"]: targets[pending["items"][0]["row_id"]],
    }
    service.submit_contrast_check(session.session_id, pending["check_id"], wrong)
    status = service.contrast_check_status(session.session_id)
    assert status["verdict"] == "mismatch" and status["streak"] == 0
    assert status["needs_second_round"] is True


def test_contrast_check_rejects_bad_submissions_and_stale_binding(tmp_path):
    service, session = _contrast_store(tmp_path)
    pending = service.start_contrast_check(session.session_id, session.revision)
    with pytest.raises(ValueError, match="各选一个答案"):
        service.submit_contrast_check(session.session_id, pending["check_id"], {})
    with pytest.raises(ValueError, match="来自给出的选项"):
        service.submit_contrast_check(
            session.session_id,
            pending["check_id"],
            {
                pending["items"][0]["row_id"]: "不存在的选项",
                pending["items"][1]["row_id"]: pending["options"][0],
            },
        )
    # 预览变化(修订业务说明)后,旧对比核验失效
    changed = service.answer(session.session_id, "修订说明")
    with pytest.raises(ValueError, match="失效"):
        service.submit_contrast_check(
            session.session_id,
            pending["check_id"],
            {item["row_id"]: pending["options"][0] for item in pending["items"]},
        )
    assert changed


def test_stale_verification_is_visible_after_revision(store, tmp_path):
    """修订后旧核验不再静默消失:会话带上 stale 标记,页面可明确告知需重验。"""
    service, session = store
    session = service.materialize_dataset(session.session_id, session.revision)
    pending, answers = _answers_from(service, session)
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)
    assert service.load(session.session_id).label_verification["verdict"] == "verified"

    from copy import deepcopy

    from src.workbench.intake_models import Transform
    from tests.unit.test_data_materialize import FULL

    analysis = deepcopy(session.analysis)
    analysis.recipe.inputs[0].transforms.append(Transform(operation="replace", old="x", new="y"))
    updated = service.apply_analysis(session, analysis)
    updated = service.confirm(updated.session_id, updated.revision)
    updated = service.validate_full_data(updated.session_id, updated.revision, "full.csv", FULL)
    updated = service.confirm_full_data(updated.session_id, updated.revision)
    refreshed = service.load(updated.session_id)
    assert refreshed.label_verification is not None
    assert refreshed.label_verification.get("stale") is True
    assert refreshed.label_verification.get("previous_verdict") == "verified"
    # 训练门禁仍然拦截(stale 无 verified verdict)
    training = TrainingRunService(tmp_path / "runs")
    rematerialized = service.materialize_dataset(refreshed.session_id, refreshed.revision)
    record = training.prepare(rematerialized, tmp_path, max_length=32)
    assert record["status"] == "blocked"
    assert "盲标核验" in record["issues"][-1]["message"]


def test_blind_sampling_covers_rare_classes(tmp_path):
    """抽样数少于类别数时按标签轮转:稀有类必须被抽到。"""
    from tests.unit.test_data_intake import CSV, analysis
    from tests.unit.test_full_data import FULL as TWO_CLASS_FULL
    from tests.unit.test_data_materialize import _full as _materialize_full

    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户首次描述预测类别", "工单.csv", CSV)
    session = service.apply_analysis(session, analysis())
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", TWO_CLASS_FULL
    )
    session = service.confirm_full_data(session.session_id, session.revision)
    pending = service.start_label_verification(session.session_id, session.revision, sample_size=2)
    targets = {r.row_id: r.target for r in session.full_data.preview.rows}
    sampled = {targets[item["row_id"]] for item in pending["items"]}
    assert len(sampled) == 2, f"两个名额应覆盖两个不同类别,实得 {sampled}"
