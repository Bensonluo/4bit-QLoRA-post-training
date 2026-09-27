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


def test_contrast_check_third_round_streak_keeps_counting(tmp_path):
    """二连对达标后用户可选继续第三轮:连胜如实计数,核验语义不变。"""
    service, session = _contrast_store(tmp_path)
    targets = {row.row_id: row.target for row in session.preview.rows}
    for round_number in range(1, 4):
        pending = service.start_contrast_check(session.session_id, session.revision)
        mapping = {item["row_id"]: targets[item["row_id"]] for item in pending["items"]}
        result = service.submit_contrast_check(session.session_id, pending["check_id"], mapping)
        assert result["verdict"] == "verified"
        status = service.contrast_check_status(session.session_id)
        assert status["verdict"] == "verified"
        assert status["streak"] == round_number
        assert status["needs_second_round"] is (round_number < 2)

    # 第三轮后配错一次:连胜归零,回到需要重新二连对的状态
    pending = service.start_contrast_check(session.session_id, session.revision)
    wrong = {
        pending["items"][0]["row_id"]: targets[pending["items"][1]["row_id"]],
        pending["items"][1]["row_id"]: targets[pending["items"][0]["row_id"]],
    }
    service.submit_contrast_check(session.session_id, pending["check_id"], wrong)
    status = service.contrast_check_status(session.session_id)
    assert status["verdict"] == "mismatch" and status["streak"] == 0
    assert status["needs_second_round"] is True


def test_contrast_check_status_records_every_round_history(tmp_path):
    """状态带逐轮历史:每轮的题目、选择与对错可回查,二连对有据可查。"""
    service, session = _contrast_store(tmp_path)
    targets = {row.row_id: row.target for row in session.preview.rows}
    inputs = {row.row_id: row.input for row in session.preview.rows}

    # 第 1 轮故意配错,第 2、3 轮配对:历史按轮次如实留痕
    pending = service.start_contrast_check(session.session_id, session.revision)
    wrong = {
        pending["items"][0]["row_id"]: targets[pending["items"][1]["row_id"]],
        pending["items"][1]["row_id"]: targets[pending["items"][0]["row_id"]],
    }
    service.submit_contrast_check(session.session_id, pending["check_id"], wrong)
    for _ in range(2):
        pending = service.start_contrast_check(session.session_id, session.revision)
        right = {item["row_id"]: targets[item["row_id"]] for item in pending["items"]}
        service.submit_contrast_check(session.session_id, pending["check_id"], right)

    status = service.contrast_check_status(session.session_id)
    history = status["history"]
    assert [entry["round"] for entry in history] == [1, 2, 3]
    assert [entry["verdict"] for entry in history] == ["mismatch", "verified", "verified"]
    for entry in history:
        assert len(entry["items"]) == 2
        for item in entry["items"]:
            assert item["row_id"] in inputs
            assert item["input"] == inputs[item["row_id"]], "题目取自当前预览行,不是空话"
            assert item["correct_answer"] == targets[item["row_id"]]
            assert item["chosen"], "每一轮都有真实作答记录"
            assert (item["chosen"] == item["correct_answer"]) is item["match"]
        assert all(item["match"] is (entry["verdict"] == "verified") for item in entry["items"])
    # 连胜只数最近连续 verified:第 1 轮错、第 2/3 轮对 → 恰好 2,不需要再核验
    assert status["streak"] == 2
    assert status["needs_second_round"] is False
    # 最近一轮的明细仍在(向后兼容既有字段)
    assert status["verdict"] == "verified"


def test_contrast_check_history_empty_before_any_round(tmp_path):
    """没做过核验时状态为 None,不伪造历史;做过一轮只回一轮。"""
    service, session = _contrast_store(tmp_path)
    assert service.contrast_check_status(session.session_id) is None

    targets = {row.row_id: row.target for row in session.preview.rows}
    pending = service.start_contrast_check(session.session_id, session.revision)
    mapping = {item["row_id"]: targets[item["row_id"]] for item in pending["items"]}
    service.submit_contrast_check(session.session_id, pending["check_id"], mapping)
    status = service.contrast_check_status(session.session_id)
    assert [entry["round"] for entry in status["history"]] == [1]
    assert status["history"][0]["verdict"] == "verified"


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


_BIG_FULL = (
    "编号,客户描述,类别,处理结果\n"
    + "".join(
        f"{index:03d},客户描述{index},{'质量' if index % 2 else '物流'},补发\n"
        for index in range(1, 11)
    )
).encode()


def _big_full_store(tmp_path):
    """10 条已标注全量数据:两轮抽样(各 5 条)可以换题。"""
    from tests.unit.test_full_data import approved

    service = IntakeService(tmp_path / "intake")
    session = approved(service)
    session = service.validate_full_data(
        session.session_id, session.revision, "full.csv", _BIG_FULL
    )
    session = service.confirm_full_data(session.session_id, session.revision)
    return service, session


def test_blind_verification_rolls_questions_between_rounds(tmp_path):
    """核验未通过会公布正确标签;下一轮必须换题,照抄公布答案不能通过核验。

    同一轮状态内重复抽题仍保持确定性(刷新页面不换题、无法反复抽到简单题)。
    """
    service, session = _big_full_store(tmp_path)

    # 第 1 轮:5 条题,故意全答错(模拟"看了公布答案再背题"的用户)
    first = service.start_label_verification(session.session_id, session.revision)
    wrong = {item["row_id"]: "背出来的答案" for item in first["items"]}
    service.submit_label_verification(session.session_id, first["verification_id"], wrong)

    # 第 2 轮:题目必须换一组;且本轮状态内重复抽题保持一致(确定性,防反复抽题)
    second = service.start_label_verification(session.session_id, session.revision)
    assert {i["row_id"] for i in second["items"]} != {i["row_id"] for i in first["items"]}
    twin = service.start_label_verification(session.session_id, session.revision)
    assert [i["row_id"] for i in twin["items"]] == [i["row_id"] for i in second["items"]]

    # 照抄第 1 轮公布的正确标签作答第 2 轮:第 2 轮题目已变,背题不再等于通过
    targets = {r.row_id: r.target for r in service.load(session.session_id).full_data.preview.rows}
    stale_parrot = {row_id: targets[row_id] for row_id in (i["row_id"] for i in first["items"])}
    assert set(stale_parrot) != {i["row_id"] for i in second["items"]}
    # 第 2 轮用本轮自己的题真实作答:提交 twin(后创建的那份)后状态直接落到它上
    mapping = {item["row_id"]: targets[item["row_id"]] for item in twin["items"]}
    verdict = service.submit_label_verification(
        session.session_id, twin["verification_id"], mapping
    )
    assert verdict["verdict"] == "verified"
    # 换题后新轮次的 note 说明换题事实
    assert "换一组题" in second["note"]


def test_wilson_lower_bound_matches_known_values():
    """95% Wilson 下界用已知值校验:5/5≈56.6%、4/5≈37.6%、3/10≈10.8%,纯 Python 不引依赖。"""
    from src.workbench.intake_service import wilson_lower_bound

    assert wilson_lower_bound(5, 5) == pytest.approx(0.5655, abs=1e-3)
    assert wilson_lower_bound(4, 5) == pytest.approx(0.3755, abs=1e-3)
    assert wilson_lower_bound(3, 10) == pytest.approx(0.1078, abs=1e-3)
    assert wilson_lower_bound(30, 30) == pytest.approx(0.8865, abs=1e-3)
    assert wilson_lower_bound(0, 5) == 0.0
    assert wilson_lower_bound(0, 0) == 0.0
    # 同为全对,样本越大下界越高、越接近 100%:小样本的全对不等于高可信
    assert wilson_lower_bound(5, 5) < wilson_lower_bound(20, 20) < wilson_lower_bound(30, 30) < 1.0


def test_agreement_note_has_two_honest_tiers():
    """两档文案:样本 <30 说「下界才是你能依赖的数」;≥30 改说「下界接近观测值」。"""
    from src.workbench.intake_service import agreement_evidence_note

    small = agreement_evidence_note(5, 5)
    assert "5/5 一致" in small
    assert "57%" in small
    assert "下界才是你能依赖的数" in small
    partial = agreement_evidence_note(4, 5)
    assert "4/5 一致" in partial
    assert "38%" in partial
    large = agreement_evidence_note(30, 30)
    assert "89%" in large
    assert "下界接近观测值" in large
    assert "下界才是你能依赖的数" not in large


def test_start_result_states_statistical_limit_upfront(store):
    """抽题结果自带统计局限说明:本轮 N 条即使全部一致,下界也只有约 X%。"""
    from src.workbench.intake_service import sample_evidence_note, wilson_lower_bound

    service, session = store
    pending = service.start_label_verification(session.session_id, session.revision)
    assert pending["evidence_note"] == sample_evidence_note(pending["sample_size"])
    assert "即使全部一致" in pending["evidence_note"]
    assert "下界才是你能依赖的数" in pending["evidence_note"]  # 默认 5 条属于小样本档
    assert (
        f"{wilson_lower_bound(pending['sample_size'], pending['sample_size']):.0%}"
        in pending["evidence_note"]
    )


def test_submit_result_carries_lower_bound_and_note(store):
    """提交结果附 95% 下界与统计说明,并存档回读一致:5/5 全对下界约 57%,不冒充 100%。"""
    from src.workbench.intake_service import wilson_lower_bound

    service, session = store
    pending, answers = _answers_from(service, session)
    size = pending["sample_size"]
    result = service.submit_label_verification(
        session.session_id, pending["verification_id"], answers
    )
    assert result["verdict"] == "verified"
    assert result["agreement"] == 1.0
    assert result["agreement_lower_bound"] == pytest.approx(wilson_lower_bound(size, size))
    assert result["agreement_lower_bound"] < 1.0
    assert "下界才是你能依赖的数" in result["evidence_note"]
    stored = service.load(session.session_id).label_verification
    assert stored["agreement_lower_bound"] == pytest.approx(wilson_lower_bound(size, size))
    assert stored["evidence_note"] == result["evidence_note"]


def test_mismatch_result_lower_bound_reflects_partial_agreement(store):
    """不一致结论同样附下界:4/5 的下界明显低于观测 80%,不粉饰分歧证据强度。"""
    from src.workbench.intake_service import wilson_lower_bound

    service, session = store
    pending, answers = _answers_from(
        service, session, mutate=lambda a, p: a.update({p["items"][0]["row_id"]: "明显不同"})
    )
    size = pending["sample_size"]
    result = service.submit_label_verification(
        session.session_id, pending["verification_id"], answers
    )
    assert result["verdict"] == "insufficient_agreement"
    assert result["matched"] == size - 1
    assert result["agreement_lower_bound"] == pytest.approx(wilson_lower_bound(size - 1, size))
    assert result["agreement_lower_bound"] < result["agreement"]
    assert "样本量小" in result["evidence_note"]
