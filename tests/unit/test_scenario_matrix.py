"""场景矩阵:干净分类场景应一路通过;脏数据与歧义标签应被对应关卡诚实拦下。"""

from src.workbench.scenario_matrix import ScenarioSpec, run_matrix, run_scenario

CLEAN_SAMPLE = ("编号,客户描述,类别\n001,杯子破损,质量\n002,物流未更新,物流\n").encode()
CLEAN_FULL = (
    "编号,客户描述,类别\n"
    "001,杯子破损,质量\n002,物流未更新,物流\n003,屏幕碎裂,质量\n004,快递丢失,物流\n"
    "005,开不了机,质量\n006,地址填错,物流\n007,异味,质量\n008,延迟送达,物流\n"
    "009,无法充电,质量\n010,包装破损,物流\n"
).encode()


def test_clean_classification_scenario_passes_every_gate(tmp_path):
    spec = ScenarioSpec(
        scenario_id="clean-classification",
        goal="根据客户首次描述判断售后类别",
        sample=CLEAN_SAMPLE,
        sample_name="工单.csv",
        full=CLEAN_FULL,
        target_column="类别",
        group_columns=("编号",),
        expect="passes",
    )
    result = run_scenario(spec, tmp_path)
    assert result.verdict == "as_expected", result.to_dict()
    assert all(stage == "passed" for stage in result.stages.values())


def test_dirty_full_data_is_blocked_at_validation(tmp_path):
    dirty_full = (
        "编号,客户描述\n001,杯子破损\n002,物流未更新\n"  # 缺答案列
    ).encode()
    spec = ScenarioSpec(
        scenario_id="dirty-missing-label",
        goal="根据客户首次描述判断售后类别",
        sample=CLEAN_SAMPLE,
        sample_name="工单.csv",
        full=dirty_full,
        target_column="类别",
        group_columns=("编号",),
        expect="blocked_at:validate_full",
        expect_note="全量缺答案列,必须在全量验证被拦,不能带病进入训练",
    )
    result = run_scenario(spec, tmp_path)
    assert result.verdict == "as_expected", result.to_dict()
    assert result.blocked_at == "validate_full"


def test_blind_mismatch_is_blocked_at_verification(tmp_path):
    """用户盲标与数据标签不一致(标签确有歧义):必须在盲标关被拦。"""

    def contradictory_answers(session):
        answers = {row.row_id: row.target for row in session.full_data.preview.rows}
        first = sorted(answers)[0]
        answers[first] = "和所有标签都不同的答案"
        return answers

    spec = ScenarioSpec(
        scenario_id="ambiguous-labels",
        goal="根据客户首次描述判断售后类别",
        sample=CLEAN_SAMPLE,
        sample_name="工单.csv",
        full=CLEAN_FULL,
        target_column="类别",
        group_columns=("编号",),
        expect="blocked_at:blind_verification",
        user_answers=contradictory_answers,
    )
    result = run_scenario(spec, tmp_path)
    assert result.verdict == "as_expected", result.to_dict()
    assert "盲标不一致" in result.blocked_message


def test_matrix_summary_counts_truthfully(tmp_path):
    specs = [
        ScenarioSpec(
            scenario_id="ok",
            goal="g",
            sample=CLEAN_SAMPLE,
            sample_name="s.csv",
            full=CLEAN_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
        ),
        ScenarioSpec(
            scenario_id="dirty",
            goal="g",
            sample=CLEAN_SAMPLE,
            sample_name="s.csv",
            full="编号,客户描述\n1,x\n".encode(),
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:validate_full",
        ),
    ]
    report = run_matrix(specs, tmp_path)
    assert report["summary"]["total"] == 2
    assert report["summary"]["as_expected"] == 2
    assert "不测模型效果" in report["summary"]["note"]
