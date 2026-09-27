"""场景矩阵:干净分类场景应一路通过;脏数据与歧义标签应被对应关卡诚实拦下。"""

from src.workbench.scenario_matrix import ScenarioSpec, run_matrix, run_scenario
from src.workbench.scenario_specs import builtin_scenarios

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


def test_builtin_new_scenarios_match_expected_verdicts(tmp_path):
    """GBK 编码/宽表/混合类型三个新场景:期望结局以实测为准(全旅程通过)。"""

    wanted = {"gbk-encoded-upload", "wide-table", "mixed-type-column"}
    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    missing = wanted - set(specs)
    assert not missing, f"缺少场景:{sorted(missing)}"

    wide = specs["wide-table"]
    assert len(wide.excluded_columns) >= 28, "宽表场景应把大部分列交给用户排除"

    for scenario_id in sorted(wanted):
        result = run_scenario(specs[scenario_id], tmp_path / scenario_id)
        assert result.verdict == "as_expected", result.to_dict()
        assert result.blocked_at is None, result.to_dict()


def test_builtin_matrix_all_scenarios_as_expected(tmp_path):
    """内置场景全集跑台:无论多少个,全部必须 as_expected(意外=产品缺陷)。"""
    report = run_matrix(builtin_scenarios(), tmp_path)
    total = report["summary"]["total"]
    assert total >= 16, f"内置场景应随 known-gap 清偿持续增长,当前 {total}"
    assert report["summary"]["as_expected"] == total
    assert report["summary"]["unexpected_pass"] == 0
    assert report["summary"]["unexpected_block"] == 0
    assert report["summary"]["error"] == 0


def test_utf16_excel_export_and_ultra_long_single_line_scenarios(tmp_path):
    """场景 17/18:Excel UTF-16 导出自动识别;超长单行边界如实记录。"""
    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    for scenario_id in ("utf16-excel-export", "ultra-long-single-line"):
        assert scenario_id in specs, f"缺少场景 {scenario_id}"
        result = run_scenario(specs[scenario_id], tmp_path / scenario_id)
        assert result.verdict == "as_expected", result.to_dict()
        assert result.blocked_at is None, result.to_dict()

    # 超长单行场景确实覆盖「单行数十 KB」的量级,而不是普通长文本
    long_full = specs["ultra-long-single-line"].full
    assert max(len(line) for line in long_full.split(b"\n")) >= 40_000


def test_empty_label_rows_in_full_blocked_at_validation(tmp_path):
    """场景 19:样例有标签、全量混入空答案行——必须被全量验证硬拦,不能跳过或带病通过。"""

    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    assert "empty-label-rows-in-full" in specs, "缺少场景 empty-label-rows-in-full"

    spec = specs["empty-label-rows-in-full"]
    # 夹具真实性:全量确实混有空标签行,且不止一条
    empty_label_rows = [line for line in spec.full.split(b"\n") if line.endswith(b",")]
    assert len(empty_label_rows) >= 2, "全量应混入若干空标签行"

    result = run_scenario(spec, tmp_path / spec.scenario_id)
    assert result.verdict == "as_expected", result.to_dict()
    assert result.blocked_at == "validate_full", result.to_dict()
    assert "缺少监督答案" in result.blocked_message
    # 样例阶段(分析/对比/确认)全部先通过,拦截发生在全量验证这一关
    assert all(
        result.stages[stage] == "passed"
        for stage in ("create", "baseline_analysis", "contrast_check", "confirm_sample")
    )
