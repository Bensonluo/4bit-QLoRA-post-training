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
    assert total >= 26, f"内置场景应随 known-gap 清偿持续增长,当前 {total}"
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


def test_spaced_header_names_blocked_at_baseline_analysis(tmp_path):
    """场景 20:表头带前后空格——精确名匹配在基础分析即失败,报错如实列出带空格列名。"""

    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    assert "spaced-header-names" in specs, "缺少场景 spaced-header-names"

    spec = specs["spaced-header-names"]
    # 夹具真实性:表头确实带前后空格,而用户按业务口径选「类别」
    header = spec.sample.split(b"\n")[0].decode()
    assert header != header.strip(), "表头应带前后空格"

    result = run_scenario(spec, tmp_path / spec.scenario_id)
    assert result.verdict == "as_expected", result.to_dict()
    assert result.blocked_at == "baseline_analysis", result.to_dict()
    assert "答案列" in result.blocked_message and "不在数据字段中" in result.blocked_message
    # 报错把带空格的真实列名原样列出,用户能看见差异
    assert " 编号 " in result.blocked_message
    # 入口读表本身不拦空格表头,失败发生在列选择这一步
    assert result.stages["create"] == "passed"


def test_jsonl_long_line_duplicate_header_and_full_width_scenarios(tmp_path):
    """场景 21-23:JSONL 超长行/重复表头(全量验证硬拦)/全角数字——期望结局以实测为准。"""
    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    for scenario_id in ("jsonl-long-line", "full-width-digits"):
        assert scenario_id in specs, f"缺少场景 {scenario_id}"
        result = run_scenario(specs[scenario_id], tmp_path / scenario_id)
        assert result.verdict == "as_expected", result.to_dict()
        assert result.blocked_at is None, result.to_dict()

    # 重复表头行必须被全量验证拦下,不能静默成为训练样本(曾经的已知缺口,已清偿)
    dup_spec = specs["duplicate-header-rows-in-full"]
    dup_result = run_scenario(dup_spec, tmp_path / "duplicate-header-rows-in-full")
    assert dup_result.verdict == "as_expected", dup_result.to_dict()
    assert dup_result.blocked_at == "validate_full", dup_result.to_dict()
    assert "与表头完全相同" in dup_result.blocked_message, dup_result.to_dict()
    # 样例阶段(分析/对比/确认)先通过,拦截发生在全量验证这一关
    assert all(
        dup_result.stages[stage] == "passed"
        for stage in ("create", "baseline_analysis", "contrast_check", "confirm_sample")
    )

    # JSONL 场景确实覆盖「单行数十 KB」量级,且走的是 jsonl 入口而非 CSV
    jsonl_spec = specs["jsonl-long-line"]
    assert jsonl_spec.sample_name.endswith(".jsonl")
    assert jsonl_spec.full_name.endswith(".jsonl")
    assert max(len(line) for line in jsonl_spec.full.split(b"\n")) >= 40_000

    # 重复表头场景的夹具确实在数据中部含一条与表头相同的行
    dup_lines = dup_spec.full.split(b"\n")
    assert dup_lines[0] == "编号,客户描述,类别".encode()
    assert dup_lines[0] in dup_lines[1:], "全量应含重复表头行"

    # 全角数字场景的夹具确实含全角数字
    assert "００１".encode() in specs["full-width-digits"].full


def test_single_row_sample_blocked_at_contrast_check(tmp_path):
    """场景 24:样例只有 1 行数据——配对核验需要 2 条不同答案,在哪一关拦以实测为准。"""
    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    assert "single-row-sample" in specs, "缺少场景 single-row-sample"

    spec = specs["single-row-sample"]
    # 夹具真实性:样例确实只有表头+1 行数据,且该行本身有标签(拦截纯因样例数量,不因缺标签)
    lines = spec.sample.splitlines()
    assert len(lines) == 2, "样例应为表头 + 1 行数据"
    assert not lines[1].endswith(b","), "唯一的数据行应有非空答案"

    result = run_scenario(spec, tmp_path / "single-row-sample")
    assert result.verdict == "as_expected", result.to_dict()
    assert result.blocked_at == "contrast_check", result.to_dict()
    assert "对比核验需要至少两条答案不同的已标注行" in result.blocked_message
    # create 与基础分析都不拦(一条有标签的行足以生成真实预览),拦截发生在配对核验
    assert result.stages["create"] == "passed"
    assert result.stages["baseline_analysis"] == "passed"


def test_all_empty_target_column_blocked_at_contrast_check(tmp_path):
    """场景 25:答案列所有值为空——分析阶段如实观察 0 类答案,旅程在对比核验被拦。"""
    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    assert "all-empty-target-column" in specs, "缺少场景 all-empty-target-column"

    spec = specs["all-empty-target-column"]
    # 夹具真实性:样例与全量每一条数据行的答案字段都为空(表头保留「类别」列名)
    data_lines = spec.sample.splitlines()[1:] + spec.full.splitlines()[1:]
    assert data_lines and all(line.endswith(b",") for line in data_lines), "所有数据行答案应为空"
    assert spec.sample.splitlines()[0].endswith(",类别".encode())

    result = run_scenario(spec, tmp_path / "all-empty-target-column")
    assert result.verdict == "as_expected", result.to_dict()
    assert result.blocked_at == "contrast_check", result.to_dict()
    assert "对比核验需要至少两条答案不同的已标注行" in result.blocked_message
    # 基础分析阶段既不静默通过也不拦:如实生成「0 类答案」的观察与逐行 needs_label 预览
    assert result.stages["create"] == "passed"
    assert result.stages["baseline_analysis"] == "passed"


def test_utf16_no_bom_rejected_at_create(tmp_path):
    """场景 26:无 BOM 的 UTF-16 在入口即被明确拒绝、原始数据未修改;显式指定编码可恢复。"""
    specs = {spec.scenario_id: spec for spec in builtin_scenarios()}
    assert "utf16-no-bom-rejected" in specs, "缺少场景 utf16-no-bom-rejected"

    spec = specs["utf16-no-bom-rejected"]
    # 夹具真实性:确无 BOM,且 utf-8-sig 与 gb18030 两个自动兜底都解不开——
    # 若夹具漂移成碰巧可解码的形态,场景就不再覆盖「入口明确拒绝」这条路
    assert spec.sample[:2] not in (b"\xff\xfe", b"\xfe\xff")
    for fallback in ("utf-8-sig", "gb18030"):
        try:
            spec.sample.decode(fallback)
        except UnicodeDecodeError:
            continue
        raise AssertionError(f"夹具意外可被 {fallback} 解码,拒绝路径将不复存在")
    # 字节本身是合法 UTF-16LE:显式指定编码是真实可用的恢复路径
    assert spec.sample.decode("utf-16-le").startswith("编号")

    result = run_scenario(spec, tmp_path / "utf16-no-bom-rejected")
    assert result.verdict == "as_expected", result.to_dict()
    assert result.blocked_at == "create", result.to_dict()
    assert "无法解码文件" in result.blocked_message
    assert "原始数据未修改" in result.blocked_message
    # 旅程在第一关即停,没有任何后续阶段被记录
    assert set(result.stages) == {"create"}
