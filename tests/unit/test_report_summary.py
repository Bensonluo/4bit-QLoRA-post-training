"""语言化摘要:非专家读句子,不读表;只复述观察事实,不宣称业务达标。"""

from types import SimpleNamespace

from src.workbench.report_summary import summarize_comparison


def _model(
    label, total, accuracy, statuses=(), outputs=(), prompt="题目:某指令文本较长较长较长较长"
):
    rows = []
    for index in range(total):
        status = statuses[index] if index < len(statuses) else "correct"
        output = outputs[index] if index < len(outputs) else "答案"
        rows.append({"status": status, "output": output, "prompt": prompt})
    return {
        "label": label,
        "metrics": {"total": total, "exact_match": accuracy},
        "rows": rows,
    }


def _report(models):
    return SimpleNamespace(models=models)


def test_zero_score_with_echo_says_no_model_learned_and_why():
    echo = "某指令文本较长较长较长较长" + "继续复述" * 3
    report = _report(
        [
            _model("基座", 2, 0.0, statuses=["truncated", "truncated"], outputs=[echo, echo]),
            _model("本轮微调", 2, 0.0, statuses=["truncated", "truncated"], outputs=[echo, echo]),
        ]
    )
    lines = summarize_comparison(report)
    joined = "\n".join(lines)
    assert "2 道题" in joined
    assert "复述题目" in joined
    assert "没有一个模型答对任何题" in joined
    assert "不是业务达标结论" in joined


def test_partial_and_full_scores_report_best_model_with_limits():
    report = _report(
        [
            _model("基座", 10, 0.3),
            _model("本轮微调", 10, 0.8),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "答对最多的是本轮微调(8/10)" in joined
    assert "10 道题" not in joined.split("注意:")[0].split("答对最多")[-1] or True
    assert "样本很小" not in joined  # 10 题不触发小样本提示


def test_small_sample_warns_and_full_score_stays_humble():
    report = _report([_model("本轮微调", 3, 1.0)])
    joined = "\n".join(summarize_comparison(report))
    assert "全部答对" in joined
    assert "只有 3 道题" in joined
    assert "不能据此断定业务上足够好" in joined


def test_generation_failures_are_named():
    report = _report(
        [
            _model("基座", 3, 0.0, statuses=["failed", "correct", "correct"]),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "1 题生成失败" in joined


def test_preflight_summary_names_truncation_and_blockers():
    from src.workbench.report_summary import summarize_preflight

    lines = summarize_preflight(
        {
            "status": "warnings",
            "splits": {"train": {"rows": 8, "truncated_rows": 2, "answer_lost_rows": 0}},
            "issues": [
                {"severity": "warning", "message": "max_length 超过上下文。"},
                {"severity": "info", "message": "全提示监督说明。"},
            ],
        }
    )
    joined = "\n".join(lines)
    assert "需要你核对的风险" in joined
    assert "train 2 行" in joined
    assert "[warning] max_length" in joined
    assert "不代表训练效果" in joined


def test_preflight_summary_passed_and_empty():
    from src.workbench.report_summary import summarize_preflight

    assert summarize_preflight(None) == ["尚未执行训练前检查。"]
    joined = "\n".join(
        summarize_preflight({"status": "passed", "splits": {"train": {"rows": 4}}, "issues": []})
    )
    assert "检查通过" in joined and "没有内容因长度超限被截断" in joined


def test_training_run_summary_status_and_honesty():
    from src.workbench.report_summary import summarize_training_run

    running = summarize_training_run(
        {
            "status": "running",
            "model_path": "/models/Qwen3-1.7B",
            "config": {"training": {"num_epochs": 1}},
        }
    )
    assert any("正在训练中" in line and "关闭页面不影响" in line for line in running)

    done = summarize_training_run(
        {
            "status": "succeeded",
            "model_path": "/models/Qwen3-1.7B",
            "config": {"training": {"num_epochs": 1}},
            "metrics": {"train_loss": 0.42},
        }
    )
    joined = "\n".join(done)
    assert "训练完成" in joined and "0.4200" in joined
    assert "要用同一套开发题与基座对照" in joined

    failed = summarize_training_run(
        {
            "status": "failed",
            "model_path": "/m",
            "failure": {"stage": "training", "message": "显存不足"},
        }
    )
    joined = "\n".join(failed)
    assert "training阶段：显存不足" in joined


def test_training_run_summary_renders_shared_tool_trace_lines():
    """train-* 尾行:记录带 plan_trace 时以统一格式渲染工具核查轨迹,空/缺省不加轨迹行。"""
    from src.workbench.report_summary import summarize_training_run

    record = {
        "status": "succeeded",
        "model_path": "/models/Qwen3-1.7B",
        "config": {"training": {"num_epochs": 1}},
        "metrics": {"train_loss": 0.42},
        "plan_trace": [
            {"tool": "discover_local_models", "ok": True},
            {"tool": "probe_model", "ok": True},
            {"tool": "preflight_tokenizer", "ok": False, "error": "缺 tokenizer 文件"},
        ],
    }
    lines = summarize_training_run(record)
    joined = "\n".join(lines)
    assert (
        "工具核查轨迹：3 次调用，成功 2 次、失败 1 次——"
        "失败的调用没有取到证据，训练方案只依赖成功的调用。" in joined
    )
    # 轨迹行统一放所有内容行之后(loss 行与对照报告行都在它前面)
    loss_index = next(i for i, line in enumerate(lines) if line.startswith("训练损失"))
    trace_index = next(i for i, line in enumerate(lines) if line.startswith("工具核查轨迹"))
    assert loss_index < trace_index
    assert trace_index == len(lines) - 1

    # 空 plan_trace / 旧记录缺该键:默认行为不变,无轨迹行
    record["plan_trace"] = []
    assert not any("工具核查轨迹" in line for line in summarize_training_run(record))
    del record["plan_trace"]
    assert not any("工具核查轨迹" in line for line in summarize_training_run(record))


def test_preflight_summary_gives_concrete_length_advice():
    from src.workbench.report_summary import summarize_preflight

    lines = summarize_preflight(
        {
            "status": "warnings",
            "max_length": 512,
            "splits": {"train": {"rows": 8, "truncated_rows": 2}},
            "rows": [
                {"full_tokens": 823, "answer_was_truncated": True},
                {"full_tokens": 300, "answer_was_truncated": False},
            ],
            "issues": [],
        }
    )
    joined = "\n".join(lines)
    assert "答案在截断后丢了" in joined
    assert "最长一条记录需要 823" in joined
    assert "设为 832 左右" in joined

    # 无答案丢失且长度已足够:建议核对个别行
    lines = summarize_preflight(
        {
            "status": "warnings",
            "max_length": 1024,
            "splits": {"train": {"rows": 8, "truncated_rows": 1}},
            "rows": [{"full_tokens": 700, "answer_was_truncated": False}],
            "issues": [],
        }
    )
    joined = "\n".join(lines)
    assert "答案都保留了" in joined and "已能容纳最长记录" in joined


def test_all_zero_plugs_failure_cause_counts_into_advice():
    echo = "某指令文本较长较长较长较长" + "继续复述" * 3
    report = _report(
        [
            _model("基座", 2, 0.0, statuses=["truncated", "failed"]),
            _model("本轮微调", 2, 0.0, outputs=[echo, echo]),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "微调后仍是零分" in joined
    assert "继续加数据之前" in joined
    assert "2 题在复述题目" in joined
    assert "1 题没写完被截断" in joined
    assert "1 题生成失败" in joined
    assert "没有带来可见变化" not in joined  # 全零分归入零分态,不再叠无差异结论


def test_all_zero_without_diagnostics_points_to_format_mismatch():
    report = _report(
        [
            _model("基座", 3, 0.0, outputs=["错", "错", "错"]),
            _model("本轮微调", 3, 0.0, outputs=["错", "错", "错"]),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "微调后仍是零分" in joined
    assert "没有观察到截断、生成失败、复述或重复输出" in joined
    assert "答案格式不匹配" in joined
    assert "观察到输出高度重复" not in joined  # 3 行低于最小判定数(4),披露不触发


def test_dominant_output_disclosure_names_model_counts_and_check_direction():
    diverse = [f"答案{i}" for i in range(10)]
    report = _report(
        [
            _model("基座", 10, 0.2, outputs=diverse),
            _model("本轮微调", 10, 0.0, outputs=["无法分类：缺少信息"] * 10),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "观察到输出高度重复" in joined
    assert "本轮微调 有 10/10 条输出完全相同（无法分类：缺少信息）" in joined
    assert "对照开发集答案分布" in joined
    assert "复述多数类" in joined
    assert "不认定原因" in joined
    assert joined.count("条输出完全相同") == 1  # 输出各不相同的基座不被点名


def test_all_zero_with_dominant_output_adds_repeat_cause():
    report = _report(
        [
            _model("基座", 5, 0.0, outputs=[f"不同答案{i}" for i in range(5)]),
            _model("本轮微调", 5, 0.0, outputs=["同一答案"] * 5),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "微调后仍是零分" in joined
    assert "有模型在反复输出同一答案" in joined
    assert "零分更可能来自答案格式不匹配" not in joined  # 已观察到重复输出,不归因格式


def test_dominant_output_disclosed_for_open_tasks_too():
    report = _report([_model("本轮微调", 6, None, outputs=["同一句回答"] * 6)])
    joined = "\n".join(summarize_comparison(report))
    assert "观察到输出高度重复" in joined
    assert "本轮微调 有 6/6 条输出完全相同（同一句回答）" in joined
    assert "生成了 6/6 条回答" in joined


def test_clear_finetune_gain_states_counts_with_sample_caveat():
    report = _report(
        [
            _model("基座", 10, 0.3),
            _model("本轮微调", 10, 0.6),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "本轮微调比基座答对更多(6/10 vs 3/10)" in joined
    assert "要注意样本量" in joined
    assert "逐题核对答错的部分" in joined


def test_no_difference_between_base_and_finetune_is_named():
    report = _report(
        [
            _model("基座", 10, 0.4),
            _model("本轮微调", 10, 0.4),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "微调没有带来可见变化(4/10 vs 4/10)" in joined
    assert "数据量不足或任务难度过高" in joined
    assert "比基座答对更多" not in joined


def test_middling_difference_gets_neither_extreme_claim():
    report = _report(
        [
            _model("基座", 20, 0.2),
            _model("本轮微调", 20, 0.35),
        ]
    )
    joined = "\n".join(summarize_comparison(report))
    assert "比基座答对更多" not in joined
    assert "没有带来可见变化" not in joined
    assert "答对最多的是本轮微调(7/20)" in joined


def test_zero_scores_without_finetuned_model_use_generic_head():
    report = _report([_model("基座", 2, 0.0, statuses=["failed", "failed"])])
    joined = "\n".join(summarize_comparison(report))
    assert "所有模型都是零分" in joined
    assert "微调后仍是零分" not in joined


def test_high_truncation_ratio_hints_max_new_tokens_check():
    """高比例截断时给出 max_new_tokens 核查方向，与回声提示同构、不认定原因。"""
    report = _report(
        [
            _model("基座", 5, 0.2, statuses=["truncated"] * 2 + ["correct"] * 3),
            _model(
                "本轮微调",
                5,
                0.4,
                statuses=["truncated", "truncated", "correct", "correct", "correct"],
            ),
        ]
    )
    lines = summarize_comparison(report)
    joined = "\n".join(lines)
    assert "触及生成长度上限被截断" in joined
    assert "基座 2 题" in joined and "本轮微调 2 题" in joined
    assert "max_new_tokens 是否小于最短合法答案" in joined
    assert "触及上限不等于只需增加长度" in joined
    # 报告未带协议时不编造当前值。
    assert "当前 max_new_tokens" not in joined


def test_high_truncation_ratio_names_current_limit_when_protocol_present():
    report = SimpleNamespace(
        models=[_model("基座", 2, 0.0, statuses=["truncated", "correct"])],
        protocol={"max_new_tokens": 64},
    )
    joined = "\n".join(summarize_comparison(report))
    assert "当前 max_new_tokens 为 64" in joined


def test_low_truncation_ratio_does_not_hint_max_new_tokens():
    report = _report([_model("基座", 10, 0.5, statuses=["truncated"] + ["correct"] * 9)])
    joined = "\n".join(summarize_comparison(report))
    assert "1 题没写完被截断" in joined
    assert "触及生成长度上限被截断" not in joined
    assert "max_new_tokens" not in joined


def test_summarize_dataset_temporal_states_inclusion_and_exclusion_honestly():
    """时间方案摘要:分法、纳入/排除数量、不随机补数,一句不漏也不夸大。"""
    from src.workbench.report_summary import summarize_dataset

    statistics = {
        "total_rows": 6,
        "row_counts": {"train": 3, "validation": 1, "test": 1},
        "split_method": "temporal",
        "included_rows": 5,
        "excluded_rows": 1,
        "exclusion_counts": {"label_not_mature": 1},
    }
    joined = "\n".join(summarize_dataset(statistics))
    assert "按已确认的时间边界划分" in joined
    assert "训练 3 条、验证 1 条、独立测试 1 条" in joined
    assert "共纳入 5 条（全量 6 条）" in joined
    assert "另有 1 条" in joined and "明确排除" in joined
    assert "没有随机补数" in joined and "未成熟标签当作真值" in joined
    assert "分区就绪只说明数据已按规则隔离" in joined


def test_summarize_dataset_temporal_without_exclusions_and_fixed_suite():
    from src.workbench.report_summary import summarize_dataset

    clean = {
        "total_rows": 3,
        "row_counts": {"train": 1, "validation": 1, "test": 1},
        "split_method": "temporal_fixed_evaluation_suite",
        "included_rows": 3,
        "excluded_rows": 0,
    }
    joined = "\n".join(summarize_dataset(clean))
    assert "没有记录被排除" in joined
    assert "固定开发/测试题集" in joined
    assert "新增资料不扩充评分题" in joined


def test_summarize_dataset_grouped_and_fixed_suite_split_methods():
    from src.workbench.report_summary import summarize_dataset

    grouped = {
        "total_rows": 10,
        "row_counts": {"train": 8, "validation": 1, "test": 1},
        "independent_groups": 7,
    }
    joined = "\n".join(summarize_dataset(grouped))
    assert "按业务对象隔离划分" in joined
    assert "7 个独立分组" in joined
    assert "实际比例受分组大小影响" in joined
    fixed = {
        "total_rows": 10,
        "row_counts": {"train": 8, "validation": 1, "test": 1},
        "split_method": "fixed_evaluation_suite",
    }
    joined = "\n".join(summarize_dataset(fixed))
    assert "沿用固定开发/测试题集" in joined
    assert "按业务对象隔离划分" not in joined


def test_summarize_dataset_renders_answer_coverage_note_before_boundary_line():
    """答案覆盖披露进人话摘要:稀有答案落保留分区时点名训练集缺口,位置在边界句前。"""
    from src.workbench.report_summary import summarize_dataset

    note = (
        "验证/测试集中有 1 类答案（screen×1（测试1 条））从未出现在训练集——"
        "训练按逐字学习答案，模型没有学过这些值，验证与测试仍会照常打分。"
    )
    statistics = {
        "total_rows": 18,
        "row_counts": {"train": 14, "validation": 2, "test": 2},
        "independent_groups": 18,
        "answer_counts_by_split": {
            "train": {"yes": 14},
            "validation": {"yes": 2},
            "test": {"yes": 1, "screen": 1},
        },
        "train_missing_answers": {"screen": {"test": 1}},
        "answer_coverage_note": note,
    }
    lines = summarize_dataset(statistics)
    assert lines[-2] == note
    assert lines[-1].startswith("分区就绪只说明")

    # 负例:无覆盖问题或答案取值超过 20 种缺键(如实边界)时,摘要不多说一句
    quiet = summarize_dataset(
        {"total_rows": 3, "row_counts": {"train": 1, "validation": 1, "test": 1}}
    )
    assert not any("从未出现在训练集" in line for line in quiet)


def test_summarize_dataset_renders_duplicate_note_after_coverage_note():
    """完全相同例题披露进人话摘要:渲染在边界句前、覆盖披露之后;无重复时不多说。"""
    from src.workbench.report_summary import summarize_dataset

    duplicate_note = (
        "本版本有 2 条记录与前面的记录渲染后完全相同（输入与答案逐字一致）——"
        "同一道例题会出现多次，训练等效于给这些例题加权；12 条记录去重后只有 "
        "10 道独立例题。没有自动去重，去留由你决定。"
    )
    coverage_note = "验证/测试集中有 1 类答案（screen×1（测试1 条））从未出现在训练集。"
    statistics = {
        "total_rows": 12,
        "row_counts": {"train": 10, "validation": 1, "test": 1},
        "independent_groups": 10,
        "rendered_exact_duplicate_rows": 2,
        "source_exact_duplicate_rows": 1,
        "answer_coverage_note": coverage_note,
        "duplicate_note": duplicate_note,
    }
    lines = summarize_dataset(statistics)
    # 顺序固定:…→ 覆盖披露 → 重复例题披露 → 边界句
    assert lines[-3] == coverage_note
    assert lines[-2] == duplicate_note
    assert lines[-1].startswith("分区就绪只说明")

    # 负例:无重复行(缺键)时摘要不渲染重复披露句
    quiet = summarize_dataset(
        {"total_rows": 3, "row_counts": {"train": 1, "validation": 1, "test": 1}}
    )
    assert not any("渲染后完全相同" in line for line in quiet)


def test_comparison_names_weakest_field_for_json_tasks():
    """JSON 任务逐字段准确率进人话摘要:点名最弱字段与诚实口径;非 JSON 任务沉默。"""
    base = _model("基座", 10, 0.3)
    base["metrics"]["field_accuracy"] = {"日期": 0.2, "类别": 0.9}
    tuned = _model("本轮微调", 10, 0.8)
    tuned["metrics"]["field_accuracy"] = {"日期": 0.5, "类别": 1.0}
    joined = "\n".join(summarize_comparison(_report([base, tuned])))
    assert "基座最弱的是「日期」(2/10 题答对)" in joined
    assert "本轮微调最弱的是「日期」(5/10 题答对)" in joined
    assert "全部字段都对" in joined
    assert "无法按 JSON 解析" in joined

    # 负例:分类/开放任务的 field_accuracy 为空 dict,摘要不多说
    plain = "\n".join(summarize_comparison(_report([_model("基座", 10, 0.3)])))
    assert "最弱的是" not in plain

    # 全部字段全对 → 不点名最弱(交给「全部答对」句)
    perfect = _model("本轮微调", 3, 1.0)
    perfect["metrics"]["field_accuracy"] = {"类别": 1.0}
    joined = "\n".join(summarize_comparison(_report([perfect])))
    assert "最弱的是" not in joined


def test_comparison_custom_scoring_and_open_tasks_are_not_fake_zero():
    """自定义评分对照:通过数与业务分均值如实入句,不渲染「答对 0/10」假零分结论;
    开放任务只报生成事实,不虚构答对。"""
    prompt = "题目:某指令文本较长较长较长较长"

    def custom(label, pass_rate, business_score):
        rows = [{"status": "scored", "output": "答", "prompt": prompt} for _ in range(10)]
        return {
            "label": label,
            "metrics": {
                "total": 10,
                "exact_match": None,
                "business_score": business_score,
                "pass_rate": pass_rate,
            },
            "rows": rows,
        }

    joined = "\n".join(
        summarize_comparison(_report([custom("基座", 0.3, 0.35), custom("本轮微调", 0.8, 0.75)]))
    )
    assert "基座业务评分均值 0.35、通过 3/10" in joined
    assert "本轮微调业务评分均值 0.75、通过 8/10" in joined
    assert "答对 0/10" not in joined
    assert "没有一个模型答对任何题" not in joined
    assert "微调后仍是零分" not in joined
    assert "两者都不是严格准确率" in joined
    assert "逐题查看评分理由" in joined

    # 开放任务:exact_match 与 pass_rate 都缺,摘要只说生成事实
    def open_task(label):
        rows = [
            {"status": "needs_business_review", "output": "开放回答" * 20, "prompt": prompt}
            for _ in range(10)
        ]
        return {
            "label": label,
            "metrics": {"total": 10, "exact_match": None, "field_accuracy": {}},
            "rows": rows,
        }

    joined = "\n".join(summarize_comparison(_report([open_task("基座"), open_task("本轮微调")])))
    assert "基座生成了 10/10 条回答" in joined
    assert "答对" not in joined, "开放任务没有自动评分,不得出现「答对」措辞"
    assert "开放任务不做自动评分" in joined
    assert "不能人工标为通过" in joined


def _acceptance_record(result, status="completed", metric="exact_match"):
    return {
        "status": status,
        "model": {"label": "待验收模型"},
        "criteria": {"metric": metric, "minimum_score": 0.9, "minimum_cases": 5},
        "result": result,
    }


def test_acceptance_summary_final_decisions_state_counts_and_denominator():
    """最终验收摘要:通过/未达到如实入句,分母口径(失败截断按未通过计)与隔离降级不作数。"""
    from src.workbench.report_summary import summarize_acceptance

    passed = _acceptance_record(
        {
            "decision": "passed",
            "metric": "exact_match",
            "score": 0.8,
            "accepted_cases": 8,
            "total_cases": 10,
            "usable_cases": 10,
            "minimum_score": 0.9,
            "minimum_cases": 5,
            "reason": "按运行前冻结的业务标准计算，失败及截断保留在全部题目分母中。",
        }
    )
    lines = summarize_acceptance(passed)
    joined = "\n".join(lines)
    assert "这次最终验收针对模型「待验收模型」" in joined
    assert "运行前冻结的标准是：严格匹配至少 90%，且至少 5 道独立测试题" in joined
    assert "当前结论：达到运行前冻结的验收标准。" in joined
    assert "通过 8/10 题（通过率 80.0%）" in joined
    assert "失败与截断保留在全部题目分母中，按未通过计" in joined
    assert "不会自动部署模型" in joined
    assert "证据不足" not in joined

    failed = _acceptance_record(
        {
            "decision": "failed",
            "metric": "exact_match",
            "score": 0.3,
            "accepted_cases": 3,
            "total_cases": 10,
            "usable_cases": 10,
        }
    )
    joined = "\n".join(summarize_acceptance(failed))
    assert "当前结论：未达到运行前冻结的验收标准。" in joined

    isolated = _acceptance_record(
        {
            "decision": "insufficient_evidence",
            "observed_decision": "passed",
            "metric": "exact_match",
            "score": 0.95,
            "accepted_cases": 19,
            "total_cases": 20,
            "usable_cases": 20,
            "reason": "未能核验适配器训练资料与本次固定留出题的隔离，数值结果不能作为独立业务验收通过。",
        }
    )
    joined = "\n".join(summarize_acceptance(isolated))
    assert "当前结论：证据不足，不能确认可交付。" in joined
    assert "单看通过率本会判为「达到标准」" in joined
    assert "数值不能作为独立业务验收的结论" in joined
    assert "原因：未能核验适配器训练资料与本次固定留出题的隔离" in joined

    custom = _acceptance_record(
        {
            "decision": "passed",
            "metric": "pass_rate",
            "score": 0.9,
            "accepted_cases": 9,
            "total_cases": 10,
            "usable_cases": 10,
            "business_score": 0.62,
        },
        metric="pass_rate",
    )
    joined = "\n".join(summarize_acceptance(custom))
    assert "自定义规则通过率至少 90%" in joined
    assert "业务评分均值 0.62，仅作描述" in joined

    partial_usable = _acceptance_record(
        {
            "decision": "failed",
            "metric": "exact_match",
            "score": 0.2,
            "accepted_cases": 2,
            "total_cases": 10,
            "usable_cases": 7,
        }
    )
    joined = "\n".join(summarize_acceptance(partial_usable))
    assert "其中可完整核查的有效输出只有 7/10 题" in joined


def test_acceptance_summary_pending_blocked_failed_and_bare_states():
    """最终验收摘要:未执行/被阻断/执行失败/待人工判断各态如实;裸 result 不崩溃。"""
    from src.workbench.report_summary import summarize_acceptance

    prepared = _acceptance_record(None, status="prepared")
    prepared.pop("result")
    joined = "\n".join(summarize_acceptance(prepared))
    assert "条款已冻结、验收尚未执行" in joined
    assert "不能再伪装成首次盲测" in joined

    blocked = _acceptance_record(
        {"decision": "insufficient_evidence", "reason": "heldout_already_revealed"},
        status="blocked",
    )
    joined = "\n".join(summarize_acceptance(blocked))
    assert "验收被阻断" in joined
    assert "已经揭示" in joined

    failed_run = _acceptance_record(
        {"decision": "insufficient_evidence", "reason": "模型释放失败"}, status="failed"
    )
    joined = "\n".join(summarize_acceptance(failed_run))
    assert "这次验收没有完整执行：模型释放失败" in joined

    pending = _acceptance_record(
        {"decision": "pending_review", "total_cases": 10, "reviewed_cases": 3, "score": None},
        status="needs_business_review",
    )
    joined = "\n".join(summarize_acceptance(pending))
    assert "等待逐题业务判断" in joined
    assert "已记录 3/10 题判断" in joined
    assert "不能人工改标为通过" in joined

    bare = _acceptance_record({"decision": "insufficient_evidence"})
    lines = summarize_acceptance(bare)
    joined = "\n".join(lines)
    assert "当前结论：证据不足，不能确认可交付。" in joined
    assert "原因：" not in joined, "裸 result 没有 reason,不得编造原因行"
    assert "通过 " not in joined, "裸 result 没有计数,不得编造通过数"


def test_iteration_summary_states_and_decision_echo():
    """轮次摘要:决定回显与业务理由如实入句,采用补不自动部署边界;未决状态只报流程停点。"""
    from src.workbench.report_summary import summarize_iteration

    decided = summarize_iteration(
        {
            "iteration_id": "it-x",
            "hypothesis": "已有监督覆盖不足可能影响类别判断",
            "status": "decided",
            "decision": "adopt",
            "decision_reason": "固定开发题错误减少",
        }
    )
    joined = "\n".join(decided)
    assert "这轮改进的假设是：已有监督覆盖不足可能影响类别判断。" in joined
    assert "已记录你的业务决定：采用本轮结果。" in joined
    assert "业务理由：固定开发题错误减少" in joined
    assert "不会自动部署模型" in joined
    assert "不代表业务效果达标" in joined

    insufficient = summarize_iteration({"status": "decided", "decision": "insufficient_evidence"})
    joined = "\n".join(insufficient)
    assert "已记录你的业务决定：证据不足。" in joined
    assert "保留证据局限" in joined
    assert "业务理由：" not in joined, "没有理由不得编造业务理由行"
    assert "不会自动部署模型" not in joined, "不自动部署边界只对采用记录补充"

    evaluated = summarize_iteration({"status": "evaluated"})
    joined = "\n".join(evaluated)
    assert "三模型同题对照已完成" in joined
    assert "等待你的业务决定" in joined

    proposed = summarize_iteration({"status": "proposed"})
    joined = "\n".join(proposed)
    assert "提案已保存，尚未确认" in joined
    assert "确认假设与变更范围后才会准备训练" in joined

    blocked = summarize_iteration({"status": "blocked", "failure": "父轮模型或基座已改变"})
    joined = "\n".join(blocked)
    assert "本轮训练准备被阻断：父轮模型或基座已改变。" in joined

    bare = summarize_iteration({})
    joined = "\n".join(bare)
    assert "这轮改进提案。" in joined, "裸记录没有假设,按提案口径如实降级"
    assert "不代表业务效果达标" in joined


def test_execution_summary_state_machine_translates_user_action_states():
    """自动执行摘要:进行中/暂停等确认/完成待决定/终态各如实;等确认必须点名用户动作。"""
    from src.workbench.report_summary import summarize_execution

    training = summarize_execution({"status": "training"})
    joined = "\n".join(training)
    assert "训练已按确认方案启动" in joined
    assert "关闭页面不影响执行" in joined

    ack = summarize_execution({"status": "awaiting_warning_ack"})
    joined = "\n".join(ack)
    assert "自动执行已暂停" in joined
    assert "勾选确认继续才会恢复" in joined
    assert "不会跳过提示自动训练" in joined

    done = summarize_execution(
        {"status": "completed", "run_id": "wb-run-x", "evaluation_id": "ev-x"}
    )
    joined = "\n".join(done)
    assert "三模型同题对照报告已生成" in joined
    assert "训练 wb-run-x" in joined and "评测 ev-x" in joined
    assert "停在待业务决定" in joined
    assert "重复提交不会再次训练" in joined

    blocked = summarize_execution(
        {
            "status": "blocked",
            "message": "训练准备未通过，请查看问题后处理。",
            "issues": ["数据版本过期", "预检阻断"],
        }
    )
    joined = "\n".join(blocked)
    assert "自动执行被阻断：训练准备未通过，请查看问题后处理。" in joined
    assert "问题：数据版本过期；预检阻断。" in joined
    assert "提出新的改进轮次" in joined

    failed = summarize_execution({"status": "failed"})
    joined = "\n".join(failed)
    assert "自动执行失败。" in joined, "没有 message 时如实降级,不得编造原因"
    assert "worker.log" in joined

    stopped = summarize_execution({"status": "stopped"})
    joined = "\n".join(stopped)
    assert "已按请求停止自动执行" in joined
    assert "不代表业务效果达标" in joined


def test_plan_summary_ready_states_model_params_reasons_and_boundaries():
    """ready 方案摘要:模型+关键参数+状态+理由/限制原文+确认不自动启动边界。"""
    from src.workbench.report_summary import summarize_plan

    record = {
        "status": "ready",
        "proposal": {
            "model_path": "/models/qwen-base",
            "max_length": 1024,
            "rationale": ["任务量级适合小参数模型", "本机显存可容纳 4-bit"],
            "limitations": ["未在业务留出题上验证"],
            "business_questions": [],
            "training_options": {"num_epochs": 2, "batch_size": 1, "learning_rate": 0.0002},
            "lora_options": {"r": 16},
            "model_options": {"quantization_bits": 4},
        },
    }
    lines = summarize_plan(record)
    joined = "\n".join(lines)
    head = lines[0]
    assert head.startswith("这份方案建议用 qwen-base（")
    assert "最大长度 1024" in head and "训练 2 轮" in head and "batch size 1" in head
    assert "学习率 0.0002" in head and "LoRA rank 16" in head and "4-bit 量化" in head
    assert "当前状态：方案可供确认。" in lines
    assert "推荐理由：任务量级适合小参数模型；本机显存可容纳 4-bit" in lines
    assert "尚未验证的限制：未在业务留出题上验证" in lines
    assert "确认这份方案只会准备训练，不会自动启动" in joined
    assert "不构成训练效果或业务达标的判断" in joined


def test_plan_summary_needs_data_questions_and_bare_records_do_not_invent():
    """needs_data 点名待答业务问题;无状态/无模型的裸记录如实降级不编造。"""
    from src.workbench.report_summary import summarize_plan

    record = {
        "status": "needs_data",
        "proposal": {
            "model_path": "/tmp/x",
            "rationale": ["样例不足"],
            "limitations": ["无法预检"],
            "business_questions": ["留存答案口径是哪个字段？"],
        },
    }
    lines = summarize_plan(record)
    joined = "\n".join(lines)
    assert "当前状态：需要先完善数据。" in lines
    assert "还有需要你先回答的业务问题：留存答案口径是哪个字段？" in joined
    assert "回答确认前不能准备训练" in joined
    assert "先完善数据或回答业务问题" in joined

    unsupported = summarize_plan({"status": "unsupported", "proposal": {"model_path": "/tmp/y"}})
    joined = "\n".join(unsupported)
    assert "当前状态：当前条件不支持。" in joined
    assert "换用支持的模型或机器后重新生成方案" in joined

    bare = summarize_plan({})
    assert "这份方案还没有选择基础模型。" in bare[0]
    assert len(bare) == 2, "裸记录只剩模型缺位句+固定边界句,不得编造状态"
    assert "不构成训练效果或业务达标的判断" in bare[-1]


def test_plan_summary_renders_shared_tool_trace_lines():
    """plan-* 尾行:记录带 trace 时以统一格式渲染工具核查轨迹,空/缺省不加轨迹行。"""
    from src.workbench.report_summary import summarize_plan

    record = {
        "status": "ready",
        "proposal": {
            "model_path": "/models/qwen-base",
            "rationale": ["样例结构稳定"],
            "limitations": ["未在业务留出题上验证"],
        },
        "trace": [
            {"tool": "discover_local_models", "ok": True},
            {"tool": "probe_model", "ok": False, "error": "缺 tokenizer 文件"},
        ],
    }
    lines = summarize_plan(record)
    joined = "\n".join(lines)
    assert (
        "工具核查轨迹：2 次调用，成功 1 次、失败 1 次——"
        "失败的调用没有取到证据，方案只依赖成功的调用。" in joined
    )
    boundary_index = next(i for i, line in enumerate(lines) if line.startswith("方案就绪与推荐理由"))
    trace_index = next(i for i, line in enumerate(lines) if line.startswith("工具核查轨迹"))
    assert trace_index < boundary_index, "轨迹行必须在收尾边界句之前"

    # 空 trace / 缺省:默认行为不变,无轨迹行
    record["trace"] = []
    assert not any("工具核查轨迹" in line for line in summarize_plan(record))
    del record["trace"]
    assert not any("工具核查轨迹" in line for line in summarize_plan(record))


def test_scoring_summary_draft_and_confirmed_states_with_boundaries():
    """评分规则摘要:标准+通过线+正反例验证+确认/待确认边界+不构成达标判断。"""
    from src.workbench.report_summary import summarize_scoring

    record = {
        "status": "draft",
        "recipe": {
            "business_standard": "回答须含全部必要处理步骤",
            "pass_threshold": 0.8,
            "examples": [
                {"name": "完整步骤", "kind": "business"},
                {"name": "缺项反例", "kind": "counterexample"},
            ],
        },
        "validation": {"status": "passed", "backend": "fixture-os"},
    }
    lines = summarize_scoring(record)
    joined = "\n".join(lines)
    assert lines[0] == "这套规则要判断的业务标准：回答须含全部必要处理步骤。"
    assert "单题得分达到 0.8 才计为通过；均分与通过率分开展示。" in lines
    assert "用 1 条业务正例与 1 条同题反例在真实隔离后端验证。" in lines
    assert "隔离验证：passed（后端：fixture-os）。" in lines
    assert "草稿待你核对实际正反例分数与理由后确认；软件不会自动确认评分规则。" in lines
    assert "不等于严格准确率，也不构成业务达标的判断" in joined

    record["status"] = "confirmed"
    confirmed = summarize_scoring(record)
    joined = "\n".join(confirmed)
    assert "已确认：规则绑定当前业务目标与输入/答案语义" in joined
    assert "数据修订后兼容规则可继续用" in joined
    assert "业务目标或含义变更需重新确认" in joined
    assert "软件不会自动确认" not in joined


def test_scoring_summary_needs_business_input_and_bare_records_do_not_invent():
    """needs_business_input 摘要:原因+待补业务问题+没有可确认方案;裸记录不编造。"""
    from src.workbench.report_summary import summarize_scoring

    record = {
        "status": "needs_business_input",
        "reason": "专业程度需要明确可判定要求",
        "questions": ["哪些关键步骤不可缺少？", "缺少时如何扣分？"],
    }
    lines = summarize_scoring(record)
    joined = "\n".join(lines)
    assert lines[0] == "业务评分标准还不能转成可执行的规则。"
    assert "原因：专业程度需要明确可判定要求" in joined
    assert "需要你先补充的业务问题：哪些关键步骤不可缺少？；缺少时如何扣分？" in joined
    assert "当前没有可确认的评分方案" in joined
    assert "业务达标" not in joined, "澄清态没有规则与验证事实,不得带达标边界句"

    bare = summarize_scoring({})
    assert bare[0] == "这套规则没有写明业务标准。"
    assert "软件不会自动确认评分规则" in "\n".join(bare)
    assert "不等于严格准确率" in bare[-1]


def test_scoring_summary_renders_shared_tool_trace_lines():
    """scoring-* 尾行:记录带 trace 时以统一格式渲染工具核查轨迹,空/缺省不加轨迹行。"""
    from src.workbench.report_summary import summarize_scoring

    record = {
        "status": "draft",
        "recipe": {"business_standard": "须含全部处理步骤", "pass_threshold": 0.8},
        "trace": [
            {"tool": "profile_data", "ok": True},
            {"tool": "inspect_rows", "ok": True},
            {"tool": "read_cell_content", "ok": False, "error": "行不存在"},
        ],
    }
    lines = summarize_scoring(record)
    joined = "\n".join(lines)
    assert (
        "工具核查轨迹：3 次调用，成功 2 次、失败 1 次——"
        "失败的调用没有取到证据，评分只依赖成功的调用。" in joined
    )
    boundary_index = next(i for i, line in enumerate(lines) if line.startswith("业务评分均值"))
    trace_index = next(i for i, line in enumerate(lines) if line.startswith("工具核查轨迹"))
    assert trace_index < boundary_index, "轨迹行必须在收尾边界句之前"

    # needs_business_input 记录同样落盘 trace(src/agent/scoring.py),澄清态也带轨迹行
    clarifying = summarize_scoring(
        {
            "status": "needs_business_input",
            "reason": "标准不可判定",
            "trace": [{"tool": "profile_data", "ok": True}, {"tool": "read_cell", "ok": False}],
        }
    )
    assert "工具核查轨迹：2 次调用，成功 1 次、失败 1 次" in "\n".join(clarifying)
    assert "评分只依赖成功的调用" in "\n".join(clarifying)

    # 空 trace / 缺省:默认行为不变,无轨迹行
    record["trace"] = []
    assert not any("工具核查轨迹" in line for line in summarize_scoring(record))
    del record["trace"]
    assert not any("工具核查轨迹" in line for line in summarize_scoring(record))


def test_suite_summary_counts_lock_scope_and_boundary():
    """固定题集摘要:引用与完整清单两形态都报题数+锁定+不扩充+比较基线边界。"""
    from src.workbench.report_summary import summarize_suite

    reference = {
        "suite_id": "a" * 64,
        "case_counts": {"validation": 12, "test": 5},
        "cases_digest": "b" * 64,
    }
    lines = summarize_suite(reference)
    joined = "\n".join(lines)
    assert lines[0] == "这套固定题集含开发题 12 道、最终测试题 5 道，供后续各轮用同一套题比较。"
    assert "题目内容已按摘要 bbbbbbbbbbbb… 锁定，原评分题不能修改。" in lines
    assert "同对象新增行保留在对应分区，不自动扩充评分题" in joined
    assert "锚定数据版本" not in joined, "引用形态没有锚定信息,不得编造"
    assert "--suite-id 指定即可复用" in joined
    assert lines[-1] == "固定题集只保证各轮比较基线一致，不代表业务效果达标。"

    manifest = {
        "suite_id": "a" * 64,
        "case_counts": {"validation": 12, "test": 5},
        "cases_digest": "b" * 64,
        "scope_note": "原开发/最终测试评分题固定；同对象新增行保留在对应分区但不自动扩充评分题。",
        "anchor_dataset": {"name": "my-domain", "version": 3},
    }
    lines = summarize_suite(manifest)
    joined = "\n".join(lines)
    assert "原开发/最终测试评分题固定；同对象新增行保留在对应分区但不自动扩充评分题。" in lines
    assert "锚定数据版本：my-domain（version 3）。" in lines

    bare = summarize_suite({})
    assert bare[0] == "这套固定题集没有记录题数。"
    assert bare[-1] == "固定题集只保证各轮比较基线一致，不代表业务效果达标。"
    assert "锁定" not in "\n".join(bare), "裸记录没有摘要值,不得编造锁定句"


def test_assessment_summary_evidence_hypotheses_and_boundaries():
    """Agent 解读摘要:观察/假设分离+决策名+轨迹计数(含失败)+不自动执行边界。"""
    from src.workbench.report_summary import summarize_assessment

    record = {
        "model": "tool-fixture",
        "assessment": {
            "summary": "基座在日期字段上系统性缺漏。",
            "observations": [
                {"statement": "日期字段 10 题错 8 题", "evidence_ids": ["case:1"]},
                {"statement": "截断占比低", "evidence_ids": ["case:2"]},
            ],
            "hypotheses": [
                {
                    "statement": "日期格式在训练集中分布不足",
                    "verification": "统计训练集日期字段的取值分布",
                    "evidence_ids": ["case:1"],
                }
            ],
            "next_steps": ["核对日期字段的监督覆盖", "补充日期样例后重训对照"],
            "decision": "inspect_data",
            "limitations": ["开发集只有 10 题，样本量小"],
            "business_questions": ["日期字段的真实业务口径是哪个？"],
        },
        "tool_trace": [
            {"tool": "inspect_evaluation_summary", "ok": True},
            {"tool": "inspect_bad_cases", "ok": True},
            {"tool": "inspect_case_content", "ok": False, "error": "参数错误"},
            {"tool": "submit_evaluation_assessment", "ok": True},
        ],
    }
    lines = summarize_assessment(record)
    joined = "\n".join(lines)
    assert lines[0] == "这份解读由 Agent 在核查真实工具证据后给出（模型 tool-fixture）。"
    assert "总述：基座在日期字段上系统性缺漏。" in lines
    assert "有证据的观察 2 条、待核查原因 1 条——这是假设不是事实，每条附验证方式" in joined
    assert "共引用 2 处工具证据" in joined, "evidence_ids 去重计数(1 条假设复用 case:1)"
    assert "建议优先处理：先核查数据。" in lines
    assert "建议下一步：核对日期字段的监督覆盖；补充日期样例后重训对照" in lines
    assert "解读自己声明的局限：开发集只有 10 题，样本量小" in lines
    assert "需要你先回答的业务问题：日期字段的真实业务口径是哪个？" in lines
    assert "工具核查轨迹：4 次调用，成功 3 次、失败 1 次" in joined
    assert "失败的调用没有取到证据" in joined
    assert lines[-1].startswith(
        "以上是开发集诊断建议：软件不会据此自动改标签、删除坏例或采纳方案变更"
    )
    assert "不代表业务效果达标" in lines[-1]

    # 裸记录:不编造观察/决策/轨迹,只留缺位句+边界句
    bare = summarize_assessment({})
    assert bare[0] == "这份解读没有可读的内容。"
    assert len(bare) == 2
    assert "不代表业务效果达标" in bare[-1]


def test_full_report_summary_states_issues_and_boundary():
    """全量验证报告摘要:来源+结论四态+逐条问题原文(阻断在前)+转换计数+边界句。"""
    from src.workbench.report_summary import summarize_full_report

    record = {
        "source": {"name": "main-full", "rows": [{}, {}, {}], "digest": "a" * 64},
        "status": "needs_revision",
        "issues": [
            {
                "code": "repeated_header_rows",
                "severity": "review",
                "message": "1 条记录与表头完全相同（通常是导出拼接产生的重复表头行）",
                "row_ids": ["r000002"],
            },
            {
                "code": "missing_required",
                "severity": "blocking",
                "message": "全量文件缺少当前方案必需字段：类别",
                "row_ids": ["r000001", "r000003"],
            },
        ],
        "preview": {"counts": {"ready": 3, "needs_label": 1, "invalid": 0, "conflict": 0}},
    }
    lines = summarize_full_report(record)
    joined = "\n".join(lines)
    assert (
        lines[0]
        == "这份全量验证针对资料 main-full（3 条、摘要 aaaaaaaaaaaa…），报告中的行 ID 仅属于这份全量文件。"
    )
    assert "当前结论：存在阻断问题，需先按下面的问题修正资料或业务规则，再重新验证全量。" in lines
    # 阻断在前,逐条渲染报告原文并附证据行条数;零计数态不渲染
    blocking_index = next(i for i, line in enumerate(lines) if line.startswith("[阻断]"))
    review_index = next(i for i, line in enumerate(lines) if line.startswith("[需核对]"))
    assert blocking_index < review_index
    assert "[阻断] 全量文件缺少当前方案必需字段：类别（涉及 2 条证据行）" in lines
    assert "全量真实转换：已生成预览 3 条、缺少答案 1 条。" in lines
    assert lines[-1] == "全量验证只核对数据事实与已确认方案的一致性，不代表模型效果或业务达标。"

    record["status"] = "confirmed"
    joined = "\n".join(summarize_full_report(record))
    assert "当前结论：全量数据含义已确认，可以准备生成分区；尚未开始训练。" in joined

    record["status"] = "stale"
    joined = "\n".join(summarize_full_report(record))
    assert "业务理解或方案已变化，这份全量报告已失效" in joined

    bare = summarize_full_report({})
    assert bare[0] == "这份全量报告没有可读的内容。"
    assert len(bare) == 2
    assert "不代表模型效果或业务达标" in bare[-1]


def test_listing_summary_empty_names_entry_non_empty_counts():
    """清单尾行:空清单点名下一步入口(分不清「还没有」和「查错了任务」),非空只给计数。"""
    from src.workbench.report_summary import summarize_listing

    empty = summarize_listing("已保存的训练方案", [], "先运行 plan-recommend 让 Agent 推荐方案。")
    assert empty == ["当前任务还没有已保存的训练方案；先运行 plan-recommend 让 Agent 推荐方案。"]

    counted = summarize_listing("已保存的训练方案", [{}, {}, {}], "先运行 plan-recommend。")
    assert counted == ["共 3 条已保存的训练方案。"]


def test_model_discovery_summary_empty_guidance_and_breakdown():
    """model-list 尾行:空态点名准备入口,非空分档计数+页面同源边界句。"""
    from src.workbench.report_summary import summarize_model_discovery

    empty = summarize_model_discovery([])
    assert len(empty) == 1
    assert empty[0].startswith("暂未发现本地候选模型。")
    assert "--root" in empty[0]

    mixed = summarize_model_discovery(
        [
            {"status": "available"},
            {"status": "incomplete", "issues": ["a.safetensors: 文件缺失、下载未完成或内容为空。"]},
        ]
    )
    assert (
        mixed[0]
        == "发现 2 个本地模型：1 个文件完整、1 个文件不完整（缺什么看 JSON 里的 issues 字段）。"
    )
    # 边界句与页面候选模型区逐字同源(文件完整≠兼容或能训练)
    assert (
        mixed[1]
        == "文件完整只表示可以进一步检查；模型是否兼容、训练长度和机器是否适合，仍由方案检查判断。"
    )

    complete = summarize_model_discovery([{"status": "available"}, {"status": "available"}])
    assert complete[0] == "发现 2 个本地模型：2 个文件完整、0 个文件不完整。"
    assert complete[1] == mixed[1]


def test_analysis_summary_translates_findings_questions_and_gaps():
    """analyze 尾行:发现/待确认问题/能力缺口/微调思路与页面同词汇,裸记录只给缺位句。"""
    from src.workbench.report_summary import summarize_analysis

    record = {
        "findings": [
            {
                "kind": "observed",
                "message": "不同问题可能都通过补发处理，处理结果不是问题类别。",
                "evidence_row_ids": ["r000001", "r000002"],
            }
        ],
        "questions": [
            {
                "question": "处理结果是否需要作为输入特征？",
                "why": "同一处理结果对应多种问题类型",
                "options": ["是", "否"],
            }
        ],
        "capability_gaps": ["缺少问题类别的历史人工标注"],
        "training_approach": "确认标签后可考虑 SFT，规模等待全量检查。",
        "next_steps": ["核对样例转换，随后提供全量数据。"],
    }
    lines = summarize_analysis(record)
    assert lines[0] == "这份分析给出数据判断与待确认问题：发现 1 条、待确认问题 1 个。"
    # message 与（证据：之间有一个空格(f-string 拼接后 rstrip 只削行尾,不削内部)
    assert (
        lines[1]
        == "已观察：不同问题可能都通过补发处理，处理结果不是问题类别。 （证据：r000001, r000002）"
    )
    assert (
        lines[2]
        == "待确认问题：处理结果是否需要作为输入特征？——同一处理结果对应多种问题类型（可选解释：是 / 否）"
    )
    assert lines[3] == "当前能力缺口：缺少问题类别的历史人工标注"
    assert lines[4] == "暂定微调思路：确认标签后可考虑 SFT，规模等待全量检查。"
    assert lines[5] == "下一步：核对样例转换，随后提供全量数据。"
    assert (
        lines[-1]
        == "以上发现中「已观察」是数据里的事实，其余是待确认的推断或业务解释；"
        "分析待你确认并经真实预览核对，不代表业务效果达标。"
    )

    # 其余 kind 同样按页面词汇翻译;无证据行的发现经 rstrip 收尾不带尾空格。
    variant = summarize_analysis(
        {"findings": [{"kind": "needs_full_data", "message": "标签变体需全量核对。"}]}
    )
    assert variant[0] == "这份分析给出数据判断与待确认问题：发现 1 条、待确认问题 0 个。"
    assert variant[1] == "需要全量验证：标签变体需全量核对。"

    bare = summarize_analysis({})
    assert len(bare) == 2
    assert bare[0] == "这份分析没有可读的内容。"
    assert bare[1] == lines[-1]


def test_analysis_summary_renders_shared_tool_trace_lines():
    """analyze 尾行:传入 tool_trace 时以评测解读同一格式渲染,空/None 不加轨迹行。"""
    from src.workbench.report_summary import summarize_analysis

    record = {"findings": [{"kind": "observed", "message": "类别分布不均衡。"}]}
    trace = [
        {"tool": "profile_data", "ok": True},
        {"tool": "inspect_rows", "ok": True},
        {"tool": "read_cell_content", "ok": False, "error": "行不存在"},
    ]
    lines = summarize_analysis(record, trace)
    joined = "\n".join(lines)
    assert (
        "工具核查轨迹：3 次调用，成功 2 次、失败 1 次——"
        "失败的调用没有取到证据，分析只依赖成功的调用。" in joined
    )
    boundary_index = next(i for i, line in enumerate(lines) if line.startswith("以上发现中"))
    trace_index = next(i for i, line in enumerate(lines) if line.startswith("工具核查轨迹"))
    assert trace_index < boundary_index, "轨迹行必须在边界句之前"

    # 全成功:只计数,不渲染失败子句
    all_ok = summarize_analysis(
        record, [{"tool": "profile_data", "ok": True}, {"tool": "inspect_rows", "ok": True}]
    )
    assert "工具核查轨迹：2 次调用，成功 2 次。" in all_ok
    assert not any("失败" in line for line in all_ok if line.startswith("工具核查轨迹"))

    # 空 trace / None:默认行为不变,无轨迹行
    assert not any("工具核查轨迹" in line for line in summarize_analysis(record, []))
    assert not any("工具核查轨迹" in line for line in summarize_analysis(record))


def test_tool_trace_summary_subject_names_the_reader():
    """单一来源轨迹行:subject 点名当前读者(解读/分析),空轨迹返回空清单。"""
    from src.workbench.report_summary import summarize_tool_trace

    trace = [
        {"tool": "profile_data", "ok": True},
        {"tool": "inspect_rows", "ok": False, "error": "行不存在"},
    ]
    assert summarize_tool_trace(trace, "解读") == [
        "工具核查轨迹：2 次调用，成功 1 次、失败 1 次——"
        "失败的调用没有取到证据，解读只依赖成功的调用。"
    ]
    assert summarize_tool_trace(trace, "分析") == [
        "工具核查轨迹：2 次调用，成功 1 次、失败 1 次——"
        "失败的调用没有取到证据，分析只依赖成功的调用。"
    ]
    assert summarize_tool_trace(None) == []
    assert summarize_tool_trace([]) == []


def test_registration_summary_covers_all_states():
    """注册状态摘要:已注册点名版本与别名、未注册复述命令原文、失败如实、裸记录缺位。"""
    from src.workbench.report_summary import summarize_registration

    registered = {
        "run_id": "wb-1",
        "status": "registered",
        "versions": [
            {
                "name": "工单分类",
                "version": 3,
                "aliases": ["champion"],
                "current_stage": "Production",
            },
            {"name": "工单分类", "version": 2, "aliases": [], "current_stage": "None"},
        ],
    }
    lines = summarize_registration(registered)
    assert lines[0] == "这次训练已注册到模型库：工单分类 v3（champion）、工单分类 v2。"
    assert lines[1] == "注册只说明模型库记录了这次训练的产物与血缘，不代表业务效果达标。"

    not_registered = {
        "run_id": "wb-1",
        "status": "not_registered",
        "message": "这次训练尚未注册到模型库。",
        "how_to_register": "先合并导出为独立模型,再注册(带血缘旗标):\npython scripts/merge_adapter.py",
    }
    lines = summarize_registration(not_registered)
    assert lines[0] == "这次训练尚未注册到模型库。"
    assert "python scripts/merge_adapter.py" in lines
    assert lines[-1] == "注册只说明模型库记录了这次训练的产物与血缘，不代表业务效果达标。"

    failed = {"run_id": "wb-1", "status": "lookup_failed", "message": "模型库查询失败:连不上"}
    assert summarize_registration(failed) == ["模型库查询失败:连不上"]
    unavailable = {
        "run_id": "wb-1",
        "status": "mlflow_unavailable",
        "message": "未安装 mlflow,无法查询模型库。",
    }
    assert summarize_registration(unavailable) == ["未安装 mlflow,无法查询模型库。"]
    # 失败态没有 message 时不编造,只如实点名状态;裸记录给缺位句。
    bare_failed = {"run_id": "wb-1", "status": "lookup_failed"}
    assert summarize_registration(bare_failed) == ["模型库查询返回状态 lookup_failed，没有更多说明。"]
    assert summarize_registration({}) == ["这份注册状态没有可读的内容。"]


def test_lineage_summary_dash_fallbacks_and_external_states():
    """反向血缘摘要:workbench 五要素缺项显「-」不显 None、外部/无来源如实、mlflow 缺位不编造。"""
    from src.workbench.report_summary import summarize_lineage

    workbench = {
        "model": "工单分类 v3",
        "status": "workbench",
        "workbench_run_id": "wb-9",
        "dataset_version": None,
        "config_digest": "abcdef1234567890",
        "training_dataset": None,
        "metrics": {},
    }
    lines = summarize_lineage(workbench)
    assert lines[0] == "模型：工单分类 v3"
    assert lines[1] == "训练运行：wb-9"
    assert lines[2] == "数据版本：-"
    assert lines[3] == "训练数据：-"
    assert lines[4] == "配置摘要：abcdef123456…"
    assert (
        lines[5] == "以上血缘把模型、训练运行与数据版本关联起来，只保证可追溯，不代表业务效果达标。"
    )

    external = {
        "model": "旧模型 v1",
        "status": "external",
        "mlflow_run_id": "mfr-1",
        "base_model": "Qwen/Qwen3-1.7B",
        "message": "该版本来自旧体系或其他训练入口;血缘以 MLflow 参数为准。",
    }
    lines = summarize_lineage(external)
    assert lines[0] == "模型：旧模型 v1"
    assert lines[1] == "基座模型：Qwen/Qwen3-1.7B"
    assert lines[-1] == "该版本来自旧体系或其他训练入口;血缘以 MLflow 参数为准。"

    no_run = {
        "model": "手工 v1",
        "status": "no_source_run",
        "message": "该版本没有关联的训练运行记录（可能是手工注册的目录）。",
    }
    assert summarize_lineage(no_run) == [
        "模型：手工 v1",
        "该版本没有关联的训练运行记录（可能是手工注册的目录）。",
    ]
    # mlflow_unavailable 态服务端不带 message:如实点名状态含义,不 .get 编造。
    unavailable = {"model": "X v1", "status": "mlflow_unavailable"}
    assert summarize_lineage(unavailable) == ["模型：X v1", "未安装 mlflow，无法查询这份血缘。"]
    assert summarize_lineage({}) == ["这份血缘记录没有可读的内容。"]


def test_export_summary_covers_all_states():
    """summarize_export 六态:exported/already/ready/blocked/unknown/empty,边界句固定收尾。"""
    from src.workbench.report_summary import summarize_export

    exported = {
        "status": "exported",
        "output_dir": "/out/wb-1",
        "run_id": "wb-1",
        "dataset_version": "ds-v3",
    }
    lines = summarize_export(exported)
    assert lines[0] == "合并导出完成：这次训练的适配器已并入基础模型，输出目录 /out/wb-1。"
    assert "可被 vLLM、Ollama、LM Studio 直接加载" in lines[1]
    assert "export_evidence.json" in lines[1]
    assert lines[-1] == "导出只产出模型文件与证据记录，不代表业务效果达标，也不会自动部署。"

    already = summarize_export({"status": "already_exported", "output_dir": "/out/wb-1"})
    assert already[0] == (
        "这次训练此前已合并导出到 /out/wb-1；目录已完整，重复导出不会改变模型内容。"
    )
    assert already[-1].startswith("导出只产出")

    ready = summarize_export({"status": "ready", "run_id": "wb-9", "output_dir": "/m/wb-9"})
    assert any("python scripts/data_intake.py train-export wb-9" in line for line in ready)
    assert "默认输出目录：/m/wb-9。" in ready
    assert ready[-1].startswith("导出只产出")

    blocked = summarize_export(
        {"status": "adapter_missing", "reasons": ["训练记录为成功，但产物目录缺少完整的 adapter 文件：/x"]}
    )
    assert blocked[0].startswith("训练记录为成功，但产物目录缺少")
    assert blocked[-1].startswith("导出只产出")

    # 无 reasons 的未知态只显状态,不编造原因。
    assert summarize_export({"status": "weird"}) == ["导出盘点返回状态 weird，没有更多说明。"]
    assert summarize_export({}) == ["这份导出记录没有可读的内容。"]
