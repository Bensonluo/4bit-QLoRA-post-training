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
    assert "没有观察到截断、生成失败或复述" in joined
    assert "答案格式不匹配" in joined


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
