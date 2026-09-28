"""Plain-language comparison summaries: the non-expert reads sentences, not tables.

Deterministic, honest, and bounded: it restates observed facts (correct counts,
truncation, instruction echo, sample size) and their limits. It never claims
business success, never recommends adoption, and never speculates about causes
beyond what the report's own diagnostics observed.
"""

from __future__ import annotations

from typing import Any


def _pick_counterpart(
    stats: dict[str, dict[str, Any]], exact: str, contains: str | None = None
) -> dict[str, Any] | None:
    """按标签挑出基座/本轮微调那一方:先精确匹配,再按关键词取最后出现的那个。"""
    if exact in stats:
        return stats[exact]
    if contains:
        for label in reversed(list(stats)):
            if contains in label:
                return stats[label]
    return None


def summarize_comparison(report: Any) -> list[str]:
    """Turn a comparison report into a few honest sentences for a non-expert."""
    models = report.models
    if not models:
        return ["该报告没有模型结果。"]
    lines: list[str] = []
    total = models[0]["metrics"].get("total") or 0
    lines.append(f"这次对照在固定开发集的 {total} 道题上进行,所有模型用同样的题目和评分规则。")

    from src.workbench.evaluation_diagnostics import (
        count_instruction_echo,
        high_truncation_models,
        output_echoes_prompt,
    )

    first_metrics = models[0]["metrics"]
    strict = first_metrics.get("exact_match") is not None
    custom = (not strict) and first_metrics.get("pass_rate") is not None
    best_label, best_correct, best_score = None, -1, -1.0
    stats: dict[str, dict[str, Any]] = {}
    truncated_questions: set[int] = set()
    failed_questions: set[int] = set()
    echo_questions: set[int] = set()
    for model in models:
        metrics = model["metrics"]
        rows = model["rows"]
        score_value = metrics.get("exact_match")
        correct = round((score_value or 0.0) * total)
        passed = round((metrics.get("pass_rate") or 0.0) * total)
        truncated = sum(row.get("status") == "truncated" for row in rows)
        failed = sum(row.get("status") == "failed" for row in rows)
        echo = count_instruction_echo(rows)
        stats[model["label"]] = {"correct": correct, "score": score_value or 0.0}
        for position, row in enumerate(rows):
            if row.get("status") == "truncated":
                truncated_questions.add(position)
            if row.get("status") == "failed":
                failed_questions.add(position)
            if output_echoes_prompt(row.get("output"), row.get("prompt")):
                echo_questions.add(position)
        if strict:
            parts = [f"{model['label']}答对 {correct}/{total}"]
        elif custom:
            mean_score = metrics.get("business_score") or 0.0
            parts = [f"{model['label']}业务评分均值 {mean_score:.2f}、通过 {passed}/{total}"]
        else:
            generated = sum(row.get("output") is not None for row in rows)
            parts = [f"{model['label']}生成了 {generated}/{total} 条回答"]
        if truncated:
            parts.append(f"{truncated} 题没写完被截断")
        if failed:
            parts.append(f"{failed} 题生成失败")
        if echo:
            parts.append(f"{echo} 题在复述题目而不是作答")
        if strict and not (truncated or failed or echo) and correct == total:
            parts.append("全部答对")
        lines.append("· " + "、".join(parts) + "。")
        score = score_value or 0.0
        if strict and score > best_score:
            best_label, best_correct, best_score = model["label"], correct, score

    field_models = models if strict else ()  # 「答对」措辞只在严格评分下成立
    field_parts: list[str] = []
    for model in field_models:
        field_accuracy = model["metrics"].get("field_accuracy") or {}
        if not field_accuracy:
            continue
        lowest = min(value if value is not None else 0.0 for value in field_accuracy.values())
        if lowest >= 1.0:
            continue  # 每个字段都全对,披露没有信息量,交给「全部答对」句
        weakest = [name for name, value in field_accuracy.items() if (value or 0.0) == lowest]
        weakest_correct = round(lowest * total)
        names = "、".join(f"「{name}」" for name in weakest)
        field_parts.append(f"{model['label']}最弱的是{names}({weakest_correct}/{total} 题答对)")
    if field_parts:
        lines.append(
            "JSON 答案按声明的字段逐项核对,上面的「答对」指全部字段都对:"
            + "、".join(field_parts)
            + "。生成失败、截断或无法按 JSON 解析的题按该字段错误计入;"
            "要知道该字段具体错在哪里,请逐题查看完整输出。"
        )

    truncation_models = high_truncation_models(models)
    if truncation_models:
        limit = (getattr(report, "protocol", None) or {}).get("max_new_tokens")
        lines.append(
            "多个输出因触及生成长度上限被截断（"
            + "、".join(f"{label} {count} 题" for label, count in truncation_models)
            + (f"，当前 max_new_tokens 为 {limit}" if limit is not None else "")
            + "）；先核查 max_new_tokens 是否小于最短合法答案、输出是否在重复生成，"
            "再决定是否加长——触及上限不等于只需增加长度。"
        )

    base_stats = _pick_counterpart(stats, "基座")
    tuned_stats = _pick_counterpart(stats, "本轮微调", "微调")

    if custom:
        lines.append(
            "自定义业务评分按已确认规则逐题打分:上面的「通过」指达到单题通过分数,"
            "业务评分均值是各题得分的平均数,两者都不是严格准确率;"
            "要知道哪里扣分,请逐题查看评分理由。"
        )
    elif not strict:
        lines.append(
            "开放任务不做自动评分:以上只有生成与失败事实,每题通过与否由你逐题人工判断,"
            "生成失败、缺失或截断的回答不能人工标为通过。"
        )
    elif best_score == 0:
        lines.append(
            "没有一个模型答对任何题:目前不能说任何模型学会了这个任务,常见原因是题目太难、数据太少或提示格式不匹配,可查看每题的完整输出再判断。"
        )
        cause_parts = []
        if echo_questions:
            cause_parts.append(f"{len(echo_questions)} 题在复述题目")
        if truncated_questions:
            cause_parts.append(f"{len(truncated_questions)} 题没写完被截断")
        if failed_questions:
            cause_parts.append(f"{len(failed_questions)} 题生成失败")
        zero_head = (
            "微调后仍是零分,说明按当前数据量和任务定义学不出这个任务;"
            if tuned_stats
            else "所有模型都是零分;"
        )
        if cause_parts:
            lines.append(
                zero_head
                + "继续加数据之前,先核对失败原因——本次对照观察到"
                + "、".join(cause_parts)
                + ",逐题查看完整输出定位属于哪一类。"
            )
        else:
            lines.append(
                zero_head + "本次没有观察到截断、生成失败或复述,零分更可能来自答案格式不匹配;"
                "继续加数据之前,先核对输出格式与期望答案是否对得上。"
            )
    elif best_score == 1.0:
        lines.append(
            f"{best_label}在本次题目上全部答对;但题目只有 {total} 道,样本很小,不能据此断定业务上足够好。"
        )
    else:
        lines.append(
            f"答对最多的是{best_label}({best_correct}/{total});请结合逐题输出判断答错的部分是否可接受。"
        )
    if strict and best_score > 0 and base_stats and tuned_stats:
        diff = tuned_stats["score"] - base_stats["score"]
        if diff >= 0.2:
            lines.append(
                f"本轮微调比基座答对更多({tuned_stats['correct']}/{total} vs "
                f"{base_stats['correct']}/{total})——但要注意样本量,并逐题核对答错的部分再下判断。"
            )
        elif abs(diff) < 0.05:
            lines.append(
                f"微调没有带来可见变化({tuned_stats['correct']}/{total} vs "
                f"{base_stats['correct']}/{total})——数据量不足或任务难度过高都可能是原因;"
                "先逐题核对输出,再决定是加数据还是改任务定义。"
            )
    if (strict or custom) and 0 < total < 20:
        lines.append(f"注意:开发集只有 {total} 道题,任何百分比都受单题影响很大,只当方向参考。")
    lines.append("以上是观察事实,不是业务达标结论;是否采用仍由你按业务标准决定。")
    return lines


def summarize_dataset(statistics: dict) -> list[str]:
    """把数据集分区统计翻译成人话：怎么分的、各多少、排除了什么、边界声明在哪。

    只复述统计里的事实：时间方案明确说出排除与不随机补数；分组方案如实说明
    实际比例受分组大小影响。不宣称训练效果，也不替用户判断业务达标。
    """
    counts = statistics.get("row_counts") or {}
    train = counts.get("train", 0)
    validation = counts.get("validation", 0)
    test = counts.get("test", 0)
    total = statistics.get("total_rows") or sum(counts.values())
    method = statistics.get("split_method", "")
    lines: list[str] = []
    if method.startswith("temporal"):
        included = statistics.get("included_rows", train + validation + test)
        excluded = statistics.get("excluded_rows", 0)
        lines.append(
            f"本版本按已确认的时间边界划分：训练 {train} 条、验证 {validation} 条、"
            f"独立测试 {test} 条，共纳入 {included} 条（全量 {total} 条）；"
            "分界线与本版本实际使用的时间字段以已确认的时间方案为准。"
        )
        if excluded:
            lines.append(
                f"另有 {excluded} 条因标签未成熟、跨越分区边界或与排除记录同源被明确排除，"
                "原行完整保留在排除明细里；没有随机补数，也没有把未成熟标签当作真值。"
            )
        else:
            lines.append("没有记录被排除，全部按时间边界纳入对应分区。")
        if method == "temporal_fixed_evaluation_suite":
            lines.append(
                "本版本同时沿用固定开发/测试题集：后续轮次在同一批题目上比较，新增资料不扩充评分题。"
            )
    elif method == "fixed_evaluation_suite":
        lines.append(
            f"本版本沿用固定开发/测试题集：训练 {train} 条、验证 {validation} 条、"
            f"独立测试 {test} 条；新增独立资料进入训练，评分题目保持不变。"
        )
    else:
        groups = statistics.get("independent_groups", "?")
        lines.append(
            f"本版本按业务对象隔离划分：训练 {train} 条、验证 {validation} 条、"
            f"独立测试 {test} 条，共 {total} 条、{groups} 个独立分组；"
            "同一对象的记录保持在同一分区，实际比例受分组大小影响。"
        )
    coverage_note = statistics.get("answer_coverage_note")
    if coverage_note:
        lines.append(coverage_note)
    duplicate_note = statistics.get("duplicate_note")
    if duplicate_note:
        lines.append(duplicate_note)
    lines.append("分区就绪只说明数据已按规则隔离、可以进入训练前检查；不代表模型效果或业务达标。")
    return lines


_PLAN_STATUS_NAMES = {
    "ready": "方案可供确认",
    "needs_data": "需要先完善数据",
    "unsupported": "当前条件不支持",
}


def summarize_plan(record: dict) -> list[str]:
    """把一份 Agent 训练方案翻译成人话:建议用什么、关键参数、理由与边界。

    只复述记录里的事实:推荐理由与限制是 Agent 写下的原文;就绪只说明参数、
    数据与实际预检检查通过。确认准备不会自动启动训练,方案就绪也不代表
    训练效果或业务达标。
    """
    proposal = record.get("proposal") or {}
    status = record.get("status") or proposal.get("status")
    model_path = str(proposal.get("model_path") or "").rstrip("/")
    training = proposal.get("training_options") or {}
    lora = proposal.get("lora_options") or {}
    model_options = proposal.get("model_options") or {}
    parts: list[str] = []
    if proposal.get("max_length") is not None:
        parts.append(f"最大长度 {proposal['max_length']}")
    if training.get("num_epochs") is not None:
        parts.append(f"训练 {training['num_epochs']} 轮")
    if training.get("batch_size") is not None:
        parts.append(f"batch size {training['batch_size']}")
    if training.get("learning_rate") is not None:
        parts.append(f"学习率 {training['learning_rate']}")
    if lora.get("r") is not None:
        parts.append(f"LoRA rank {lora['r']}")
    if model_options.get("quantization_bits"):
        parts.append(f"{model_options['quantization_bits']}-bit 量化")
    if model_path:
        head = f"这份方案建议用 {model_path.split('/')[-1]}"
        if parts:
            head += "（" + "、".join(parts) + "）"
        lines = [head + "。"]
    else:
        lines = ["这份方案还没有选择基础模型。"]
    if status:
        lines.append(f"当前状态：{_PLAN_STATUS_NAMES.get(status, status)}。")
    rationale = [str(item) for item in (proposal.get("rationale") or []) if str(item).strip()]
    if rationale:
        lines.append("推荐理由：" + "；".join(rationale))
    limitations = [str(item) for item in (proposal.get("limitations") or []) if str(item).strip()]
    if limitations:
        lines.append("尚未验证的限制：" + "；".join(limitations))
    questions = [
        str(item) for item in (proposal.get("business_questions") or []) if str(item).strip()
    ]
    if questions:
        lines.append(
            "还有需要你先回答的业务问题：" + "；".join(questions) + "——回答确认前不能准备训练。"
        )
    if status == "ready":
        lines.append(
            "就绪只说明参数、数据与实际预检检查通过；确认这份方案只会准备训练，不会自动启动。"
        )
    elif status == "needs_data":
        lines.append("先完善数据或回答业务问题，再让 Agent 重新生成方案。")
    elif status == "unsupported":
        lines.append("换用支持的模型或机器后重新生成方案。")
    lines.append("方案就绪与推荐理由不构成训练效果或业务达标的判断，是否采用由你按业务决定。")
    return lines


def summarize_preflight(preflight: dict) -> list[str]:
    """把训练前检查报告翻译成人话:答案是否保留、截断多少、问题在哪。"""
    if not preflight:
        return ["尚未执行训练前检查。"]
    lines: list[str] = []
    status = preflight.get("status")
    splits = preflight.get("splits") or {}
    truncated = sum(split.get("truncated_rows", 0) for split in splits.values())
    lost = sum(split.get("answer_lost_rows", 0) for split in splits.values())
    total_rows = sum(split.get("rows", 0) for split in splits.values())
    if status == "passed":
        lines.append(
            f"训练前检查通过：{total_rows} 行数据按所选模型的分词方式处理后，答案都完整保留。"
        )
    elif status == "warnings":
        lines.append("训练前检查有需要你核对的风险，确认理解后才能启动训练。")
    else:
        lines.append("训练前检查发现阻断问题，训练不能开始；按下面列出的原因修复数据或配置。")
    if truncated:
        detail = "、".join(
            f"{name} {split.get('truncated_rows', 0)} 行"
            for name, split in splits.items()
            if split.get("truncated_rows")
        )
        rows = preflight.get("rows") or []
        answer_partial = sum(1 for row in rows if row.get("answer_was_truncated"))
        if answer_partial:
            lines.append(
                f"有 {answer_partial} 行的答案在截断后丢了一部分（共截断 {detail}）——"
                "优先加大长度或缩短输入；答案不完整的样本教不会模型正确作答。"
            )
        else:
            lines.append(
                f"有内容超出长度上限被截断（{detail}）——答案都保留了，截掉的是输入内容；请核对被截部分是否关键。"
            )
        full_tokens = [
            row.get("full_tokens") for row in rows if isinstance(row.get("full_tokens"), int)
        ]
        used = preflight.get("max_length")
        if full_tokens and isinstance(used, int):
            needed = max(full_tokens)
            suggested = ((needed + 63) // 64) * 64
            if used < needed:
                lines.append(
                    f"当前长度 {used}，最长一条记录需要 {needed}——把长度设为 {suggested} 左右即可全部放下；"
                    "显存吃紧时优先压缩最长的输入字段，而不是压长度。"
                )
            else:
                lines.append(
                    f"当前长度 {used} 已能容纳最长记录（{needed}），截断来自个别超长行，可单独核对。"
                )
    else:
        lines.append("没有内容因长度超限被截断。")
    if lost:
        lines.append(
            f"有 {lost} 行的答案在截断后完全丢失，这属于阻断问题，需要缩短内容或加大长度。"
        )
    issues = preflight.get("issues") or []
    for issue in issues:
        if issue.get("severity") in {"blocking", "warning"}:
            lines.append(f"[{issue.get('severity')}] {issue.get('message', '')}")
    lines.append("检查通过只说明数据能被正确消费，不代表训练效果或业务达标。")
    return lines


def summarize_training_run(record: dict) -> list[str]:
    """把一次训练的状态翻译成人话:用的是什么、进展如何、产物在哪、边界在哪。"""
    status = record.get("status")
    model_name = str(record.get("model_path", "")).rstrip("/").split("/")[-1]
    config = (record.get("config") or {}).get("training", {})
    epochs = config.get("num_epochs", "?")
    base = f"这次训练基于 {model_name}，计划训练 {epochs} 轮，"
    names = {
        "prepared": "方案已准备好并通过检查，还没有开始训练。",
        "running": "正在训练中；关闭页面不影响后台训练，可稍后回来刷新。",
        "succeeded": "训练完成，产出了微调适配器。",
        "failed": "训练失败了。",
        "stopped": "训练被手动停止。",
        "blocked": "准备阶段被问题阻断，训练没有开始。",
        "stopping": "正在停止训练。",
    }
    lines = [base + names.get(status, f"当前状态：{status}。")]
    failure = record.get("failure") or {}
    if failure:
        lines.append(f"失败发生在{failure.get('stage', '未知')}阶段：{failure.get('message', '')}")
    metrics = record.get("metrics") or {}
    if metrics.get("train_loss") is not None:
        lines.append(
            f"训练损失（loss）最终为 {metrics['train_loss']:.4f}；它下降说明模型在记题，不代表业务效果。"
        )
    if status == "succeeded":
        lines.append(
            "训练完成只说明产出了模型；效果要用同一套开发题与基座对照来判断，请看对照报告。"
        )
    return lines


_ACCEPTANCE_METRIC_NAMES = {
    "exact_match": "严格匹配",
    "pass_rate": "自定义规则通过率",
    "manual_acceptance_rate": "人工逐题判断",
}


def summarize_acceptance(record: dict) -> list[str]:
    """把一次最终验收翻译成人话：冻结了什么条款、结论是哪一态、哪些数字不作数。

    只复述记录里的事实：通过数与分母口径（失败与截断按未通过计）、隔离未核验时
    数值不作数、业务评分均值仅作描述。结论只对这次冻结的条款与固定测试题负责，
    不替用户宣判可交付。
    """
    model = record.get("model") or {}
    criteria = record.get("criteria") or {}
    result = record.get("result") or {}
    decision = result.get("decision") or "pending_run"
    status = record.get("status") or "prepared"
    metric = criteria.get("metric")
    minimum_score = criteria.get("minimum_score")
    minimum_cases = criteria.get("minimum_cases")

    head = f"这次最终验收针对模型「{model.get('label', '未命名')}」"
    if minimum_score is not None and minimum_cases is not None:
        metric_name = _ACCEPTANCE_METRIC_NAMES.get(metric, metric or "评分")
        head += f"，运行前冻结的标准是：{metric_name}至少 {minimum_score:.0%}，且至少 {minimum_cases} 道独立测试题"
    lines = [head + "。"]

    if decision == "pending_run":
        lines.append(
            "条款已冻结、验收尚未执行。执行时会先核验这些留出题是否已经揭示；"
            "执行后这些测试题即被占用，即使生成失败也不能再伪装成首次盲测。"
        )
    elif status == "blocked":
        lines.append(
            "验收被阻断：这些测试题或同任务业务对象此前已经揭示，"
            "换模型、重上传或换套件标识都不能重新变成盲测；请准备真正独立的新保留资料。"
        )
    elif status == "failed":
        reason = result.get("reason") or "执行过程失败，没有形成可用的结论。"
        lines.append(f"这次验收没有完整执行：{reason}")
    elif decision == "pending_review":
        reviewed = result.get("reviewed_cases")
        total = result.get("total_cases")
        counts = (
            f"已记录 {reviewed}/{total} 题判断，"
            if reviewed is not None and total is not None
            else ""
        )
        lines.append(
            f"等待逐题业务判断：{counts}生成失败、缺失或截断的回答已按未通过锁定，"
            "不能人工改标为通过。"
        )
    else:
        names = {
            "passed": "达到运行前冻结的验收标准",
            "failed": "未达到运行前冻结的验收标准",
            "insufficient_evidence": "证据不足，不能确认可交付",
        }
        lines.append(f"当前结论：{names.get(decision, decision)}。")
        accepted = result.get("accepted_cases")
        total = result.get("total_cases")
        score = result.get("score")
        if accepted is not None and total and score is not None:
            lines.append(
                f"通过 {accepted}/{total} 题（通过率 {score:.1%}）；"
                "失败与截断保留在全部题目分母中，按未通过计。"
            )
        usable = result.get("usable_cases")
        if usable is not None and total and usable < total:
            lines.append(f"其中可完整核查的有效输出只有 {usable}/{total} 题。")
        observed = result.get("observed_decision")
        if decision == "insufficient_evidence" and observed in {"passed", "failed"}:
            wording = "达到标准" if observed == "passed" else "未达到标准"
            lines.append(
                f"单看通过率本会判为「{wording}」；但训练资料与留出题的隔离未能核验，"
                "数值不能作为独立业务验收的结论。"
            )
        reason = result.get("reason")
        if decision == "insufficient_evidence" and reason:
            lines.append(f"原因：{reason}")
        business_score = result.get("business_score")
        if business_score is not None:
            lines.append(
                f"业务评分均值 {business_score:.2f}，仅作描述；"
                "验收结论按冻结的逐题通过门槛与整体通过率计算。"
            )
    lines.append(
        "以上结论只对这次冻结的条款与固定测试题负责；达到标准也不会自动部署模型，"
        "是否交付由你按业务决定。"
    )
    return lines


_ITERATION_DECISION_NAMES = {
    "adopt": "采用本轮结果",
    "continue": "继续改进",
    "stop": "停止本轮路线",
    "insufficient_evidence": "证据不足",
}


def summarize_iteration(record: dict) -> list[str]:
    """把一个改进轮次记录翻译成人话：当前停在哪一步、业务决定是否已记录。

    只复述记录里的事实：状态机位置、已记录的决定与业务理由；采用记录不会
    自动部署模型，流程状态不代表业务效果达标。
    """
    hypothesis = record.get("hypothesis")
    lines = [f"这轮改进的假设是：{hypothesis}。" if hypothesis else "这轮改进提案。"]
    status = record.get("status")
    if status == "decided":
        decision = record.get("decision")
        lines.append(f"已记录你的业务决定：{_ITERATION_DECISION_NAMES.get(decision, decision)}。")
        reason = record.get("decision_reason")
        if reason:
            lines.append(f"业务理由：{reason}")
        if decision == "adopt":
            lines.append("采用记录不会自动部署模型；是否上线由你按业务决定。")
        elif decision == "insufficient_evidence":
            lines.append("固定题数太少、输出截断或开放任务尚无评分时，应明确保留证据局限。")
    elif status == "evaluated":
        lines.append(
            "基座、父轮与本轮的三模型同题对照已完成，正等待你的业务决定："
            "采用、继续、停止或证据不足。"
        )
    elif status == "running":
        lines.append("本轮训练已启动；训练完成只说明产出模型，效果要看三模型同题对照。")
    elif status == "preparing":
        lines.append("正在准备本轮训练。")
    elif status == "prepared":
        lines.append("本轮训练方案已准备，尚未启动。")
    elif status == "confirmed":
        lines.append("本轮假设与变更范围已确认，尚未准备训练。")
    elif status == "blocked":
        failure = record.get("failure")
        lines.append(f"本轮训练准备被阻断：{failure}。" if failure else "本轮训练准备被阻断。")
        lines.append("请处理问题后重新准备；处理不了就提出新的改进轮次。")
    else:
        lines.append("提案已保存，尚未确认；确认假设与变更范围后才会准备训练。")
    lines.append("以上只是流程状态与已记录的决定，不代表业务效果达标。")
    return lines


_EXECUTION_IN_PROGRESS = {
    "queued": "已受理，等待后台执行开始。",
    "materializing": "正在用本轮固定题集准备数据版本。",
    "preparing": "正在按已确认方案准备训练。",
    "training": "训练已按确认方案启动。",
    "waiting_for_release": "训练进程已退出，等待释放资源后开始开发集对照。",
    "evaluating": "正在按父轮协议比较基座、父轮与本轮模型。",
}


def summarize_execution(record: dict) -> list[str]:
    """把一次自动执行记录翻译成人话：后台走到哪一步、是否需要用户动作、终态事实。

    awaiting_warning_ack 表示已暂停等待用户核对，completed 表示对照完成、轮次
    停在待业务决定——都不是业务效果结论。
    """
    status = record.get("status")
    lines = []
    if status in _EXECUTION_IN_PROGRESS:
        lines.append(f"自动执行正在后台推进：{_EXECUTION_IN_PROGRESS[status]}")
        lines.append("后台进程独立于页面与终端运行，关闭页面不影响执行。")
    elif status == "awaiting_warning_ack":
        lines.append("自动执行已暂停：预检存在需核对的提示。")
        lines.append("请查看提示内容后，在原入口勾选确认继续才会恢复；不会跳过提示自动训练。")
    elif status == "completed":
        lines.append("自动执行已完成：三模型同题对照报告已生成。")
        pointers = []
        if record.get("run_id"):
            pointers.append(f"训练 {record['run_id']}")
        if record.get("evaluation_id"):
            pointers.append(f"评测 {record['evaluation_id']}")
        if pointers:
            lines.append(f"相关记录：{'、'.join(pointers)}。")
        lines.append(
            "轮次停在待业务决定，请核对三模型结果后选择采用、继续、停止或证据不足；"
            "重复提交不会再次训练，只返回这份报告。"
        )
    elif status in {"blocked", "failed"}:
        label = "被阻断" if status == "blocked" else "失败"
        message = record.get("message")
        lines.append(f"自动执行{label}：{message}" if message else f"自动执行{label}。")
        issues = record.get("issues") or []
        if issues:
            shown = "；".join(str(issue) for issue in issues[:3])
            if len(issues) > 3:
                shown += "等"
            lines.append(f"问题：{shown}。")
        lines.append("请查看执行记录与 worker.log，处理问题后提出新的改进轮次。")
    elif status == "stopped":
        lines.append("已按请求停止自动执行；本轮不再推进，如需继续请提出新的改进轮次。")
    else:
        message = record.get("message")
        lines.append(f"自动执行状态：{message}" if message else "自动执行状态未记录。")
    lines.append("以上是自动执行的当前状态，不代表业务效果达标。")
    return lines


def summarize_scoring(record: dict) -> list[str]:
    """把一份业务评分规则记录翻译成人话：标准是什么、通过线在哪、验证与确认状态。

    只复述记录里的事实：规则是确定性代码，须在真实隔离后端用业务正例与同题
    反例验证；确认由用户完成，绑定当前业务目标与输入/答案语义。业务分均值与
    通过率是规则口径的描述，不等于严格准确率。
    """
    if record.get("status") == "needs_business_input":
        lines = ["业务评分标准还不能转成可执行的规则。"]
        reason = str(record.get("reason") or "").strip()
        if reason:
            lines.append(f"原因：{reason}")
        questions = [str(item) for item in (record.get("questions") or []) if str(item).strip()]
        if questions:
            lines.append("需要你先补充的业务问题：" + "；".join(questions))
        lines.append("请补充业务标准后重新拟定规则；当前没有可确认的评分方案。")
        return lines
    recipe = record.get("recipe") or {}
    standard = str(recipe.get("business_standard") or "").strip()
    lines = [
        f"这套规则要判断的业务标准：{standard}。" if standard else "这套规则没有写明业务标准。"
    ]
    threshold = recipe.get("pass_threshold")
    if threshold is not None:
        lines.append(f"单题得分达到 {threshold} 才计为通过；均分与通过率分开展示。")
    examples = recipe.get("examples") or []
    if examples:
        positives = sum(1 for item in examples if item.get("kind") == "business")
        counterexamples = sum(1 for item in examples if item.get("kind") == "counterexample")
        lines.append(
            f"用 {positives} 条业务正例与 {counterexamples} 条同题反例在真实隔离后端验证。"
        )
    validation = record.get("validation") or {}
    if validation.get("status"):
        backend = validation.get("backend") or "未记录"
        lines.append(f"隔离验证：{validation['status']}（后端：{backend}）。")
    if record.get("status") == "confirmed":
        lines.append(
            "已确认：规则绑定当前业务目标与输入/答案语义；数据修订后兼容规则可继续用，"
            "业务目标或含义变更需重新确认。"
        )
    else:
        lines.append("草稿待你核对实际正反例分数与理由后确认；软件不会自动确认评分规则。")
    lines.append("业务评分均值与通过率是规则口径的描述，不等于严格准确率，也不构成业务达标的判断。")
    return lines


def summarize_suite(record: dict) -> list[str]:
    """把固定评测题集翻译成人话：多少题、内容锁定不可改、复用方式与比较基线边界。

    兼容两种记录形态：suite-freeze 返回的引用（只有题数与内容摘要）与
    suite-show 读出的完整清单（另有锚定版本与固定范围原文）。只复述记录里的
    事实：题目内容按摘要锁定，原评分题不能修改；同对象新增行不自动扩充评分题。
    """
    counts = record.get("case_counts") or {}
    dev = counts.get("validation")
    test = counts.get("test")
    if dev is not None or test is not None:
        parts = []
        if dev is not None:
            parts.append(f"开发题 {dev} 道")
        if test is not None:
            parts.append(f"最终测试题 {test} 道")
        lines = [f"这套固定题集含{'、'.join(parts)}，供后续各轮用同一套题比较。"]
    else:
        lines = ["这套固定题集没有记录题数。"]
    digest = str(record.get("cases_digest") or "").strip()
    if digest:
        lines.append(f"题目内容已按摘要 {digest[:12]}… 锁定，原评分题不能修改。")
    scope = str(record.get("scope_note") or "").strip()
    lines.append(
        scope or "原开发/最终测试评分题固定；同对象新增行保留在对应分区，不自动扩充评分题。"
    )
    anchor = record.get("anchor_dataset") or {}
    name = str(anchor.get("name") or "").strip()
    version = anchor.get("version")
    if name or version is not None:
        anchor_text = f"锚定数据版本：{name or '未记录'}"
        if version is not None:
            anchor_text += f"（version {version}）"
        lines.append(anchor_text + "。")
    lines.append("materialize 时用 --suite-id 指定即可复用这套题集。")
    lines.append("固定题集只保证各轮比较基线一致，不代表业务效果达标。")
    return lines


def summarize_assessment(record: dict) -> list[str]:
    """把 Agent 评测解读翻译成人话：证据事实、待核查假设、建议与不自动执行的边界。

    只复述记录里的事实：观察与假设分离（假设是待核查原因，不是事实），工具
    调用轨迹如实计数（含失败）。解读只给建议，不改数据、不启动训练，也不
    代表业务效果达标。
    """
    assessment = record.get("assessment") or {}
    if not assessment:
        return [
            "这份解读没有可读的内容。",
            "这是开发集诊断建议，不代表业务效果达标。",
        ]
    lines = []
    model = str(record.get("model") or "").strip()
    head = "这份解读由 Agent 在核查真实工具证据后给出"
    if model:
        head += f"（模型 {model}）"
    lines.append(head + "。")
    summary = str(assessment.get("summary") or "").strip()
    if summary:
        lines.append(f"总述：{summary}")
    observations = [item for item in (assessment.get("observations") or []) if item]
    hypotheses = [item for item in (assessment.get("hypotheses") or []) if item]
    evidence_count = len(
        {
            identity
            for item in [*observations, *hypotheses]
            for identity in (item.get("evidence_ids") or [])
        }
    )
    parts = [f"有证据的观察 {len(observations)} 条"]
    if hypotheses:
        parts.append(f"待核查原因 {len(hypotheses)} 条——这是假设不是事实，每条附验证方式")
    if evidence_count:
        parts.append(f"共引用 {evidence_count} 处工具证据")
    lines.append("、".join(parts) + "。")
    decision = str(assessment.get("decision") or "").strip()
    if decision:
        names = {
            "inspect_data": "先核查数据",
            "revise_pipeline": "先修订处理方案",
            "inspect_training": "先核查训练行为",
            "collect_evidence": "先补充证据",
            "business_review": "需要业务核对",
        }
        lines.append(f"建议优先处理：{names.get(decision, decision)}。")
    steps = [str(item) for item in (assessment.get("next_steps") or []) if str(item).strip()]
    if steps:
        lines.append("建议下一步：" + "；".join(steps))
    limitations = [str(item) for item in (assessment.get("limitations") or []) if str(item).strip()]
    if limitations:
        lines.append("解读自己声明的局限：" + "；".join(limitations))
    questions = [
        str(item) for item in (assessment.get("business_questions") or []) if str(item).strip()
    ]
    if questions:
        lines.append("需要你先回答的业务问题：" + "；".join(questions))
    trace = [item for item in (record.get("tool_trace") or []) if isinstance(item, dict)]
    if trace:
        ok_count = sum(1 for item in trace if item.get("ok"))
        failed = len(trace) - ok_count
        trace_text = f"工具核查轨迹：{len(trace)} 次调用，成功 {ok_count} 次"
        if failed:
            trace_text += f"、失败 {failed} 次——失败的调用没有取到证据，解读只依赖成功的调用"
        lines.append(trace_text + "。")
    lines.append(
        "以上是开发集诊断建议：软件不会据此自动改标签、删除坏例或采纳方案变更，"
        "也不会自动启动下一轮训练；是否有效仍需你的业务判断，不代表业务效果达标。"
    )
    return lines


def summarize_full_report(record: dict) -> list[str]:
    """把全量验证报告翻译成人话：来源、结论四态、逐条问题原文与不达标边界。

    只复述记录里的事实：问题按报告原文逐条渲染（阻断在前，附涉及证据行条数），
    状态与转换计数的词汇与页面全量验证区一致；全量验证只核对数据与已确认方案
    的一致性，不代表模型效果或业务达标。
    """
    if not record:
        return [
            "这份全量报告没有可读的内容。",
            "全量验证只核对数据事实，不代表模型效果或业务达标。",
        ]
    source = record.get("source") or {}
    name = str(source.get("name") or "").strip()
    rows = source.get("rows")
    digest = str(source.get("digest") or "").strip()
    head = "这份全量验证针对资料"
    if name:
        head += f" {name}"
    facts = []
    if isinstance(rows, list):
        facts.append(f"{len(rows)} 条")
    if digest:
        facts.append(f"摘要 {digest[:12]}…")
    if facts:
        head += "（" + "、".join(facts) + "）"
    lines = [head + "，报告中的行 ID 仅属于这份全量文件。"]
    status = str(record.get("status") or "").strip()
    status_names = {
        "needs_revision": "存在阻断问题，需先按下面的问题修正资料或业务规则，再重新验证全量。",
        "review": "没有阻断问题，待你核对报告与展示的记录后确认全量数据含义。",
        "confirmed": "全量数据含义已确认，可以准备生成分区；尚未开始训练。",
        "stale": "业务理解或方案已变化，这份全量报告已失效；重新分析并确认样例方案后再验证全量。",
    }
    if status:
        lines.append(f"当前结论：{status_names.get(status, status)}")
    issues = [item for item in (record.get("issues") or []) if isinstance(item, dict)]
    order = {"blocking": 0, "review": 1, "info": 2}
    severity_names = {"blocking": "阻断", "review": "需核对", "info": "说明"}
    for issue in sorted(issues, key=lambda item: order.get(item.get("severity"), 3)):
        message = str(issue.get("message") or "").strip()
        if not message:
            continue
        text = f"[{severity_names.get(issue.get('severity'), '说明')}] {message}"
        row_ids = issue.get("row_ids") or []
        if row_ids:
            text += f"（涉及 {len(row_ids)} 条证据行）"
        lines.append(text)
    preview = record.get("preview") or {}
    counts = preview.get("counts") or {}
    state_names = {
        "ready": "已生成预览",
        "needs_label": "缺少答案",
        "invalid": "需修正处理",
        "conflict": "答案有冲突",
    }
    rendered = [
        f"{state_names[key]} {count} 条"
        for key, count in counts.items()
        if key in state_names and count
    ]
    if rendered:
        lines.append("全量真实转换：" + "、".join(rendered) + "。")
    lines.append("全量验证只核对数据事实与已确认方案的一致性，不代表模型效果或业务达标。")
    return lines


def summarize_listing(label: str, items: list, first_step: str) -> list[str]:
    """*-list 命令的人话尾行:空清单点名下一步入口,非空给计数。

    空清单静默是最差体验——用户分不清「还没有」和「查错了任务」;
    每类清单的下一步入口不同,由调用方给出,不在此编造。
    """
    if not items:
        return [f"当前任务还没有{label}；{first_step}"]
    return [f"共 {len(items)} 条{label}。"]


def summarize_model_discovery(items: list) -> list[str]:
    """model-list 尾行:与页面候选模型区同词汇的发现计数与文件完整边界。

    空态点名准备入口;非空给文件完整/不完整分档计数,并复述页面的边界句——
    重量文件只 stat 未加载,文件完整不等于兼容或能训练,适配由方案检查判断。
    """
    if not items:
        return [
            "暂未发现本地候选模型。先把完整的基础模型放进本机目录（如 models/），"
            "或用 --root 指定已有目录，再运行 model-list 核对。"
        ]
    complete = sum(1 for item in items if item.get("status") == "available")
    incomplete = len(items) - complete
    count_line = f"发现 {len(items)} 个本地模型：{complete} 个文件完整、{incomplete} 个文件不完整"
    if incomplete:
        count_line += "（缺什么看 JSON 里的 issues 字段）"
    return [
        count_line + "。",
        "文件完整只表示可以进一步检查；模型是否兼容、训练长度和机器是否适合，仍由方案检查判断。",
    ]
