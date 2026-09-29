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
    # 回声分流引导与页面警告同源（echo_triage_lines 单一来源）
    models = report.models
    if not models:
        return ["该报告没有模型结果。"]
    # 逐题查看出口(R98 点名→R101 帧尾单命令):评分理由与完整输出只在已存盘报告的
    # rows 里,各语境只报事实与要判断什么,命令在结语行前单处发射、整帧至多一条;
    # 缺键的极简夹具回退大写占位符,不假装知道 ID。
    evaluation_id = getattr(report, "evaluation_id", None) or "EVALUATION_ID"
    inspect_targets: list[str] = []  # 帧尾单命令的查看目标,保序去重拼接
    lines: list[str] = []
    total = models[0]["metrics"].get("total") or 0
    lines.append(f"这次对照在固定开发集的 {total} 道题上进行,所有模型用同样的题目和评分规则。")

    from src.workbench.evaluation_diagnostics import (
        count_instruction_echo,
        dominant_output_models,
        echo_triage_lines,
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
            "要知道该字段具体错在哪里,需逐题查看完整输出。"
        )
        inspect_targets.append("完整输出")

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

    dominant_models = dominant_output_models(models)
    if dominant_models:
        dominant_parts = []
        for label, count, generated, top_output in dominant_models:
            clipped = top_output if len(top_output) <= 24 else top_output[:24] + "…"
            dominant_parts.append(f"{label} 有 {count}/{generated} 条输出完全相同（{clipped}）")
        lines.append(
            "观察到输出高度重复（"
            + "、".join(dominant_parts)
            + "）：该模型在反复输出同一答案。请对照开发集答案分布——分布本身集中时,"
            "模型可能只是复述多数类,不一定是学坏了;分布不集中时,需确认每题是否"
            "本该有不同答案。这是观察事实,不认定原因。"
        )
        inspect_targets.append("完整输出")

    for model in models:
        lines.extend(
            echo_triage_lines(
                model["label"],
                model["rows"],
                protocol=(getattr(report, "protocol", None) or {}),
            )
        )

    base_stats = _pick_counterpart(stats, "基座")
    tuned_stats = _pick_counterpart(stats, "本轮微调", "微调")

    if custom:
        lines.append(
            "自定义业务评分按已确认规则逐题打分:上面的「通过」指达到单题通过分数,"
            "业务评分均值是各题得分的平均数,两者都不是严格准确率;"
            "要知道哪里扣分,需逐题查看评分理由。"
        )
        inspect_targets.append("评分理由")
    elif not strict:
        lines.append(
            "开放任务不做自动评分:以上只有生成与失败事实,每题通过与否由你逐题人工判断,"
            "生成失败、缺失或截断的回答不能人工标为通过。"
        )
    elif best_score == 0:
        lines.append(
            "没有一个模型答对任何题:目前不能说任何模型学会了这个任务,常见原因是题目太难、数据太少或提示格式不匹配;属于哪一类,需逐题判断。"
        )
        inspect_targets.append("完整输出")
        cause_parts = []
        if echo_questions:
            cause_parts.append(f"{len(echo_questions)} 题在复述题目")
        if truncated_questions:
            cause_parts.append(f"{len(truncated_questions)} 题没写完被截断")
        if failed_questions:
            cause_parts.append(f"{len(failed_questions)} 题生成失败")
        if dominant_models:
            cause_parts.append("有模型在反复输出同一答案")
        # 全技术性零分:所有题在所有模型里都没写完或生成失败——零分只证明生成没走通,
        # 不构成「学不出这个任务」的证据;此时不给对号行(头部已给行动顺序)。
        all_technical = total > 0 and all(
            all(row.get("status") in {"truncated", "failed"} for row in model["rows"])
            for model in models
        )
        if all_technical:
            zero_head = ("微调后仍是零分" if tuned_stats else "所有模型都是零分") + (
                f"——但全部 {total} 道题都没写完或生成失败,这个零分只说明生成环节没走通,"
                "还不构成「按当前数据量和任务定义学不出这个任务」的证据;"
            )
            zero_tail = "先修生成长度与失败原因后重测,再谈补数据、改任务定义或停止。"
        else:
            zero_head = (
                "微调后仍是零分,说明按当前数据量和任务定义学不出这个任务;"
                if tuned_stats
                else "所有模型都是零分;"
            )
            zero_tail = "再逐题定位属于哪一类。"
        if cause_parts:
            lines.append(
                zero_head
                + "继续加数据之前,先核对失败原因——本次对照观察到"
                + "、".join(cause_parts)
                + "，"
                + zero_tail
            )
            if not all_technical:
                # 介入点编码:每类失败原因的第一步各不相同,补数据排在这些技术原因之后。
                remedy_parts = []
                if echo_questions:
                    remedy_parts.append("复述题目——补数据治不了回声,先核对提示模板与指令长度")
                if truncated_questions:
                    remedy_parts.append("截断——先加生成长度重测,当前分数低估了模型")
                if failed_questions:
                    remedy_parts.append("生成失败——先修失败原因,失败题没有测到模型")
                if dominant_models:
                    remedy_parts.append("重复输出——先对照上面的答案分布披露判断")
                lines.append(
                    "失败原因对号处理："
                    + "；".join(remedy_parts)
                    + "。补数据是这些技术原因逐一排除后的选项;改任务定义还是停止,"
                    "在排除后再按业务判断。"
                )
        else:
            lines.append(
                zero_head
                + "本次没有观察到截断、生成失败、复述或重复输出,零分更可能来自答案格式不匹配;"
                "继续加数据之前,先核对输出格式与期望答案是否对得上;"
                "格式对得上之前,补数据和改任务定义都还不是下一步。"
            )
    elif best_score == 1.0:
        lines.append(
            f"{best_label}在本次题目上全部答对;但题目只有 {total} 道,样本很小,不能据此断定业务上足够好。"
        )
    else:
        lines.append(
            f"答对最多的是{best_label}({best_correct}/{total});请结合逐题输出判断答错的部分是否可接受。"
        )
        inspect_targets.append("完整输出")
    if strict and best_score > 0 and base_stats and tuned_stats:
        diff = tuned_stats["score"] - base_stats["score"]
        if diff >= 0.2:
            lines.append(
                f"本轮微调比基座答对更多({tuned_stats['correct']}/{total} vs "
                f"{base_stats['correct']}/{total})——但要注意样本量,并逐题核对答错的部分再下判断。"
            )
            inspect_targets.append("完整输出")
        elif abs(diff) < 0.05:
            lines.append(
                f"微调没有带来可见变化({tuned_stats['correct']}/{total} vs "
                f"{base_stats['correct']}/{total})——数据量不足或任务难度过高都可能是原因;"
                "先逐题核对输出,再决定是加数据还是改任务定义。"
            )
            inspect_targets.append("完整输出")
    if (strict or custom) and 0 < total < 20:
        lines.append(f"注意:开发集只有 {total} 道题,任何百分比都受单题影响很大,只当方向参考。")
    if inspect_targets and (strict or custom):
        # 帧尾单命令(R101):各语境的查看需求汇成一条命令,整帧 count==1;
        # 目标保序去重——复述+自定义叠加帧为「完整输出与评分理由」。
        # 开放任务不点名命令(r98-reviewer nit-1 既定决策):该命令不解决开放
        # 任务的人工判断问题,点名反而是噪音——开放帧即使复述语境置位也不发射。
        target = "与".join(dict.fromkeys(inspect_targets))
        lines.append(f"逐题查看{target}:eval-show {evaluation_id}。")
    lines.append("以上是观察事实,不是业务达标结论;是否采用仍由你按业务标准决定。")
    return lines


def summarize_dataset(statistics: dict) -> list[str]:
    """把数据集分区统计翻译成人话：怎么分的、各多少、排除了什么、边界声明在哪。

    只复述统计里的事实：时间方案明确说出排除与不随机补数；分组方案如实说明
    实际比例受分组大小影响；独立测试集过小时附上每题权重与偶然性提醒（单一来源）。
    不宣称训练效果，也不替用户判断业务达标。
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
    # 独立测试集过小时的诚实算术提醒(1/N 确定性算术 + 30 条经验阈值):紧跟各方案
    # 的条数行,单一来源 training_guidance,页面产物区与 CLI materialize 同源同词汇。
    from src.workbench.training_guidance import small_test_set_line

    caution = small_test_set_line(test)
    if caution:
        lines.append(caution)
    coverage_note = statistics.get("answer_coverage_note")
    if coverage_note:
        lines.append(coverage_note)
    duplicate_note = statistics.get("duplicate_note")
    if duplicate_note:
        lines.append(duplicate_note)
    lines.append("分区就绪只说明数据已按规则隔离、可以进入训练前检查；不代表模型效果或业务达标。")
    return lines


# 方案状态枚举 → 人话名的单一来源（公开常量）：CLI plan-* 摘要的状态行与页面
# 方案卡的状态标签从这里取名。README 的三态说明由 test_readme_alignment 钉住同词汇。
PLAN_STATUS_NAMES = {
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
        lines.append(f"当前状态：{PLAN_STATUS_NAMES.get(status, status)}。")
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
    lines.extend(summarize_tool_trace(record.get("trace"), "方案"))
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
        # 训练完成出口点名(R96):悬空「请看对照报告」换成真实命令与真实 ID——
        # 落盘记录必带 run_id/session_id/session_revision(training_runs.py 落盘形状),
        # 缺键的旧记录/极简夹具回退大写占位符,不假装知道 ID。
        session_id = record.get("session_id") or "SESSION_ID"
        run_ref = record.get("run_id") or "RUN_ID"
        revision = record.get("session_revision")
        if revision is None:  # 显式判 None:revision 是 int,or 链会把 0 误判成缺键
            revision = "REVISION"
        lines.append(
            f"训练完成只说明产出了模型；效果要用同一套开发题与基座对照来判断，"
            f"可运行 eval-compare {session_id} {run_ref} --revision {revision} 生成对照报告"
            "（改进轮次的子训练加 --iteration-id）。"
        )
    # 方案快照的工具核查轨迹(plan_trace):不经方案的直接启动为空,不渲染轨迹行。
    lines.extend(summarize_tool_trace(record.get("plan_trace"), "训练方案"))
    return lines


_ACCEPTANCE_METRIC_NAMES = {
    "exact_match": "严格匹配",
    "pass_rate": "自定义规则通过率",
    "manual_acceptance_rate": "人工逐题判断",
}


def acceptance_gate_lines(
    minimum_score: float | None,
    minimum_cases: int | None,
    test_count: int | None,
) -> list[str]:
    """验收门槛分辨率算术（单一来源，页面冻结表单与验收摘要同词汇）。

    「最低通过率 90%」配不同题数含义完全不同：10 道题容错 1 道、5 道题容错 0 道。
    只做确定性算术：把通过率门槛换算成需通过题数与容错题数，并给出每题占多少
    个百分点；最低题数超过固定测试题实际题数时预告执行必判证据不足。门槛数值
    本身仍由用户设定，软件不代设、也不作统计结论。
    """
    if (
        not isinstance(minimum_score, (int, float))
        or isinstance(minimum_score, bool)
        or not 0 <= minimum_score <= 1
        or not isinstance(minimum_cases, int)
        or isinstance(minimum_cases, bool)
        or not isinstance(test_count, int)
        or isinstance(test_count, bool)
        or test_count <= 0
    ):
        return []
    # 用与服务判定同一式的浮点除法（k/N >= 门槛）找最小通过题数：
    # 验收执行按 accepted/total 的浮点商与门槛比较，这里逐位同式，算术才与判定一致。
    required = next(k for k in range(test_count + 1) if k / test_count >= minimum_score)
    tolerance = test_count - required
    lines = [
        f"按 {minimum_score:.0%} 通过率门槛与 {test_count} 道最终测试题算："
        f"需通过 {required} 道、最多容错 {tolerance} 道未通过"
        f"（每题占通过率 {100 / test_count:.5g} 个百分点）。"
    ]
    if tolerance <= 0:
        lines.append(
            "容错为 0 道：任何一题未通过（生成失败与截断都按未通过计）都会判未达标——"
            "小题集配高门槛时，单题偶然误差会直接决定结论。"
        )
    if minimum_cases > test_count:
        lines.append(
            f"最低测试题数 {minimum_cases} 道超过这套固定测试题的实际 {test_count} 道——"
            "按此条款执行验收必判「证据不足」，需先固定题数更多的测试题或调低最低题数。"
        )
    return lines


# 验收结论枚举 → 人话名的单一来源（公开常量）：报告摘要的终态结论句与页面验收
# 横幅从这里取名。pending_run/pending_review 在摘要里是带动态事实的场景长句、
# 在漏斗里是紧凑短名——同枚举的场景化分层，键集由 test_report_summary 的
# test_acceptance_decision_names_single_source_pins 键集等值钉锁定，不在此强求同值。
ACCEPTANCE_DECISION_NAMES = {
    "passed": "达到运行前冻结的验收标准",
    "failed": "未达到运行前冻结的验收标准",
    "insufficient_evidence": "证据不足，不能确认可交付",
    "pending_review": "等待逐题业务判断",
    "pending_run": "等待按冻结条款执行",
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
    # 门槛分辨率算术与页面冻结表单同源（acceptance_gate_lines 单一来源）。
    suite = record.get("evaluation_suite")
    test_count = (suite.get("case_counts") or {}).get("test") if isinstance(suite, dict) else None
    lines.extend(acceptance_gate_lines(minimum_score, minimum_cases, test_count))

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
        # 终态结论名与页面验收横幅同源（ACCEPTANCE_DECISION_NAMES 单一来源,
        # R97 前页面手抄 map 已与摘要漂移:「已冻结」vs「运行前冻结」）。
        lines.append(f"当前结论：{ACCEPTANCE_DECISION_NAMES.get(decision, decision)}。")
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
    spec = record.get("task_spec") or {}
    if spec:
        # 冻结时点引用的规约四要素:与任务规约投影同源同词汇(spec_anchor_lines 单一来源)。
        from src.workbench.task_spec_projection import spec_anchor_lines

        lines.append("冻结时引用的任务规约口径：")
        lines.extend(spec_anchor_lines(spec))
    lines.append(
        "以上结论只对这次冻结的条款与固定测试题负责；达到标准也不会自动部署模型，"
        "是否交付由你按业务决定。"
    )
    return lines


# 迭代决策枚举 → 人话名的单一来源（公开常量）：报告摘要、页面决策表单与已决策回显
# 都从这里取名。漏斗的紧凑短名（采用/继续/停止/证据不足）是同枚举的场景化分层，
# 键集由 test_funnel_report 的键集等值钉锁定，不在此强求同值。
ITERATION_DECISION_NAMES = {
    "adopt": "采用本轮结果",
    "continue": "继续改进",
    "stop": "停止本轮路线",
    "insufficient_evidence": "证据不足",
}


def three_model_delta_lines(
    results: list[dict] | None, decision_metric: str = "exact_match"
) -> list[str]:
    """三模型对照分数差的题数换算（单一来源，页面决策卡与 CLI 状态摘要同词汇）。

    分数差不直观——0.625 对 0.750 是差 1 题还是差 5 题，没有统计背景难以分辨。
    只做确定性算术：把分数差换算成题数差，并给出每题占多少个百分点的分辨率；
    差距是否值得再投一轮，仍由用户结合失败形态与逐题核对判断。
    """
    if not results:
        return []
    verb = "通过" if decision_metric == "pass_rate" else "答对"

    def answered(item: dict) -> tuple[int, int] | None:
        metrics = item.get("metrics") or {}
        score, total = metrics.get(decision_metric), metrics.get("total")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total <= 0
        ):
            return None
        return round(score * total), total

    lines: list[str] = []
    total_seen = 0
    own = next((item for item in results if item.get("label") == "本轮微调"), None)
    if own is None:
        return []
    for other_label, other_name in (("父轮模型", "父轮"), ("基座", "基座")):
        other = next((item for item in results if item.get("label") == other_label), None)
        if other is None:
            continue
        own_counts, other_counts = answered(own), answered(other)
        if own_counts is None or other_counts is None:
            continue
        own_count, total = own_counts
        other_count, other_total = other_counts
        if total != other_total:
            continue
        total_seen = total
        delta = own_count - other_count
        if delta > 0:
            changed = f"比{other_name}多{verb} {delta} 题"
        elif delta < 0:
            changed = f"比{other_name}少{verb} {-delta} 题"
        else:
            changed = f"与{other_name}{verb}题数相同"
        lines.append(
            f"{other_name}对照：本轮微调{changed}（{own_count}/{total} vs {other_count}/{total}）。"
        )
    if not lines:
        return []
    lines.append(
        f"开发集共 {total_seen} 道题，每题约占 {100 / total_seen:.5g} 个百分点——"
        "差距在 1 题量级时方向参考价值有限，是否值得再投一轮还要结合失败形态"
        "（截断/生成失败/复述指令）与逐题核对判断；以上是题数算术，不是统计结论。"
    )
    return lines


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
        lines.append(f"已记录你的业务决定：{ITERATION_DECISION_NAMES.get(decision, decision)}。")
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
        # 题数差算术与页面决策卡同源（three_model_delta_lines 单一来源）。
        lines.extend(
            three_model_delta_lines(
                record.get("results"), record.get("decision_metric") or "exact_match"
            )
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
        # 命令行恢复出口(R99):「勾选」是页面词汇,CLI 用户没有可勾的框——真实恢复
        # 命令按记录三键插值;缺键的旧记录/极简夹具回退大写占位符,不假装知道 ID。
        session_id = record.get("session_id") or "SESSION_ID"
        iteration_id = record.get("iteration_id") or "ITERATION_ID"
        revision = record.get("session_revision")
        if revision is None:  # 显式判 None:revision 是 int,or 链会把 0 误判成缺键
            revision = "REVISION"
        lines.append(
            f"命令行恢复：可运行 iteration-execute {session_id} {iteration_id}"
            f" --revision {revision} --acknowledge-warnings 继续本轮执行。"
        )
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
        # worker.log 位置插值(R99):记录在启动时落盘 log_path,据此给出完整位置;
        # 旧记录缺该键时回退泛指句——不编造不存在的路径。
        log_path = record.get("log_path")
        if log_path:
            lines.append(f"执行日志 worker.log 位于 {log_path}；处理问题后提出新的改进轮次。")
        else:
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
        lines.extend(summarize_tool_trace(record.get("trace"), "评分"))
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
    lines.extend(summarize_tool_trace(record.get("trace"), "评分"))
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


def summarize_tool_trace(trace: list[dict] | None, subject: str = "解读") -> list[str]:
    """工具核查轨迹的统一人话行：评测解读与数据分析共用同一格式（单一来源）。"""
    entries = [item for item in (trace or []) if isinstance(item, dict)]
    if not entries:
        return []
    ok_count = sum(1 for item in entries if item.get("ok"))
    failed = len(entries) - ok_count
    text = f"工具核查轨迹：{len(entries)} 次调用，成功 {ok_count} 次"
    if failed:
        text += f"、失败 {failed} 次——失败的调用没有取到证据，{subject}只依赖成功的调用"
    return [text + "。"]


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
    lines.extend(summarize_tool_trace(record.get("tool_trace"), "解读"))
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


def summarize_analysis(analysis: dict, tool_trace: list[dict] | None = None) -> list[str]:
    """analyze 尾行:与页面「数据判断与待确认问题」区同词汇的发现与待确认问题翻译。

    传入 tool_trace 时以评测解读同一格式渲染工具核查轨迹。
    """
    findings = analysis.get("findings") or []
    questions = analysis.get("questions") or []
    gaps = analysis.get("capability_gaps") or []
    approach = analysis.get("training_approach") or ""
    next_steps = analysis.get("next_steps") or []
    boundary = (
        "以上发现中「已观察」是数据里的事实，其余是待确认的推断或业务解释；"
        "分析待你确认并经真实预览核对，不代表业务效果达标。"
    )
    if not findings and not questions and not approach and not next_steps:
        return ["这份分析没有可读的内容。", *summarize_tool_trace(tool_trace, "分析"), boundary]
    lines = [
        f"这份分析给出数据判断与待确认问题：发现 {len(findings)} 条、待确认问题 {len(questions)} 个。"
    ]
    kinds = {
        "observed": "已观察",
        "hypothesis": "待验证推断",
        "needs_business_input": "需要业务解释",
        "needs_full_data": "需要全量验证",
    }
    for finding in findings:
        label = kinds.get(finding.get("kind"), finding.get("kind") or "发现")
        message = finding.get("message") or ""
        row_ids = finding.get("evidence_row_ids") or []
        evidence = f"（证据：{', '.join(row_ids)}）" if row_ids else ""
        lines.append(f"{label}：{message} {evidence}".rstrip())
    for question in questions:
        text = question.get("question") or ""
        why = question.get("why") or ""
        options = question.get("options") or []
        line = f"待确认问题：{text}"
        if why:
            line += f"——{why}"
        if options:
            line += f"（可选解释：{' / '.join(options)}）"
        lines.append(line)
    for gap in gaps:
        lines.append(f"当前能力缺口：{gap}")
    if approach:
        lines.append(f"暂定微调思路：{approach}")
    if next_steps:
        lines.append(f"下一步：{'；'.join(next_steps)}")
    lines.extend(summarize_tool_trace(tool_trace, "分析"))
    lines.append(boundary)
    return lines


def summarize_registration(status: dict) -> list[str]:
    """train-lineage 尾行:训练运行在模型库的注册状态(正向血缘)人话翻译。

    registered 点名全部版本与别名;not_registered 复述服务给出的注册命令原文;
    查询失败如实报错不编造。页面训练记录区渲染同一份摘要,页面与 CLI 同源同词汇。
    """
    if not isinstance(status, dict) or not status.get("status"):
        return ["这份注册状态没有可读的内容。"]
    boundary = "注册只说明模型库记录了这次训练的产物与血缘，不代表业务效果达标。"
    state = status["status"]
    if state == "registered":
        parts = []
        for version in status.get("versions") or []:
            entry = f"{version.get('name', '?')} v{version.get('version', '?')}"
            aliases = version.get("aliases") or []
            if aliases:
                entry += f"（{','.join(aliases)}）"
            parts.append(entry)
        head = (
            f"这次训练已注册到模型库：{'、'.join(parts)}。"
            if parts
            else "这次训练已注册到模型库，但没有返回版本清单。"
        )
        return [head, boundary]
    if state == "not_registered":
        lines = [status.get("message") or "这次训练尚未注册到模型库。"]
        how_to = status.get("how_to_register")
        if how_to:
            lines.extend(how_to.splitlines())
        lines.append(boundary)
        return lines
    if state in {"mlflow_unavailable", "lookup_failed"}:
        message = status.get("message")
        return [message] if message else [f"模型库查询返回状态 {state}，没有更多说明。"]
    return [f"模型库查询返回未知状态 {state}。"]


def summarize_lineage(result: dict) -> list[str]:
    """registry_cli lineage 的反向血缘人话翻译:模型版本 → 训练运行 → 数据版本。

    workbench 态逐行给出五个要素,缺项如实显示「-」(不显示 None);外部来源、
    无来源与查询失败如实说明;mlflow 未安装时该状态本身不携带 message,不编造。
    """
    if not isinstance(result, dict) or not result.get("status"):
        return ["这份血缘记录没有可读的内容。"]
    state = result["status"]
    model = result.get("model") or "-"
    if state == "workbench":
        digest = result.get("config_digest")
        digest_text = f"{digest[:12]}…" if digest else "-"
        return [
            f"模型：{model}",
            f"训练运行：{result.get('workbench_run_id') or '-'}",
            f"数据版本：{result.get('dataset_version') or '-'}",
            f"训练数据：{result.get('training_dataset') or '-'}",
            f"配置摘要：{digest_text}",
            "以上血缘把模型、训练运行与数据版本关联起来，只保证可追溯，不代表业务效果达标。",
        ]
    if state == "external":
        lines = [f"模型：{model}", f"基座模型：{result.get('base_model') or '-'}"]
    elif state == "no_source_run":
        lines = [f"模型：{model}"]
    elif state == "mlflow_unavailable":
        return [f"模型：{model}", "未安装 mlflow，无法查询这份血缘。"]
    elif state == "lookup_failed":
        lines = [f"模型：{model}"]
    else:
        return [f"模型：{model}", f"血缘查询返回未知状态 {state}。"]
    message = result.get("message")
    if message:
        lines.append(message)
    return lines


def summarize_export(record: dict) -> list[str]:
    """train-export 尾行:合并导出(环节⑨交接)人话翻译。

    exported 点名输出目录与证据文件;already_exported 幂等说明;ready 给可照抄
    命令;阻塞态逐条复述原因。页面「📦 合并导出」折叠区渲染同一份摘要,
    页面与 CLI 同源同词汇——收尾边界与验收/采用记录同口径:导出不自动部署。
    """
    if not isinstance(record, dict) or not record.get("status"):
        return ["这份导出记录没有可读的内容。"]
    boundary = "导出只产出模型文件与证据记录，不代表业务效果达标，也不会自动部署。"
    state = record["status"]
    if state == "exported":
        return [
            f"合并导出完成：这次训练的适配器已并入基础模型，输出目录 {record.get('output_dir') or '?'}。",
            "这个目录包含完整模型与分词器，可被 vLLM、Ollama、LM Studio 直接加载；"
            "目录里的 export_evidence.json 记录了它来自哪次训练与哪个数据版本。",
            boundary,
        ]
    if state == "already_exported":
        return [
            f"这次训练此前已合并导出到 {record.get('output_dir') or '?'}；目录已完整，重复导出不会改变模型内容。",
            boundary,
        ]
    if state == "ready":
        return [
            "这次训练的产物已齐备，可合并导出成可被 vLLM、Ollama、LM Studio 直接加载的完整模型：",
            f"python scripts/data_intake.py train-export {record.get('run_id') or 'RUN_ID'}",
            f"默认输出目录：{record.get('output_dir') or '-'}。",
            boundary,
        ]
    reasons = record.get("reasons") or []
    if reasons:
        return [*reasons, boundary]
    return [f"导出盘点返回状态 {state}，没有更多说明。"]
