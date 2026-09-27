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
        parts = [f"{model['label']}答对 {correct}/{total}"]
        if truncated:
            parts.append(f"{truncated} 题没写完被截断")
        if failed:
            parts.append(f"{failed} 题生成失败")
        if echo:
            parts.append(f"{echo} 题在复述题目而不是作答")
        if not (truncated or failed or echo) and correct == total:
            parts.append("全部答对")
        lines.append("· " + "、".join(parts) + "。")
        score = score_value or 0.0
        if score > best_score:
            best_label, best_correct, best_score = model["label"], correct, score

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

    if best_score == 0:
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
    if best_score > 0 and base_stats and tuned_stats:
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
    if 0 < total < 20:
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
            "分界线与本版本实际使用的时间字段见下方。"
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
