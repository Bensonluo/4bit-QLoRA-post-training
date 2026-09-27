"""Plain-language comparison summaries: the non-expert reads sentences, not tables.

Deterministic, honest, and bounded: it restates observed facts (correct counts,
truncation, instruction echo, sample size) and their limits. It never claims
business success, never recommends adoption, and never speculates about causes
beyond what the report's own diagnostics observed.
"""

from __future__ import annotations

from typing import Any


def summarize_comparison(report: Any) -> list[str]:
    """Turn a comparison report into a few honest sentences for a non-expert."""
    models = report.models
    if not models:
        return ["该报告没有模型结果。"]
    lines: list[str] = []
    total = models[0]["metrics"].get("total") or 0
    lines.append(f"这次对照在固定开发集的 {total} 道题上进行,所有模型用同样的题目和评分规则。")

    from src.workbench.evaluation_diagnostics import count_instruction_echo

    best_label, best_correct, best_score = None, -1, -1.0
    for model in models:
        metrics = model["metrics"]
        rows = model["rows"]
        score_value = metrics.get("exact_match")
        correct = round((score_value or 0.0) * total)
        truncated = sum(row.get("status") == "truncated" for row in rows)
        failed = sum(row.get("status") == "failed" for row in rows)
        echo = count_instruction_echo(rows)
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

    if best_score == 0:
        lines.append(
            "没有一个模型答对任何题:目前不能说任何模型学会了这个任务,常见原因是题目太难、数据太少或提示格式不匹配,可查看每题的完整输出再判断。"
        )
    elif best_score == 1.0:
        lines.append(
            f"{best_label}在本次题目上全部答对;但题目只有 {total} 道,样本很小,不能据此断定业务上足够好。"
        )
    else:
        lines.append(
            f"答对最多的是{best_label}({best_correct}/{total});请结合逐题输出判断答错的部分是否可接受。"
        )
    if 0 < total < 20:
        lines.append(f"注意:开发集只有 {total} 道题,任何百分比都受单题影响很大,只当方向参考。")
    lines.append("以上是观察事实,不是业务达标结论;是否采用仍由你按业务标准决定。")
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
