"""Learnability probe: cheapest evidence that this data can teach this task at all.

Runs the base model zero-shot on a small fixed development sample and compares
exact-match with the majority-class guess baseline. The output is evidence with
explicit sample-size limits — never a verdict that the task is impossible, and
never a claim of business quality.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.workbench.sources import content_digest


def probe_learnability(
    session,
    base_model_path: str,
    *,
    runtime_factory=None,
    sample_size: int = 8,
    max_new_tokens: int = 32,
) -> dict:
    """Zero-shot probe on the development split; compare with majority baseline."""
    from src.workbench.training_preflight import _verified_partitions

    if type(sample_size) is not int or not 1 <= sample_size <= 50:
        raise ValueError("抽样数量必须是 1 到 50 之间的整数。")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须是正整数。")
    if session.dataset is None:
        raise ValueError("请先生成独立分区；探针使用固定开发集抽样。")
    recipe = session.analysis.recipe
    if recipe is None or recipe.output_format != "text" or len(recipe.targets) != 1:
        raise ValueError("可学性探针当前支持单字段文本答案任务；其他形态暂不支持。")
    verification = {"issues": []}
    rows = _verified_partitions(session, verification)["validation"]
    if not rows:
        raise ValueError("当前数据没有开发集样本，无法探测。")

    from src.workbench.business_evaluation import EvaluationModel

    model = EvaluationModel("基座零样本", base_model_path)
    runtime = (runtime_factory or _default_runtime)(model)
    sampled = rows[: min(sample_size, len(rows))]
    correct = 0
    observations = []
    try:
        from src.data.loaders import render_alpaca_prompt

        for row in sampled:
            prompt = render_alpaca_prompt({**row, "output": ""})
            generation = runtime.generate(prompt, _probe_protocol(max_new_tokens))
            answer = generation.text.strip()
            hit = answer == row["output"]
            correct += hit
            observations.append(
                {
                    "row_id": row["metadata"]["source_row_id"],
                    "expected": row["output"],
                    "generated": answer,
                    "match": hit,
                    "truncated": generation.truncated,
                }
            )
    finally:
        closer = getattr(runtime, "close", None)
        if closer is not None:
            closer()

    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["output"]] = label_counts.get(row["output"], 0) + 1
    majority_share = max(label_counts.values()) / len(rows) if label_counts else 0.0
    majority_label = max(label_counts, key=label_counts.get) if label_counts else ""
    accuracy = correct / len(sampled)

    # 标签问题候选:基座零样本与数据标签矛盾的行。基座没见过这些数据,
    # 它的分歧不带训练利益;若盲标核验中用户也不同意数据标签,则是强证据
    # (人机双信号交叉)——这正是 cleanlab 式统计检错的任务无关轻量版。
    blind = getattr(session, "label_verification", None) or {}
    blind_items = {
        item.get("row_id"): item
        for item in (blind.get("items") or [])
        if isinstance(item, dict) and item.get("match") is False
    }
    candidates = []
    for observation in observations:
        if observation["match"] or observation["truncated"]:
            continue  # 截断的生成不算分歧证据
        row_id = observation["row_id"]
        user_also_disagrees = row_id in blind_items
        candidates.append(
            {
                "row_id": row_id,
                "data_label": observation["expected"],
                "base_zero_shot": observation["generated"],
                "user_blind_answer": blind_items[row_id].get("submitted_answer")
                if user_also_disagrees
                else None,
                "evidence": (
                    "基座零样本与用户盲标都不认同数据标签——强证据,优先人工核对"
                    if user_also_disagrees
                    else "仅基座零样本不认同——模型可能错,标签也可能错,弱信号供参考"
                ),
            }
        )
    candidates.sort(key=lambda item: 0 if item["user_blind_answer"] else 1)
    result = {
        "kind": "learnability_probe",
        "dataset_version": session.dataset.version,
        "model_path": str(base_model_path),
        "protocol": {"generation": "greedy_v1", "max_new_tokens": max_new_tokens, "shots": 0},
        "sample_size": len(sampled),
        "dev_total": len(rows),
        "zero_shot_accuracy": accuracy,
        "majority_baseline": majority_share,
        "majority_label": majority_label,
        "difference": accuracy - majority_share,
        "observations": observations,
        "label_error_candidates": candidates,
        "candidates_note": (
            f"{len(candidates)} 行是标签问题候选(基座零样本与数据标签不一致;"
            f"其中 {sum(1 for c in candidates if c['user_blind_answer'])} 行与用户盲标也不一致)。"
            "候选不等于错误——模型可能错;但优先人工核对这些行是性价比最高的数据清理。"
        ),
        "note": (
            f"基座零样本 {accuracy:.0%} vs 全开发集多数类「{majority_label}」{majority_share:.0%}"
            f"（抽样 {len(sampled)}/{len(rows)} 条）。这只是数据与任务是否存在可学关联的"
            "最便宜证据：样本量小、零样本差异不能预测微调效果，也不构成业务达标结论；"
            "显著低于瞎猜基线通常意味着提示格式或任务定义需要先核查。"
        ),
    }
    return result


class _ProtocolLike:
    """Local stand-in with the fields LocalEvaluationRuntime.generate reads."""

    def __init__(self, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.strip_whitespace = True
        self.fields: tuple[str, ...] = ()
        self.custom_scoring = None


def _probe_protocol(max_new_tokens: int) -> _ProtocolLike:
    return _ProtocolLike(max_new_tokens)


def _default_runtime(model):
    from src.workbench.business_evaluation import LocalEvaluationRuntime

    return LocalEvaluationRuntime(model)


def save_probe(root: str | Path, result: dict) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{result['dataset_version']}-{content_digest(result)[:16]}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_latest_probe(root: str | Path, dataset_version: str) -> dict | None:
    """回读指定数据版本最近一次保存的探针结果;没有则返回 None。

    探针要真实加载本地基座模型,重跑成本高;结果存盘后必须能原样回看,
    页面刷新或切换会话不丢证据。按数据版本过滤:数据重物化后旧结果不会
    冒充新版本的证据。
    """
    root = Path(root)
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.glob(f"{dataset_version}-*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue  # 损坏的历史记录跳过,不阻塞回读
        if isinstance(data, dict) and data.get("kind") == "learnability_probe":
            return data
    return None


def candidates_to_csv(candidates: list[dict]) -> bytes:
    """候选表导出为 CSV(带 BOM,Excel 直开);供人工核对的离线清单。"""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["行ID", "数据标签", "基座零样本输出", "你的盲标答案", "证据"])
    for item in candidates:
        writer.writerow(
            [
                item.get("row_id", ""),
                item.get("data_label", ""),
                item.get("base_zero_shot", ""),
                item.get("user_blind_answer") or "",
                item.get("evidence", ""),
            ]
        )
    return buffer.getvalue().encode("utf-8-sig")
