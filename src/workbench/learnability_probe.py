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
