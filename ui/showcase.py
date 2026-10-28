"""Read-only public examples and recorded validation; never a training store."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.workbench.demo_task import demo_create_payload, parse_demo_sample
from ui.components.format import finite_number
from ui.study_archive import PROTOCOLS, _verify_scores

R149_SOURCE = "complete-output-handoff-live-2026-10-22.json"
R148_SOURCE = "sft-answer-supervision-live-2026-10-22.json"


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(f"公开历史记录无法核验：{message}")


def _digest(value: Any) -> str:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _read(root: Path, name: str) -> tuple[dict, dict]:
    try:
        raw = (Path(root) / "docs/validation" / name).read_bytes()
        value = json.loads(raw)
        _require(isinstance(value, dict), "文件必须是 JSON 对象。")
        return value, {
            "name": name,
            "path": f"docs/validation/{name}",
            "date": value["date"],
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except (OSError, UnicodeError, ValueError, KeyError) as exc:
        raise ValueError(f"无法读取公开历史记录 {name}：{exc}") from exc


def _number(value: Any) -> bool:
    return type(value) in (int, float) and finite_number(value)


def _verify_report(report: dict) -> None:
    protocol, dataset, models = report["protocol"], report["dataset"], report["models"]
    _require(
        dataset["purpose"] == "development_only"
        and dataset["split"] == "validation"
        and dataset["row_count"] == 8,
        "必须保留固定八道开发题。",
    )
    _require(
        protocol["scorer"] == "classification_exact"
        and protocol["generation"] == "greedy_token_contract_v2"
        and protocol["max_new_tokens"] == 16,
        "生成或分类评分协议不一致。",
    )
    _require(
        len(models) == 4 and len({model["label"] for model in models}) == 4,
        "四个独立模型记录不完整。",
    )
    questions = None
    inputs: dict[int, tuple[list, list]] = {}
    failures = False
    for model in models:
        rows = model["rows"]
        _require(
            len(rows) == 8 and [row["index"] for row in rows] == list(range(8)),
            "逐题记录缺失或重复。",
        )
        current = [(row["prompt"], row["expected"], row["source"]) for row in rows]
        _require(questions is None or current == questions, "模型之间的题目、答案或来源不同。")
        questions = current
        _require(
            protocol["prompt_digest"] == _digest([row["prompt"] for row in rows])
            and protocol["answers_digest"] == _digest([row["expected"] for row in rows]),
            "提示或答案指纹不一致。",
        )
        correct = scored = generated = 0
        for row in rows:
            status, output = row["status"], row["output"]
            _require(status in {"scored", "truncated", "failed"}, "逐题状态不受支持。")
            failures |= status != "scored"
            if output is None:
                _require(
                    status == "failed"
                    and row.get("token_proof") is None
                    and row["correct"] is False
                    and row.get("truncated") is False
                    and "generated_tokens" in row
                    and (
                        row["generated_tokens"] is None
                        or (
                            type(row["generated_tokens"]) is int
                            and 0 <= row["generated_tokens"] <= 16
                        )
                    ),
                    "失败记录不能冒充有效生成。",
                )
                continue
            _require(isinstance(output, str), "原始输出必须是文本。")
            proof = row["token_proof"]
            ids, produced, eos = (
                proof[key] for key in ("prompt_token_ids", "generated_token_ids", "eos_token_ids")
            )
            _require(
                all(
                    isinstance(values, list)
                    and values
                    and all(type(token) is int and token >= 0 for token in values)
                    for values in (ids, produced, eos)
                ),
                "生成 token 证据不完整。",
            )
            _require(
                len(eos) == len(set(eos))
                and 0 < len(produced) <= 16
                and row["generated_tokens"] == len(produced),
                "token 数或 EOS 证据不一致。",
            )
            _require(
                proof["prompt_digest"] == _digest(row["prompt"])
                and proof["output_digest"] == _digest(output),
                "生成证据与原始文本不一致。",
            )
            ended = produced[-1] in eos
            _require(
                not any(token in eos for token in produced[:-1])
                and proof["finish_reason"] == ("eos" if ended else "length")
                and row["truncated"] is (not ended)
                and (ended or len(produced) == 16),
                "停止或截断证据不一致。",
            )
            _require(
                row["index"] not in inputs or inputs[row["index"]] == (ids, eos),
                "同题输入 token 或 EOS 不一致。",
            )
            inputs[row["index"]] = (ids, eos)
            expected = row["expected"]
            _require(isinstance(expected, str), "标准答案缺失。")
            matches = (
                output.strip() == expected.strip()
                if protocol["strip_whitespace"]
                else output == expected
            )
            expected_correct = status == "scored" and matches
            _require(
                type(row["correct"]) is bool and row["correct"] == expected_correct,
                "逐题分数与原始输出不一致。",
            )
            _require((status == "truncated") == row["truncated"], "截断状态与生成结果矛盾。")
            generated += 1
            scored += status == "scored"
            correct += expected_correct
        metrics = model["metrics"]
        expected_metrics = {
            "total": 8,
            "scored": scored,
            "generated": generated,
            "exact_match": correct / 8,
            "scorable_coverage": scored / 8,
            "generation_coverage": generated / 8,
        }
        _require(
            all(
                _number(metrics.get(key)) and metrics[key] == value
                for key, value in expected_metrics.items()
            ),
            "模型汇总分数与全部八题不一致。",
        )
    _require(
        report["status"] == ("completed_with_failures" if failures else "completed"),
        "报告总状态掩盖了失败或截断。",
    )


def _handoff_development_groups(report: dict, preparation: dict) -> int | None:
    """Count observed groups only when the preparation metadata corroborates them."""
    try:
        groups = [row["source"]["group"] for row in report["models"][0]["rows"]]
        if not groups or any(
            not isinstance(group, dict)
            or len(group) != 1
            or any(not isinstance(value, str) or not value.strip() for value in group.values())
            for group in groups
        ):
            return None
        observed = {_digest(group) for group in groups}
        recorded = preparation["development_groups"]
        if (
            not isinstance(recorded, list)
            or any(not isinstance(value, str) or not value.strip() for value in recorded)
            or len(recorded) != len(set(recorded))
            or len(recorded) != len(observed)
            or {next(iter(group.values())) for group in groups} != set(recorded)
        ):
            return None
        return len(observed)
    except (KeyError, TypeError, IndexError, AttributeError):
        return None


def _study_development_groups(data: dict, evaluations: list[tuple[str, dict]]) -> int | None:
    """Optional group metadata must bind to this study's actual questions and prompts."""
    from src.data.sft_protocol import render_sft_prompt

    try:
        cases = data["controlled_v2"]["data"]["development"]
        if not isinstance(cases, list) or len(cases) != 16:
            return None
        by_id = {}
        for case in cases:
            if (
                any(
                    not isinstance(case[key], str)
                    for key in ("case_id", "form", "input", "instruction", "output", "group_id")
                )
                or not case["group_id"].strip()
            ):
                return None
            key = (case["case_id"], case["form"])
            if key in by_id:
                return None
            by_id[key] = case
        for protocol, evaluation in evaluations:
            rows = evaluation["rows"]
            if len(rows) != len(by_id) or {(row["case_id"], row["form"]) for row in rows} != set(
                by_id
            ):
                return None
            for row in rows:
                case = by_id[(row["case_id"], row["form"])]
                if row["expected"] != case["output"]:
                    return None
                if protocol == PROTOCOLS[2]:
                    # The recorded Qwen native prompt has a distinct user block.
                    # Do not pretend it is an Alpaca prompt or load a tokenizer.
                    prefix = "<|im_start|>user\n"
                    content, marker, _ = row["prompt"][len(prefix) :].partition("<|im_end|>")
                    expected = case["instruction"] + (
                        "\n\n" + case["input"] if case["input"] else ""
                    )
                    if not row["prompt"].startswith(prefix) or not marker or content != expected:
                        return None
                elif row["prompt"] != render_sft_prompt(case, None, protocol):
                    return None
        return len({case["group_id"] for case in cases})
    except (KeyError, TypeError, IndexError, AttributeError, ValueError):
        return None


def load_showcase(root: Path) -> dict:
    """Load the original sanitized R149 report and safe presentation metadata."""
    data, source = _read(root, R149_SOURCE)
    try:
        report, history, training = (
            data["actual_report"],
            data["actual_lr_and_optimizer_steps"],
            data["training"],
        )
        _verify_report(report)
        _require(
            len(history) == 64
            and all(type(point["step"]) is int for point in history)
            and [point["step"] for point in history] == list(range(1, 65)),
            "64 次真实更新日志不完整。",
        )
        _require(
            all(
                _number(point["loss"]) and point["loss"] >= 0 and _number(point["learning_rate"])
                for point in history
            ),
            "训练曲线存在无效数值。",
        )
        _require(
            training["global_step"] == training["loss_log_count"] == 64
            and training["status"] == "succeeded",
            "训练回执与日志不一致。",
        )
        rate = data["actual_config"]["training"]["learning_rate"]
        _require(
            all(point["learning_rate"] == rate for point in history)
            and training["learning_rates"] == [rate]
            and training["metrics"]["loss"] == history[-1]["loss"],
            "训练日志、学习率或末次 loss 与实际配置/回执不一致。",
        )
        verification = data["independent_verification"]
        _require(
            verification["status"] == "verified"
            and verification["evaluation_id"] == report["evaluation_id"]
            and verification["comparison_key"] == report["comparison_key"]
            and verification["report_status"] == report["status"]
            and verification["missing_generation_proofs"] == [],
            "独立回读记录与原始报告不一致。",
        )
        config = {
            group: {
                key: value
                for key, value in values.items()
                if key
                not in {
                    "output_dir",
                    "train_file",
                    "validation_file",
                    "dataset_name",
                    "mlflow_tracking_uri",
                    "log_dir",
                }
            }
            for group, values in data["actual_config"].items()
            if isinstance(values, dict)
        }
        config["model"]["name"] = str(config["model"]["name"]).replace("\\", "/").rsplit("/", 1)[-1]
        artifacts = {
            name: {key: value for key, value in item.items() if key != "directory"}
            for name, item in data["artifact_inventory"]["artifacts"].items()
        }
        _require(set(artifacts) == {"hf", "gguf_q4_k_m"}, "交付格式记录不完整。")
        for kind, item in artifacts.items():
            receipt = data["export_receipts"]["hf" if kind == "hf" else "gguf"]
            _require(
                item["source_run_id"] == receipt["run_id"] == training["run_id"],
                "交付产物未关联实际训练。",
            )
            _require(item["weight_files"], "交付模型缺少权重文件记录。")
            _require(
                type(item["directory_bytes"]) is int
                and finite_number(item["directory_bytes"])
                and item["directory_bytes"]
                >= sum(meta["size"] for meta in item["weight_files"].values()),
                "交付目录大小小于已记录的模型文件。",
            )
            for name, metadata in item["weight_files"].items():
                _require(
                    metadata == receipt["output_files"][name]
                    and type(metadata["size"]) is int
                    and finite_number(metadata["size"])
                    and metadata["size"] > 0
                    and re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"]) is not None,
                    "模型大小或内容摘要与导出回执不一致。",
                )
            _require(
                finite_number(sum(meta["size"] for meta in item["weight_files"].values())),
                "模型权重总大小超出可核验范围。",
            )
        _require(
            data["export_receipts"]["gguf"]["business_evaluation_status"] == "not_run",
            "导出回执不应继承后续评测分数。",
        )
        costs = data["costs"]
        _require(
            isinstance(costs, dict)
            and all(
                value is None or (_number(value) and value >= 0)
                for key, value in costs.items()
                if key.endswith("_seconds") or key == "electricity_kwh"
            ),
            "阶段耗时或电量记录不完整。",
        )
        return {
            "id": "r149",
            "title": "完整分类任务 · HF 与 GGUF 交接",
            "date": data["date"],
            "scope": data["scope"],
            "report": report,
            "history": history,
            "counts": {
                "optimizer_steps": training["global_step"],
                "training_rows": data["preparation"]["row_counts"]["train"],
                "development_rows": report["dataset"]["row_count"],
                "development_groups": _handoff_development_groups(report, data["preparation"]),
                "models": len(report["models"]),
            },
            "config": config,
            "training": {
                key: training[key]
                for key in (
                    "status",
                    "run_id",
                    "finished_at",
                    "elapsed_seconds",
                    "global_step",
                    "loss_log_count",
                    "learning_rates",
                    "worker_status",
                    "worker_exit_observed",
                    "metrics",
                    "scope",
                )
                if key in training
            },
            "costs": costs,
            "artifacts": artifacts,
            "source": source,
            "limitations": data["limitations"],
        }
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as exc:
        raise ValueError(f"公开历史记录 {R149_SOURCE} 缺损或不一致：{exc}") from exc


_EXAMPLE_SPECS = (
    ("classification", None, "售后工单分类", "类别", ["客户描述"], "violet", "category"),
    ("extraction", "extraction", "工单编号抽取", "工单号", ["客户描述"], "cyan", "data_object"),
    ("generation", "generation", "客服答复生成", "标准答复", ["客户问题"], "rose", "chat"),
)


def _example_task(root: Path, spec: tuple) -> dict:
    task_id, variant, title, target, inputs, accent, icon = spec
    try:
        payload = demo_create_payload(Path(root), variant)
    except ValueError as exc:
        raise ValueError(f"内置示例 {title} 无法读取：{exc}") from exc
    if payload is None:
        raise ValueError(f"内置示例 {title} 的 CSV 文件缺失。")
    goal, raw, name, description = payload
    try:
        rows = parse_demo_sample(raw, variant)
    except ValueError as exc:
        raise ValueError(f"内置示例 {title} 无法读取：{exc}") from exc
    return {
        "id": task_id,
        "title": title,
        "description": description,
        "goal": goal,
        "name": name,
        "bytes": raw,
        "rows": rows,
        "row_count": len(rows),
        "target_column": target,
        "input_columns": inputs,
        "accent": accent,
        "icon": icon,
        "kind": "fictional_example",
    }


def example_tasks(root: Path) -> list[dict]:
    """Return all fictional examples, failing if any source is unavailable."""
    return [_example_task(root, spec) for spec in _EXAMPLE_SPECS]


def example_tasks_tolerant(root: Path) -> tuple[list[dict], list[dict]]:
    """Return readable examples and identify unavailable ones without local paths."""
    tasks: list[dict] = []
    errors: list[dict] = []
    for spec in _EXAMPLE_SPECS:
        try:
            tasks.append(_example_task(root, spec))
        except (OSError, ValueError, csv.Error):
            errors.append({"id": spec[0], "title": spec[2]})
    return tasks, errors


_CATALOG_RECORDS = (
    ("r149", "完整分类任务 · HF 与 GGUF 交接", R149_SOURCE),
    ("r148-controlled-v2", "监督协议受控对照 · 同次三臂", R148_SOURCE),
    ("r148-product-smoke", "仅答案监督 · 产品链路验证", R148_SOURCE),
)
_CATALOG_ERRORS = (
    OSError,
    UnicodeError,
    KeyError,
    TypeError,
    IndexError,
    ValueError,
    AttributeError,
)


def _handoff_catalog_entry(root: Path) -> dict:
    handoff = load_showcase(root)
    return {
        "id": "r149",
        "date": handoff["date"],
        "title": handoff["title"],
        "kind": "workbench_handoff",
        "scope": handoff["scope"],
        "counts": dict(handoff["counts"]),
        "results": [
            {
                "label": model["label"],
                "correct": sum(row["correct"] for row in model["rows"]),
                "total": 8,
                "truncated": sum(row["truncated"] for row in model["rows"]),
            }
            for model in handoff["report"]["models"]
        ],
        "sourcefile": handoff["source"]["name"],
        "source": handoff["source"],
    }


def _controlled_catalog_entry(data: dict, source: dict) -> dict:
    controlled = data["controlled_v2"]
    arms = controlled["results"]["arms"]
    _require(set(arms) == set(PROTOCOLS), "受控实验必须保持同次三臂及其原始协议身份。")
    baselines = controlled["results"]["baseline"]
    _require(set(baselines) == {PROTOCOLS[0], PROTOCOLS[2]}, "受控实验的两种提示基座记录不完整。")
    for evaluation in baselines.values():
        _verify_scores(evaluation["rows"], evaluation["summary"])
    for protocol, arm in arms.items():
        _require(arm["protocol_identity"]["protocol"] == protocol, "训练臂协议身份与名称不一致。")
    results = [
        {
            "label": name,
            "correct": arm["evaluation"]["summary"]["all"]["accepted"],
            "total": arm["evaluation"]["summary"]["all"]["total"],
            "truncated": arm["evaluation"]["summary"]["all"]["truncated"],
        }
        for name, arm in arms.items()
    ]
    for arm in arms.values():
        _verify_scores(arm["evaluation"]["rows"], arm["evaluation"]["summary"])
    return {
        "id": "r148-controlled-v2",
        "date": data["date"],
        "title": "监督协议受控对照 · 同次三臂",
        "kind": "controlled_experiment",
        "scope": "合成机制实验；同次三臂，不是三轮业务迭代；native chat 仅实验。",
        "counts": {
            "arms": len(arms),
            "steps_per_arm": controlled["settings"]["steps"],
            "training_rows": len(controlled["data"]["train"]),
            "development_rows": len(next(iter(arms.values()))["evaluation"]["rows"]),
            "development_groups": _study_development_groups(
                data,
                [*baselines.items(), *[(name, arm["evaluation"]) for name, arm in arms.items()]],
            ),
        },
        "results": results,
        "sourcefile": source["name"],
        "source": dict(source),
    }


def _product_catalog_entry(data: dict, source: dict) -> dict:
    smoke = data["product_smoke"]
    evaluations = smoke["results"]["evaluations"]
    _require(set(evaluations) == {"base", "trained"}, "产品链路必须保留基座与训练模型。")
    results = [
        {
            "label": name,
            "correct": item["summary"]["all"]["accepted"],
            "total": item["summary"]["all"]["total"],
            "truncated": item["summary"]["all"]["truncated"],
        }
        for name, item in evaluations.items()
    ]
    for item in evaluations.values():
        _verify_scores(item["rows"], item["summary"])
    return {
        "id": "r148-product-smoke",
        "date": data["date"],
        "title": "仅答案监督 · 产品链路验证",
        "kind": "product_smoke",
        "scope": "独立产品 smoke；不属于前述三臂受控实验，不代表真实业务验收。",
        "counts": {
            "optimizer_steps": smoke["results"]["training"]["global_step"],
            "development_rows": results[0]["total"],
            "development_groups": (
                _study_development_groups(
                    data, [(PROTOCOLS[1], item) for item in evaluations.values()]
                )
                if smoke.get("data_source") == "controlled_v2.data"
                else None
            ),
            "models": len(evaluations),
        },
        "results": results,
        "sourcefile": source["name"],
        "source": dict(source),
    }


def history_catalog(root: Path) -> list[dict]:
    """Strict three-study catalog: any unavailable record still raises."""
    handoff = _handoff_catalog_entry(root)
    data, source = _read(root, R148_SOURCE)
    try:
        return [
            handoff,
            _controlled_catalog_entry(data, source),
            _product_catalog_entry(data, source),
        ]
    except _CATALOG_ERRORS as exc:
        raise ValueError(f"公开历史目录 {R148_SOURCE} 缺损或不一致：{exc}") from exc


def available_history_catalog(root: Path) -> tuple[list[dict], list[dict[str, str]]]:
    """Keep each healthy public study and disclose failed records independently.

    Error text intentionally contains no underlying exception or server path.
    This is a tolerant presentation API; history_catalog remains strict.
    """
    entries: list[dict] = []
    errors: list[dict[str, str]] = []

    def unavailable(index: int) -> None:
        identity, title, name = _CATALOG_RECORDS[index]
        errors.append(
            {
                "id": identity,
                "title": title,
                "sourcefile": name,
                "message": "来源文件缺失、不可读或内容核验未通过；未显示该档案。",
            }
        )

    try:
        entries.append(_handoff_catalog_entry(root))
    except _CATALOG_ERRORS:
        unavailable(0)
    try:
        data, source = _read(root, R148_SOURCE)
    except _CATALOG_ERRORS:
        unavailable(1)
        unavailable(2)
    else:
        for index, builder in ((1, _controlled_catalog_entry), (2, _product_catalog_entry)):
            try:
                entries.append(builder(data, source))
            except _CATALOG_ERRORS:
                unavailable(index)
    return entries, errors
