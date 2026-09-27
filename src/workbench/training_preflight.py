"""Verify immutable partitions and actual tokenizer consumption without loading a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.workbench.intake_models import IntakeSession
from src.workbench.materialize import SPLITS, dataset_is_current
from src.workbench.sources import canonical


def load_local_tokenizer(path: str | Path, *, local_files_only: bool = True) -> Any:
    """Load tokenizer files only; downloads require an explicit caller opt-in."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=local_files_only, trust_remote_code=False
    )


def _issue(report, code, severity, message, *, split=None, row_ids=None):
    report["issues"].append(
        {
            "code": code,
            "severity": severity,
            "message": message,
            "split": split,
            "row_ids": row_ids or [],
        }
    )


def _finish(report):
    severities = {issue["severity"] for issue in report["issues"]}
    report["status"] = (
        "blocked"
        if "blocking" in severities
        else "warnings"
        if "warning" in severities
        else "passed"
    )
    return report


def _verified_partitions(session, report):
    from src.data_flywheel.dataset_registry import LocalDatasetRegistry

    artifact = session.dataset
    registry = LocalDatasetRegistry(artifact.registry_root)
    manifest = registry.get_split_manifest(artifact.name, artifact.version)
    splits = {
        split: registry.load_split(artifact.name, artifact.version, split) for split in SPLITS
    }
    recipe = session.analysis.recipe
    metadata = manifest["metadata"]
    temporal_policy = getattr(recipe, "temporal_split", None)
    temporal = None
    if temporal_policy is not None:
        from src.workbench.recipes import preview_recipe
        from src.workbench.temporal_split import temporal_assignment

        if session.full_data.preview != preview_recipe(session.full_data.source, recipe):
            raise ValueError("时间分区预览与实际来源或处理规则不一致。")
        if metadata.get("temporal_policy") != temporal_policy.model_dump():
            raise ValueError("分区清单时间策略与已确认方案不一致。")
        temporal = temporal_assignment(
            session.full_data.preview.rows, temporal_policy, recipe.group_columns
        )
        if metadata.get("excluded_rows") != temporal["excluded_rows"]:
            raise ValueError("排除台账与按来源重新计算的时间、窗口或排除理由不一致。")
    elif "temporal_policy" in metadata or "excluded_rows" in metadata:
        raise ValueError("普通分区不能附加未经确认的时间策略或排除记录。")
    if (
        metadata.get("source_digest") != artifact.source_digest
        or metadata.get("recipe_digest") != artifact.recipe_digest
        or metadata.get("sample_source_digest") != session.source.digest
        or metadata.get("group_columns") != recipe.group_columns
        or (not recipe.group_columns and metadata.get("independent_rows_confirmed") is not True)
    ):
        raise ValueError("分区清单的来源、处理规则或分组依据与当前确认不一致。")
    directory = Path(artifact.registry_root).resolve() / artifact.name / "splits" / artifact.version
    expected_paths = {split: directory / manifest["splits"][split]["path"] for split in SPLITS}
    expected_paths["manifest"] = directory / "manifest.json"
    if any(
        Path(artifact.paths.get(key, "")).resolve() != path for key, path in expected_paths.items()
    ):
        raise ValueError("任务记录的分区路径与已验签的不可变数据文件不一致。")
    config = artifact.data_config
    if (
        any(
            Path(config.get(key) or "").resolve() != expected_paths[split]
            for key, split in (
                ("dataset_name", "train"),
                ("train_file", "train"),
                ("validation_file", "validation"),
            )
        )
        or config.get("validation_split") != 0
        or config.get("dataset_loader") != "alpaca"
        or config.get("format") != "alpaca"
        or config.get("max_samples") is not None
    ):
        raise ValueError(
            "训练配置不匹配已确认分区：必须使用指定 Alpaca loader、完整 train/dev 文件且关闭再次切分。"
        )
    expected = {row.row_id: row for row in session.full_data.preview.rows}
    found: set[str] = set()
    seen: dict[tuple[str, str], tuple[str, str]] = {}
    for split, rows in splits.items():
        for row in rows:
            origin = row.get("metadata")
            row_id = origin.get("source_row_id") if isinstance(origin, dict) else None
            if not isinstance(row_id, str) or row_id not in expected or row_id in found:
                raise ValueError("分区来源行标识缺失、重复或不属于已确认的全量资料。")
            found.add(row_id)
            preview = expected[row_id]
            if (
                origin.get("source_digest") != artifact.source_digest
                or origin.get("group") != preview.group
                or row["instruction"] != recipe.instruction
                or row["input"] != preview.input
                or row["output"] != preview.target
                or (
                    origin.get("temporal") != temporal["times_by_row"][row_id]
                    if temporal is not None
                    else "temporal" in origin
                )
            ):
                raise ValueError(f"来源行 {row_id} 的输入、答案或分组与已确认预览不一致。")
            keys = [("input", row["input"])] + [
                (f"group:{column}", canonical(preview.group[column]))
                for column in recipe.group_columns
            ]
            for key in keys:
                previous = seen.setdefault(key, (split, row_id))
                if previous[0] != split:
                    _issue(
                        report,
                        "cross_split_leakage",
                        "blocking",
                        "同一业务对象或相同模型输入出现在不同分区。",
                        split=split,
                        row_ids=[previous[1], row_id],
                    )
    excluded = temporal["excluded_rows"] if temporal is not None else []
    excluded_ids = {row["row_id"] for row in excluded}
    if (
        len(excluded_ids) != len(excluded)
        or found & excluded_ids
        or found | excluded_ids != set(expected)
    ):
        raise ValueError("三个分区与排除台账必须互斥且完整保留全量资料的所有行。")
    if temporal is not None:
        preview_rows = session.full_data.preview.rows
        for split, records in splits.items():
            expected_ids = {preview_rows[index].row_id for index in temporal["assignments"][split]}
            if {row["metadata"]["source_row_id"] for row in records} != expected_ids:
                raise ValueError("物化分区违反已确认时间边界或未来标签窗口规则。")
    virtual_anchor = False
    if metadata.get("evaluation_suite") != artifact.evaluation_suite:
        if metadata.get("evaluation_suite") is None and artifact.evaluation_suite:
            from src.workbench.evaluation_suites import verify_suite

            suite = verify_suite(artifact.evaluation_suite)
            virtual_anchor = all(
                getattr(artifact, key) == value for key, value in suite["anchor_dataset"].items()
            )
        if not virtual_anchor:
            raise ValueError("分区清单与任务绑定的固定评测套件不一致。")
    if artifact.evaluation_suite is not None:
        from src.workbench.evaluation_suites import assert_compatible

        assignment = assert_compatible(session, artifact.evaluation_suite)
        preview_rows = session.full_data.preview.rows
        for split, records in splits.items():
            expected_ids = {
                preview_rows[index].row_id for index in assignment["assignments"][split]
            }
            if {record["metadata"]["source_row_id"] for record in records} != expected_ids:
                raise ValueError("物化分区违反固定评测对象保留规则，不能训练或评测。")
            for record in records if not virtual_anchor else []:
                case_id = assignment["case_ids_by_row"].get(record["metadata"]["source_row_id"])
                if record["metadata"].get("evaluation_case_id") != case_id or record[
                    "metadata"
                ].get("evaluation_scored") != (case_id is not None):
                    raise ValueError("固定评分题与保留但不评分的行标记不一致。")
    return splits


def preflight_dataset(session: IntakeSession, tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Inspect actual causal-LM consumption; this is not business acceptance."""
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length 必须是正整数，并由选定模型与业务长度共同决定。")
    artifact = session.dataset
    report = {
        "status": "passed",
        "dataset_name": artifact.name if artifact else None,
        "version": artifact.version if artifact else None,
        "source_digest": artifact.source_digest if artifact else None,
        "recipe_digest": artifact.recipe_digest if artifact else None,
        "full_confirmed_revision": artifact.full_confirmed_revision if artifact else None,
        "max_length": max_length,
        "tokenizer": {
            key: getattr(tokenizer, key, None)
            for key in (
                "name_or_path",
                "is_fast",
                "pad_token_id",
                "eos_token_id",
                "padding_side",
                "truncation_side",
                "model_max_length",
            )
        },
        "supervision_strategy": "full_prompt_padding_masked",
        "issues": [],
        "splits": {},
        "rows": [],
        "scope_note": "只验证当前不可变数据的分区与 token 消费；不代表答案正确、模型可装入硬件或业务效果达标。输入 token 数包含指令、模板与特殊 token。",
    }
    report["tokenizer"]["class"] = type(tokenizer).__name__
    _issue(
        report,
        "full_prompt_supervision",
        "info",
        "Alpaca 训练对整个 prompt 计算标签，完整记录带 EOS；collator 按 attention_mask 屏蔽 padding，因果损失跳过批次首位置，并非仅监督答案。",
    )
    if not dataset_is_current(session):
        _issue(
            report,
            "stale_dataset",
            "blocking",
            "当前没有与已确认全量方案一致的数据版本，请重新验证并物化。",
        )
        return _finish(report)
    try:
        splits = _verified_partitions(session, report)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        _issue(report, "artifact_mismatch", "blocking", f"数据产物或训练配置校验失败：{exc}")
        return _finish(report)
    if any(issue["severity"] == "blocking" for issue in report["issues"]):
        return _finish(report)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        _issue(
            report,
            "missing_pad_token",
            "blocking",
            "当前 formatter 使用固定长度 padding，但 tokenizer 没有 pad token；请明确配置后重新预检。",
        )
        return _finish(report)
    capacity = getattr(tokenizer, "model_max_length", None)
    if isinstance(capacity, int) and 0 < capacity < 1_000_000_000 and max_length > capacity:
        _issue(
            report,
            "tokenizer_context_limit",
            "warning",
            "max_length 超过 tokenizer 声明的上下文长度；请结合选定模型的实际配置核对。",
        )
    from src.data.loaders import render_alpaca_prompt, tokenize_alpaca_record
    from src.data.sft_collator import AttentionMaskCausalCollator

    collator = AttentionMaskCausalCollator(tokenizer)
    offsets_available = True
    for split, rows in splits.items():
        split_report = {
            "rows": len(rows),
            "truncated_rows": 0,
            "answer_lost_rows": 0,
            "empty_supervision_rows": 0,
            "unverified_answer_rows": 0,
        }
        report["splits"][split] = split_report
        for row in rows:
            row_id = row["metadata"]["source_row_id"]
            answer_start = len(render_alpaca_prompt({**row, "output": ""}))
            try:
                try:
                    full = tokenize_alpaca_record(
                        row, tokenizer, return_offsets_mapping=offsets_available
                    )
                    actual = tokenize_alpaca_record(
                        row, tokenizer, max_length, return_offsets_mapping=offsets_available
                    )
                except (NotImplementedError, TypeError):
                    offsets_available = False
                    full = tokenize_alpaca_record(row, tokenizer)
                    actual = tokenize_alpaca_record(row, tokenizer, max_length)
                full_ids, actual_ids = list(full["input_ids"]), list(actual["input_ids"])
                attention = list(actual["attention_mask"])
                if len(actual_ids) != len(attention) or any(
                    type(token) is not int for token in full_ids + actual_ids
                ):
                    raise ValueError("tokenizer 未返回一维整数 token 及对应 attention_mask。")
                if len(actual_ids) > max_length:
                    raise ValueError(
                        "tokenizer 实际输出超过 max_length，当前特殊 token 或截断设置无法遵守该长度。"
                    )
                if any(mask not in (0, 1) for mask in attention):
                    raise ValueError("tokenizer attention_mask 不是二值掩码。")
                if any(mask == 0 and token != pad_id for token, mask in zip(actual_ids, attention)):
                    _issue(
                        report,
                        "padding_mask_mismatch",
                        "warning",
                        "attention_mask 标为 padding 的 token 与 pad_token_id 不同；collator 仍按 attention_mask 屏蔽这些位置，请核对 tokenizer。",
                        split=split,
                        row_ids=[row_id],
                    )
                batch = collator([{
                    "input_ids": actual_ids,
                    "attention_mask": attention,
                }])
                batch_ids = batch["input_ids"][0].tolist()
                extra_padding = len(batch_ids) - len(actual_ids)
                if "offset_mapping" in actual and extra_padding:
                    padding_offsets = [(0, 0)] * extra_padding
                    actual["offset_mapping"] = (
                        padding_offsets + list(actual["offset_mapping"])
                        if tokenizer.padding_side == "left"
                        else list(actual["offset_mapping"]) + padding_offsets
                    )
                actual_ids = batch_ids
                attention = batch["attention_mask"][0].tolist()
                supervised = [
                    i for i, label in enumerate(batch["labels"][0].tolist())
                    if i > 0 and label != -100
                ]
                active_positions = [i for i, mask in enumerate(attention) if mask]
                verified = "offset_mapping" in full and "offset_mapping" in actual
                full_answer = kept_answer = visible_answer = None
                if verified:
                    if len(full["offset_mapping"]) != len(full_ids) or len(
                        actual["offset_mapping"]
                    ) != len(actual_ids):
                        raise ValueError("tokenizer offset_mapping 与 token 数量不一致。")
                    full_answer = sum(
                        end > answer_start and end > start for start, end in full["offset_mapping"]
                    )
                    kept_answer = sum(
                        actual["offset_mapping"][i][1] > answer_start
                        and actual["offset_mapping"][i][1] > actual["offset_mapping"][i][0]
                        for i in supervised
                    )
                    visible_answer = sum(
                        end > answer_start and end > start and attention[i] == 1
                        for i, (start, end) in enumerate(actual["offset_mapping"])
                    )
                truncated = len(full_ids) > sum(attention)
                detail = {
                    "split": split,
                    "row_id": row_id,
                    "full_tokens": len(full_ids),
                    "input_tokens": len(full_ids) - full_answer if verified else None,
                    "answer_tokens": full_answer,
                    "kept_tokens": sum(attention),
                    "supervised_tokens": len(supervised),
                    "answer_supervised_tokens": kept_answer,
                    "answer_kept_tokens": visible_answer,
                    "was_truncated": truncated,
                    "answer_was_truncated": visible_answer < full_answer if verified else None,
                    "answer_boundary_verified": verified,
                    "terminal_eos_supervised": bool(
                        active_positions
                        and actual_ids[active_positions[-1]] == eos_id
                        and active_positions[-1] in supervised
                    ),
                }
                report["rows"].append(detail)
                if truncated:
                    split_report["truncated_rows"] += 1
                if not supervised:
                    split_report["empty_supervision_rows"] += 1
                    _issue(
                        report,
                        "empty_supervision",
                        "blocking",
                        "因果位移与 padding 屏蔽后没有可计算损失的 token。",
                        split=split,
                        row_ids=[row_id],
                    )
                if verified and (full_answer == 0 or kept_answer == 0):
                    split_report["answer_lost_rows"] += 1
                    _issue(
                        report,
                        "answer_lost",
                        "blocking",
                        "当前 tokenizer、截断或 padding 屏蔽使答案没有任何受监督 token；请查看长度与字段处理。",
                        split=split,
                        row_ids=[row_id],
                    )
                if not verified:
                    split_report["unverified_answer_rows"] += 1
            except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                _issue(
                    report,
                    "tokenizer_error",
                    "blocking",
                    f"无法按训练 formatter 的参数消费本行：{exc}",
                    split=split,
                    row_ids=[row_id],
                )
        split_report["truncation_ratio"] = split_report["truncated_rows"] / len(rows)
        if split_report["truncated_rows"]:
            _issue(
                report,
                "truncated_records",
                "warning",
                "部分记录发生截断，请核对被截掉的上下文或答案；未用固定长度阈值代替业务判断。",
                split=split,
                row_ids=[
                    row["row_id"]
                    for row in report["rows"]
                    if row["split"] == split and row["was_truncated"]
                ],
            )
        if split_report["unverified_answer_rows"]:
            _issue(
                report,
                "answer_offsets_unavailable",
                "warning",
                "tokenizer 未提供字符到 token 的边界，无法可靠证明答案保留；没有把估算 token 差值当成答案覆盖证据。",
                split=split,
            )
    return _finish(report)
