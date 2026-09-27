"""Materialize business-confirmed full data as immutable, isolated partitions."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.workbench.full_data import full_data_is_current
from src.workbench.intake_models import DatasetArtifact, IntakeSession, PreviewRow
from src.workbench.recipes import preview_recipe
from src.workbench.sources import canonical, is_missing

SPLITS = ("train", "validation", "test")


def dataset_is_current(session: IntakeSession) -> bool:
    artifact, report = session.dataset, session.full_data
    return bool(
        artifact
        and full_data_is_current(session)
        and report
        and report.status == "confirmed"
        and report.confirmed_revision is not None
        and artifact.full_confirmed_revision == report.confirmed_revision
        and artifact.source_digest == report.source.digest
        and artifact.recipe_digest == report.approved_recipe_digest
    )


def _connected_groups(rows: list[PreviewRow], columns: list[str]) -> list[list[int]]:
    """Any shared entity field or rendered input connects rows transitively."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = [("input", row.input)]
        for column in columns:
            value = row.group.get(column)
            if is_missing(value):
                raise ValueError(f"全量第 {row.row_id} 行缺少分组字段 {column}，请修正后重新确认。")
            keys.append((f"group:{column}", canonical(value)))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


_SPLIT_LABELS = {"validation": "验证", "test": "测试"}


def _answer_coverage_note(train_missing: dict[str, dict[str, int]], cause: str) -> str:
    """把「训练集没见过的答案」翻成人话：点名值、条数与落点，分述行为，不自动重切。"""
    parts = []
    for value, where in list(train_missing.items())[:5]:
        locations = "、".join(
            f"{_SPLIT_LABELS[split]}{where[split]} 条"
            for split in ("validation", "test")
            if split in where
        )
        parts.append(f"{value}×{sum(where.values())}（{locations}）")
    listing = "、".join(parts) + ("等" if len(train_missing) > 5 else "")
    return (
        f"验证/测试集中有 {len(train_missing)} 类答案（{listing}）从未出现在训练集——"
        "训练按逐字学习答案，模型没有学过这些值，验证与测试仍会照常打分。"
        f"{cause}"
        "补充该类别的独立业务对象后可重新生成分区版本；没有自动重新切分，也不会把记录挪回训练集。"
    )


def _duplicate_note(rendered_extra: int, source_extra: int, total_rows: int) -> str:
    """把「完全相同例题」翻成人话：分述两种成因，点名加权效应，不自动去重。"""
    if source_extra and rendered_extra > source_extra:
        cause = (
            f"其中 {source_extra} 条原始行完全重复（每个字段都一致，多见于导出拼接或关联重复）；"
            f"另有 {rendered_extra - source_extra} 条是不同原始行渲染成同一例题"
            "（原始字段不同、例题相同）。"
        )
    elif source_extra:
        cause = "均为原始行完全重复（每个字段都一致，多见于导出拼接或关联重复）。"
    else:
        cause = "均为不同原始行渲染成同一例题（原始字段不同、例题相同）。"
    return (
        f"本版本有 {rendered_extra} 条记录与前面的记录渲染后完全相同（输入与答案逐字一致）——"
        f"同一道例题会出现多次，训练等效于给这些例题加权；{total_rows} 条记录去重后只有 "
        f"{total_rows - rendered_extra} 道独立例题。{cause}"
        "相同输入配不同答案已被全量验证拦下，不会出现在任何分区；"
        "完全相同的记录保持在同一分区。全部记录原样保留；没有自动去重，去留由你决定。"
    )


def materialize_dataset(
    session: IntakeSession,
    *,
    registry_root: str | Path,
    name: str,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    independent_rows_confirmed: bool = False,
    evaluation_suite: dict | None = None,
) -> DatasetArtifact:
    """Preserve every confirmed row; split connected entities before registration."""
    recipe = session.analysis.recipe if session.analysis else None
    policy = recipe.temporal_split if recipe else None
    fractions = (validation_fraction, test_fraction)
    if not policy and (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in fractions
        )
        or sum(fractions) >= 1
    ):
        raise ValueError("验证集和测试集比例必须是有限正数，且两者之和必须小于 1。")
    if not policy and type(seed) is not int:
        raise ValueError("切分随机种子必须是整数。")
    if type(independent_rows_confirmed) is not bool:
        raise ValueError("独立记录确认必须明确为是或否。")
    report = session.full_data
    if (
        not full_data_is_current(session)
        or report is None
        or report.status != "confirmed"
        or report.confirmed_revision is None
        or report.preview is None
        or recipe is None
        or any(issue.severity == "blocking" for issue in report.issues)
    ):
        raise ValueError("请先解决全量问题并确认当前全量报告，才能生成数据分区。")
    if report.preview != preview_recipe(report.source, recipe):
        raise ValueError("全量转换预览与当前来源或方案不一致，请重新验证并确认。")
    rows = report.preview.rows
    from src.workbench.temporal_split import is_pending_label, temporal_assignment

    if not rows or any(
        (row.status != "ready" or row.target is None) and not is_pending_label(row, policy)
        for row in rows
    ):
        raise ValueError("全量仍有缺标签或转换问题，不能删除问题行后继续物化。")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("全量行标识不唯一，请重新验证来源。")
    if not recipe.group_columns and not independent_rows_confirmed:
        raise ValueError("当前方案没有分组字段，请明确确认每行代表独立业务对象，或补充分组字段。")
    groups = _connected_groups(rows, recipe.group_columns)
    if len(groups) < 3:
        raise ValueError(
            f"仅有 {len(groups)} 个独立分组，至少需要 3 个才能形成非空训练、验证和测试集；"
            "请补充独立业务对象，不能拆开同一对象或相同模型输入来凑数。"
        )

    ratios = (
        None
        if policy
        else {
            "train": 1 - validation_fraction - test_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        }
    )
    suite_assignment = None
    temporal = None
    if evaluation_suite is not None:
        from src.workbench.evaluation_suites import assert_compatible

        suite_assignment = assert_compatible(session, evaluation_suite)
        assigned = suite_assignment["assignments"]
        group_counts = suite_assignment["group_counts"]
        if policy:
            temporal = suite_assignment
    elif policy:
        temporal = temporal_assignment(rows, policy, recipe.group_columns)
        assigned, group_counts = temporal["assignments"], temporal["group_counts"]
    else:
        # Preserve complete connected components even when target ratios cannot be attained.
        random.Random(seed).shuffle(groups)
        groups.sort(key=len, reverse=True)
        assigned: dict[str, list[int]] = {split: [] for split in SPLITS}
        group_counts = dict.fromkeys(SPLITS, 0)
        for position, group in enumerate(groups):
            empty = [split for split in SPLITS if not assigned[split]]
            candidates = empty if len(groups) - position == len(empty) else list(SPLITS)
            split = max(candidates, key=lambda key: ratios[key] * len(rows) - len(assigned[key]))
            assigned[split].extend(group)
            group_counts[split] += 1

    if any(not assigned[split] for split in SPLITS):
        raise ValueError(
            "时间或固定评测规则未形成三个非空分区；请补充符合窗口的资料，不会随机补数。"
        )

    records: dict[str, list[dict[str, Any]]] = {}
    for split, indices in assigned.items():
        records[split] = [
            {
                "instruction": recipe.instruction,
                "input": rows[index].input,
                "output": rows[index].target,
                "metadata": {
                    "source_digest": report.source.digest,
                    "source_row_id": rows[index].row_id,
                    "group": rows[index].group,
                    "origins": (
                        report.composition_report["origins"][rows[index].row_id]
                        if report.composition_report
                        else report.adapter_report["origins"][rows[index].row_id]
                        if report.adapter_report
                        else [{"source_digest": report.source.digest, "row_id": rows[index].row_id}]
                    ),
                },
            }
            for index in sorted(indices)
        ]
        if suite_assignment:
            for record in records[split]:
                case_id = suite_assignment["case_ids_by_row"].get(
                    record["metadata"]["source_row_id"]
                )
                record["metadata"].update(
                    evaluation_case_id=case_id, evaluation_scored=case_id is not None
                )
        if temporal:
            for record in records[split]:
                record["metadata"]["temporal"] = temporal["times_by_row"][
                    record["metadata"]["source_row_id"]
                ]
    answer_counts = {
        split: dict(Counter(record["output"] for record in part)) for split, part in records.items()
    }
    distinct_answers = set().union(*(set(counts) for counts in answer_counts.values()))
    train_missing: dict[str, dict[str, int]] = {}
    if len(distinct_answers) <= 20:
        train_values = set(answer_counts["train"])
        for value in sorted(distinct_answers - train_values):
            train_missing[value] = {
                split: answer_counts[split][value]
                for split in ("validation", "test")
                if value in answer_counts[split]
            }
    rendered_counts = Counter((row.input, row.target) for row in rows)
    raw_counts = Counter(canonical(row.original) for row in rows)
    statistics = {
        "total_rows": len(rows),
        "independent_groups": len(groups),
        "requested_ratios": None if suite_assignment else ratios,
        "actual_ratios": {split: len(part) / len(rows) for split, part in records.items()},
        "row_counts": {split: len(part) for split, part in records.items()},
        "group_counts": group_counts,
        "rendered_exact_duplicate_rows": sum(count - 1 for count in rendered_counts.values()),
        "source_exact_duplicate_rows": sum(count - 1 for count in raw_counts.values()),
        "stratified": False,
        "note": "保留所有记录；按业务对象及相同模型输入隔离，实际比例受分组大小影响；未做类别分层。",
    }
    if suite_assignment:
        statistics.update(
            split_method="fixed_evaluation_suite",
            evaluation_suite_id=suite_assignment["suite_id"],
            reserved_rows=suite_assignment["reserved_rows"],
            fixed_case_counts=evaluation_suite["case_counts"],
            note="固定开发与最终测试评分题；新增独立资料进入训练，同评测对象新增行仍保留在对应分区但不扩充评分题。比例及随机种子不重新切分固定题。",
        )
    if temporal:
        statistics.update(
            split_method="temporal_fixed_evaluation_suite" if suite_assignment else "temporal",
            included_rows=sum(len(part) for part in records.values()),
            excluded_rows=len(temporal["excluded_rows"]),
            exclusion_counts=dict(Counter(row["reason"] for row in temporal["excluded_rows"])),
            note="按已确认时间边界分区；跨窗口、未成熟及关联排除记录完整保留；不随机补数，不把未成熟标签当真值。",
        )
    # 答案覆盖披露（分组随机/时间/固定题集同口径）：逐字学习下训练集没见过的值不可学。
    # 仅在全部答案的不同取值 ≤ 20 时计算——开放文本/大量类别时逐值点名没有信息量，缺键即这一如实边界。
    if len(distinct_answers) <= 20:
        statistics["answer_counts_by_split"] = answer_counts
        statistics["train_missing_answers"] = train_missing
        if train_missing:
            method = statistics.get("split_method", "")
            if method.startswith("temporal"):
                cause = "时间边界先于比例，窗口内只出现一次的类别会整体落在单一分区。"
            elif method == "fixed_evaluation_suite":
                cause = "固定题集把既定评分题保留在原分区，训练段新增类别可能只出现在验证或测试。"
            else:
                cause = (
                    "分组隔离优先于比例且未做类别分层，稀有类别的整组记录可能全部落在验证或测试。"
                )
            statistics["answer_coverage_note"] = _answer_coverage_note(train_missing, cause)
    # 完全相同例题披露：重复例题=隐式加权；相同输入配不同答案已被 conflict 守卫硬拦。
    rendered_extra = statistics["rendered_exact_duplicate_rows"]
    if rendered_extra:
        statistics["duplicate_note"] = _duplicate_note(
            rendered_extra, statistics["source_exact_duplicate_rows"], len(rows)
        )
    metadata = {
        "operation": "confirmed_intake_alpaca_split_v1",
        "source_digest": report.source.digest,
        "sample_source_digest": session.source.digest,
        "recipe_digest": report.approved_recipe_digest,
        "composition": session.analysis.composition,
        "adapter": session.analysis.adapter,
        "source_files": {name: source.digest for name, source in report.sources.items()},
        "group_columns": recipe.group_columns,
        "independent_rows_confirmed": independent_rows_confirmed,
        "seed": None if suite_assignment or policy else seed,
        "statistics": statistics,
    }
    if evaluation_suite is not None:
        metadata["evaluation_suite"] = evaluation_suite
    if temporal:
        metadata["temporal_policy"] = policy.model_dump()
        metadata["excluded_rows"] = temporal["excluded_rows"]
    # Lazy import keeps basic intake/Agent configuration independent of training libraries.
    from src.data_flywheel.dataset_registry import LocalDatasetRegistry

    root = Path(registry_root).resolve()
    registry = LocalDatasetRegistry(root)
    version = registry.register_splits(name, records, metadata=metadata)
    manifest = registry.get_split_manifest(name, version)
    directory = root / name / "splits" / version
    paths = {split: str(directory / manifest["splits"][split]["path"]) for split in SPLITS}
    paths["manifest"] = str(directory / "manifest.json")
    return DatasetArtifact(
        name=name,
        version=version,
        registry_root=str(root),
        source_digest=report.source.digest,
        recipe_digest=report.approved_recipe_digest,
        full_confirmed_revision=report.confirmed_revision,
        paths=paths,
        data_config={
            "dataset_name": paths["train"],
            "train_file": paths["train"],
            "validation_file": paths["validation"],
            "validation_split": 0,
            "max_samples": None,
            "format": "alpaca",
            "dataset_loader": "alpaca",
        },
        statistics=statistics,
        evaluation_suite=evaluation_suite,
    )
