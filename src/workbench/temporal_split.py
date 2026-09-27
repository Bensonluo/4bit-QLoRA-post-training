"""Explicit point-in-time partitions with retained exclusions, never random fallback."""

from __future__ import annotations

from datetime import datetime, timezone

from src.workbench.intake_models import PreviewRow, TemporalSplitPolicy


def parse_timestamp(value, *, label="时间") -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}缺失或不是带时区ISO时间。")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}不是有效ISO时间：{value}") from exc
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(f"{label}必须明确时区。")
    return instant.astimezone(timezone.utc)


def temporal_row_times(row: PreviewRow, policy: TemporalSplitPolicy) -> dict[str, datetime]:
    times = {
        name: parse_timestamp(row.original.get(column), label=f"第{row.row_id}行{column}")
        for name, column in (
            ("available_at", policy.available_at_column),
            ("prediction_at", policy.prediction_at_column),
            ("label_end_at", policy.label_end_at_column),
        )
    }
    if not times["available_at"] <= times["prediction_at"] < times["label_end_at"]:
        raise ValueError(
            f"第{row.row_id}行需满足available_at <= prediction_at < label_end_at；不能使用预测之后才可获得的信息。"
        )
    return times


def temporal_assignment(
    rows: list[PreviewRow], policy: TemporalSplitPolicy, group_columns: list[str]
) -> dict:
    from src.workbench.materialize import _connected_groups

    policy = TemporalSplitPolicy.model_validate(
        policy.model_dump() if isinstance(policy, TemporalSplitPolicy) else policy
    )
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("时间分区行标识必须唯一。")
    b1, b2, end = [
        parse_timestamp(value)
        for value in (policy.validation_start, policy.test_start, policy.observation_end)
    ]
    times_by_row, destinations, reasons = {}, {}, {}
    for index, row in enumerate(rows):
        times = temporal_row_times(row, policy)
        times_by_row[row.row_id] = {
            key: value.isoformat().replace("+00:00", "Z") for key, value in times.items()
        }
        prediction, label_end = times["prediction_at"], times["label_end_at"]
        immature = label_end > end
        if (
            row.status in {"invalid", "conflict"}
            or (row.status == "needs_label" or row.target is None)
            and not immature
        ):
            raise ValueError(f"第{row.row_id}行存在转换冲突或成熟样本缺标签，不能以时间排除掩盖。")
        if immature:
            destinations[index], reasons[index] = None, "label_not_mature"
        elif prediction < b1:
            destinations[index] = "train" if label_end < b1 else None
            if destinations[index] is None:
                reasons[index] = "label_window_crosses_validation_start"
        elif prediction < b2:
            destinations[index] = "validation" if label_end < b2 else None
            if destinations[index] is None:
                reasons[index] = "label_window_crosses_test_start"
        else:
            destinations[index] = "test"
    assignments = {split: [] for split in ("train", "validation", "test")}
    group_counts = dict.fromkeys(assignments, 0)
    for group in _connected_groups(rows, group_columns):
        included = {destinations[index] for index in group} - {None}
        if len(included) > 1:
            identities = [rows[index].row_id for index in group]
            raise ValueError(
                f"同一事件组或相同模型输入跨时间分区：{identities}；不能拆分同一业务对象。"
            )
        if any(destinations[index] is None for index in group):
            for index in group:
                if destinations[index] is not None:
                    destinations[index] = None
                    reasons[index] = "connected_to_excluded_row"
            continue
        split = next(iter(included))
        assignments[split].extend(group)
        group_counts[split] += 1
    return {
        "assignments": {split: sorted(indices) for split, indices in assignments.items()},
        "group_counts": group_counts,
        "excluded_rows": [
            {
                "row_id": row.row_id,
                "reason": reasons[index],
                "times": times_by_row[row.row_id],
                "original": row.original,
                "input": row.input,
                "target": row.target,
                "status": row.status,
                "group": row.group,
            }
            for index, row in enumerate(rows)
            if destinations[index] is None
        ],
        "times_by_row": times_by_row,
    }


def is_pending_label(row: PreviewRow, policy: TemporalSplitPolicy | None) -> bool:
    """Only valid, unmatured label windows may retain a genuinely missing target."""
    if policy is None or row.status != "needs_label":
        return False
    try:
        return temporal_row_times(row, policy)["label_end_at"] > parse_timestamp(
            policy.observation_end
        )
    except ValueError:
        return False
