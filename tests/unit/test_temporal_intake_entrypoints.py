"""Real time-policy intake keeps pending labels visible and materializes explicit exclusions."""

import json
from pathlib import Path

import pytest

from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService, next_action
from tests.unit.test_full_data_cli import invoke

POLICY = {
    "available_at_column": "available",
    "prediction_at_column": "predict",
    "label_end_at_column": "end",
    "validation_start": "2024-02-01T00:00:00Z",
    "test_start": "2024-03-01T00:00:00Z",
    "observation_end": "2024-04-01T00:00:00Z",
}
DATA = (
    "event,text,answer,available,predict,end\n"
    "t,训练信息,上涨,2024-01-09T00:00:00Z,2024-01-10T00:00:00Z,2024-01-20T00:00:00Z\n"
    "x,跨窗口信息,未上涨,2024-01-24T00:00:00Z,2024-01-25T00:00:00Z,2024-02-02T00:00:00Z\n"
    "v,验证信息,上涨,2024-02-04T00:00:00Z,2024-02-05T00:00:00Z,2024-02-10T00:00:00Z\n"
    "h,测试信息,未上涨,2024-03-04T00:00:00Z,2024-03-05T00:00:00Z,2024-03-10T00:00:00Z\n"
    "p,未成熟信息,,2024-03-24T00:00:00Z,2024-03-25T00:00:00Z,2024-04-10T00:00:00Z\n"
).encode()


def temporal_session(service, *, data=DATA, confirmed=True):
    session = service.create("预测事件后的方向", "events.csv", data, scope="full")
    proposal = IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": session.goal,
                "usage_input": "事件已知文本",
                "desired_output": "未来方向",
                "row_meaning": "一个事件",
                "supervision_source": "已观察的后续结果",
                "success_criteria": ["方向正确"],
                "field_roles": [
                    {
                        "column": name,
                        "role": "group"
                        if name == "event"
                        else "input"
                        if name == "text"
                        else "target"
                        if name == "answer"
                        else "metadata",
                        "reason": "明确字段用途",
                        **({"available_at_prediction": True} if name == "text" else {}),
                    }
                    for name in session.source.columns
                ],
            },
            "recipe": {
                "instruction": "根据已知事件文本预测方向",
                "inputs": [{"column": "text", "label": "文本"}],
                "targets": [{"column": "answer", "label": "方向", "value_kind": "categorical"}],
                "group_columns": ["event"],
                "temporal_split": POLICY,
            },
            "findings": [],
            "training_approach": "确认后SFT",
            "next_steps": ["核对观察窗口和排除记录"],
        }
    )
    session = service.apply_analysis(session, proposal)
    if confirmed:
        session = service.confirm(session.session_id, session.revision)
        session = service.validate_full_data(session.session_id, session.revision)
        session = service.confirm_full_data(session.session_id, session.revision)
    return session


def test_temporal_cli_materialize_reports_retained_exclusions_and_ignores_random_ratios(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = temporal_session(service)
    assert all(item.expected_target is not None for item in session.confirmed_examples)
    assert all(item.expected_target is not None for item in session.full_data.confirmed_examples)
    result = invoke(
        service,
        "materialize",
        session.session_id,
        "--revision",
        session.revision,
        "--validation-fraction",
        0.4,
        "--test-fraction",
        0.4,
        "--seed",
        999,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    stats = payload["dataset"]["statistics"]
    assert stats["split_method"] == "temporal"
    assert stats["row_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert stats["included_rows"] == 3 and stats["excluded_rows"] == 2
    assert "随机比例与种子不生效" in result.stderr
    # 摘要句式:纳入/排除走 summarize_dataset 的统一词汇(共纳入 N 条/另有 N 条),
    # 而非 CLI 自造句;逐原因计数与 manifest 指引由 CLI 补充行单独承载。
    assert "共纳入 3 条（全量 5 条）" in result.stderr
    assert "另有 2 条" in result.stderr
    assert "以已确认的时间方案为准" in result.stderr
    assert "排除原因计数" in result.stderr
    assert "原行明细见 dataset.paths.manifest" in result.stderr
    manifest = json.loads(Path(payload["dataset"]["paths"]["manifest"]).read_text())
    excluded = manifest["metadata"]["excluded_rows"]
    assert {item["reason"] for item in excluded} == {
        "label_not_mature",
        "label_window_crosses_validation_start",
    }
    pending = next(item for item in excluded if item["reason"] == "label_not_mature")
    assert pending["target"] is None and pending["original"]["event"] == "p"


def test_only_pending_or_invalid_time_cannot_be_confirmed_as_real_supervision(tmp_path):
    service = IntakeService(tmp_path / "intake")
    header, *rows = DATA.decode().splitlines()
    pending_only = (header + "\n" + rows[-1] + "\n").encode()
    session = temporal_session(service, data=pending_only, confirmed=False)
    assert next_action(session) == "needs_labels"
    with pytest.raises(ValueError, match="不能确认"):
        service.confirm(session.session_id, session.revision)
    session = temporal_session(service, confirmed=False)
    with pytest.raises(ValueError, match="至少核对"):
        service.confirm(session.session_id, session.revision, ["r000005"])
    future_input = DATA.replace(b"2024-01-09T00:00:00Z", b"2024-01-11T00:00:00Z")
    invalid = temporal_session(service, data=future_input, confirmed=False)
    assert next_action(invalid) == "needs_data_revision"
