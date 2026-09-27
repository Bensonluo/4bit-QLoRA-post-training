"""CLI keeps named originals through actual multi-table full-data validation."""

import json

import pytest

from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService
from tests.unit.test_full_data_cli import invoke

MAIN = "ticket,text\n1,杯子破了\n2,物流慢\n3,配件损坏\n".encode()
LABELS = "id,category\n1,质量\n2,物流\n3,质量\n".encode()
FULL_MAIN = "ticket,text\n10,新杯子损坏\n11,新物流延迟\n12,新配件缺陷\n".encode()
FULL_LABELS = "id,category\n10,质量\n11,物流\n12,质量\n".encode()


def combined_plan():
    return IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": "判断工单类别",
                "usage_input": "客户描述",
                "desired_output": "人工审核类别",
                "row_meaning": "一个工单",
                "supervision_source": "labels中人工审核类别",
                "success_criteria": ["类别正确"],
                "field_roles": [
                    {"column": "ticket", "role": "group", "reason": "工单编号"},
                    {
                        "column": "text",
                        "role": "input",
                        "reason": "首次咨询已知",
                        "available_at_prediction": True,
                    },
                    {"column": "label_id", "role": "unused", "reason": "关联核对"},
                    {"column": "label_category", "role": "target", "reason": "人工审核答案"},
                ],
            },
            "composition": {
                "base_source": "main",
                "steps": [
                    {
                        "operation": "join",
                        "right_source": "labels",
                        "left_on": ["ticket"],
                        "right_on": ["id"],
                        "how": "left",
                        "cardinality": "one_to_one",
                        "prefix": "label_",
                    }
                ],
            },
            "recipe": {
                "instruction": "判断工单类别",
                "inputs": [{"column": "text", "label": "描述"}],
                "targets": [
                    {"column": "label_category", "label": "类别", "value_kind": "categorical"}
                ],
                "group_columns": ["ticket"],
            },
            "findings": [],
            "training_approach": "SFT",
            "next_steps": ["核对类别映射"],
        }
    )


def composition_session(service, *, scope="sample"):
    session = service.create("判断工单类别", "tickets.csv", MAIN, scope=scope)
    session = service.add_source(
        session.session_id, session.revision, "labels", "labels.csv", LABELS, scope=scope
    )
    session = service.apply_analysis(session, combined_plan())
    return service.confirm(session.session_id, session.revision)


def test_cli_add_source_then_validate_actual_full_tables(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("判断工单类别", "tickets.csv", MAIN)
    labels = tmp_path / "labels.csv"
    labels.write_bytes(LABELS)
    result = invoke(
        service,
        "add-source",
        session.session_id,
        "--revision",
        session.revision,
        "--alias",
        "labels",
        "--input",
        labels,
        "--description",
        "人工审核类别，id 对应工单编号",
    )
    assert result.returncode == 0, result.stderr
    session = service.load(session.session_id)
    assert set(session.sources) == {"main", "labels"}
    assert "人工审核类别" in session.answers[-1]["answer"]
    session = service.apply_analysis(session, combined_plan())
    session = service.confirm(session.session_id, session.revision)
    main_path, labels_path = tmp_path / "full-main.csv", tmp_path / "full-labels.csv"
    main_path.write_bytes(FULL_MAIN)
    labels_path.write_bytes(FULL_LABELS)
    result = invoke(
        service,
        "full-sources",
        session.session_id,
        "--revision",
        session.revision,
        "--source",
        f"main={main_path}",
        "--source",
        f"labels={labels_path}",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["full_data"]["preview"]["counts"]["ready"] == 3
    assert set(payload["full_data"]["sources"]) == {"main", "labels"}
    assert payload["sources"]["main"]["scope"] == "sample"
    assert "review_full_data" in result.stderr
    assert payload["full_data"]["preview"]["rows"][0]["target"] == "质量"


def test_cli_does_not_promote_sample_to_full(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    result = invoke(service, "full-sources", session.session_id, "--revision", session.revision)
    assert result.returncode == 2
    assert "不能将样例自动当作全量" in result.stderr


@pytest.mark.parametrize("source_args", [["broken"], ["main=FILE", "main=FILE"], ["main=FILE"]])
def test_cli_invalid_full_source_arguments_leave_session_unchanged(tmp_path, source_args):
    service = IntakeService(tmp_path / "intake")
    session = composition_session(service)
    path = tmp_path / "main.csv"
    path.write_bytes(FULL_MAIN)
    args = [
        value for item in source_args for value in ("--source", item.replace("FILE", str(path)))
    ]
    result = invoke(
        service, "full-sources", session.session_id, "--revision", session.revision, *args
    )
    assert result.returncode == 2
    assert service.load(session.session_id).revision == session.revision
    assert "Traceback" not in result.stderr
