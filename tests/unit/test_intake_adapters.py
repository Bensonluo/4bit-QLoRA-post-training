"""Exercise real isolated enrichment through analysis, full data and lineage."""

import json
import platform

import pytest

from src.workbench.adapters import AdapterRecipe, apply_adapter
from src.workbench.intake_models import IntakeAnalysis
from src.workbench.intake_service import IntakeService, next_action
from src.workbench.sandbox import TransformSandbox
from src.workbench.sources import read_source

RAW = b"raw,label\ncode:0012,A\ncode:0013,B\n"
FULL = b"raw,label\ncode:0021,A\ncode:0022,B\ncode:0023,A\n"
CODE = r"""import re
def transform(rows, config):
    output = []
    for row in rows:
        match = re.fullmatch(r"code:(\d+)", row["raw"])
        if match is None:
            raise ValueError("invalid business code")
        output.append({**row, "number": match.group(1)})
    return output
"""


def adapter(**updates):
    values = {
        "source_code": CODE,
        "new_columns": ["number"],
        "examples": [
            {
                "name": "leading zero business code",
                "kind": "business",
                "rows": [{"__row_id": "r000001", "raw": "code:0012", "label": "A"}],
                "expected_rows": [
                    {"__row_id": "r000001", "raw": "code:0012", "label": "A", "number": "0012"}
                ],
            },
            {
                "name": "malformed code must not be invented",
                "kind": "counterexample",
                "rows": [{"__row_id": "negative", "raw": "unknown", "label": "A"}],
                "expect_error": True,
            },
        ],
    }
    values.update(updates)
    return AdapterRecipe.model_validate(values)


def analysis(adapter_recipe=None):
    return IntakeAnalysis.model_validate(
        {
            "task": {
                "goal": "从业务编号判断人工分类",
                "usage_input": "首次描述中的业务编号",
                "desired_output": "人工分类",
                "row_meaning": "一条独立业务记录",
                "supervision_source": "人工审核分类",
                "success_criteria": ["与人工类别一致"],
                "field_roles": [
                    {"column": "raw", "role": "unused", "reason": "保留原始编码用于核查"},
                    {"column": "label", "role": "target", "reason": "人工分类"},
                    {
                        "column": "number",
                        "role": "input",
                        "reason": "从预测时已有原文解析",
                        "available_at_prediction": True,
                    },
                ],
            },
            "findings": [
                {
                    "kind": "observed",
                    "message": "编码含前导零，需要保留",
                    "evidence_row_ids": ["r000001"],
                }
            ],
            "recipe": {
                "instruction": "判断业务类别",
                "inputs": [{"column": "number", "label": "业务编号"}],
                "targets": [{"column": "label", "label": "类别", "value_kind": "categorical"}],
                "group_columns": ["number"],
            },
            "adapter": (adapter_recipe or adapter()).model_dump(),
            "training_approach": "确认后使用SFT",
            "next_steps": ["核对解析后的真实模型输入"],
        }
    )


@pytest.fixture()
def real_sandbox(monkeypatch):
    import src.workbench.adapters

    sandbox = TransformSandbox(backend="macos" if platform.system() == "Darwin" else "docker")
    if not sandbox.detect():
        pytest.skip(sandbox.unavailable_reason)
    # Select the verified OS backend; actual code execution is never mocked.
    monkeypatch.setattr(src.workbench.adapters, "TransformSandbox", lambda: sandbox)
    return sandbox


def test_regex_adapter_through_full_data_and_immutable_partitions(real_sandbox, tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("按编号判断分类", "sample.csv", RAW)
    raw_digest = session.source.digest
    session = service.apply_analysis(session, analysis())
    assert session.sources["main"].digest == raw_digest
    assert session.sources["main"].rows[0].values == {"raw": "code:0012", "label": "A"}
    assert session.preview.rows[0].input == "业务编号: 0012"
    assert session.adapter_report["validation"]["status"] == "passed"
    assert session.adapter_report["origins"]["r000001"] == [
        {"source_digest": raw_digest, "row_id": "r000001"}
    ]
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    assert next_action(session) == "review_full_data"
    raw_full_digest = session.full_data.sources["main"].digest
    assert session.full_data.source.rows[0].values["number"] == "0021"
    assert session.full_data.adapter_report["validation"]["status"] == "passed"
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id, session.revision, registry_root=tmp_path / "registry"
    )
    assert next_action(session) == "ready_for_training_preflight"
    records = []
    for split in ("train", "validation", "test"):
        from pathlib import Path

        records.extend(
            json.loads(line) for line in Path(session.dataset.paths[split]).read_text().splitlines()
        )
    assert len(records) == 3
    assert {record["input"] for record in records} == {
        "业务编号: 0021",
        "业务编号: 0022",
        "业务编号: 0023",
    }
    assert all(
        record["metadata"]["origins"][0]["source_digest"] == raw_full_digest for record in records
    )
    assert {record["metadata"]["origins"][0]["row_id"] for record in records} == {
        "r000001",
        "r000002",
        "r000003",
    }


@pytest.mark.parametrize(
    "tail,message",
    [
        ("    return output[:1]\n", "一对一"),
        (
            '    if len(output) > 1: output[1]["raw"] = "changed"\n    return output\n',
            "改变了原始字段",
        ),
        (
            '    if len(output) > 1: output[1]["undeclared"] = "x"\n    return output\n',
            "仅增加声明",
        ),
    ],
)
def test_validation_examples_cannot_bypass_actual_row_contract(real_sandbox, tail, message):
    source = read_source("sample.csv", RAW)
    specification = adapter(source_code=CODE.replace("    return output\n", tail))
    with pytest.raises(ValueError, match=message):
        apply_adapter(source, specification)


def test_source_change_never_reuses_old_preview(real_sandbox, tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("按编号判断分类", "sample.csv", RAW)
    session = service.apply_analysis(session, analysis())
    original_spec = session.adapter_report["spec_digest"]
    original_source = session.source.digest
    session = service.confirm(session.session_id, session.revision)
    changed = adapter(source_code=CODE + "\n# revised source version\n")
    session = service.apply_analysis(session, analysis(changed))
    assert session.adapter_report["spec_digest"] != original_spec
    assert session.source.digest != original_source
    assert session.confirmed_revision is None
    assert next_action(session) == "review_preview"
    broken = adapter(source_code=CODE.replace("match.group(1)", "match.group(0)"))
    revision = session.revision
    with pytest.raises(ValueError, match="隔离适配验证未通过"):
        service.apply_analysis(session, analysis(broken))
    assert service.load(session.session_id).revision == revision


def test_business_example_must_reference_actual_source_before_execution():
    source = read_source("sample.csv", RAW)
    spec = adapter()
    spec.examples[0].rows[0]["raw"] = "code:9999"
    with pytest.raises(ValueError, match="真实输入及行 ID"):
        apply_adapter(source, spec)
    with pytest.raises(ValueError, match="独立反例"):
        adapter(examples=[spec.examples[0].model_dump(), spec.examples[0].model_dump()])


def test_sparse_jsonl_preserves_absent_fields_without_inventing_nulls(real_sandbox):
    records = [
        {"raw": "code:0012", "label": "A"},
        {"raw": "code:0013", "label": "B", "optional": None},
    ]
    source = read_source("sparse.jsonl", "\n".join(json.dumps(row) for row in records).encode())
    result = apply_adapter(source, adapter())
    assert "optional" in result.source.columns
    assert "optional" not in result.source.rows[0].values
    assert result.source.rows[1].values["optional"] is None
    assert [row.values["number"] for row in result.source.rows] == ["0012", "0013"]
    added_null = adapter(
        source_code=CODE.replace(
            "    return output\n",
            '    if len(output) > 1: output[0]["optional"] = None\n    return output\n',
        )
    )
    with pytest.raises(ValueError, match="仅增加声明"):
        apply_adapter(source, added_null)


def test_missing_backend_cannot_fallback_to_host_execution(tmp_path, monkeypatch):
    source = read_source("sample.csv", RAW)
    marker = tmp_path / "must-not-be-created"
    spec = adapter(
        source_code=CODE.replace(
            "    output = []", f'    open({str(marker)!r}, "w").write("forbidden")\n    output = []'
        )
    )
    runner = TransformSandbox()
    monkeypatch.setattr(runner, "detect", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        apply_adapter(source, spec, sandbox=runner)
    assert not marker.exists()
