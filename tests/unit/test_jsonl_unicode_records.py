"""JSONL splits on physical line breaks, not Unicode characters inside values."""

import json

import pytest

from src.workbench.evaluation_suites import evaluation_cases
from src.workbench.intake_service import IntakeService
from src.workbench.sources import read_source
from src.workbench.training_preflight import _verified_partitions
from tests.unit.test_full_data import approved


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_unicode_separator_inside_json_string_is_not_a_record_boundary(separator):
    rows = [
        {"编号": "101", "客户描述": f"物流{separator}延迟", "类别": "物流", "处理结果": "查询"},
        {"编号": "102", "客户描述": "杯子破损", "类别": "质量", "处理结果": "补发"},
        {"编号": "103", "客户描述": "其他破损", "类别": "质量", "处理结果": "补发"},
    ]
    encoded = [json.dumps(row, ensure_ascii=False) for row in rows]
    data = (encoded[0] + "\r\n\r" + encoded[1] + "\n" + encoded[2]).encode()
    source = read_source("full.jsonl", data, scope="full")
    assert [row.values for row in source.rows] == rows
    assert [row.line for row in source.rows] == [1, 3, 4]
    assert "第 2 行" in source.blank_note


def test_unicode_separator_survives_real_upload_and_dataset_materialization(tmp_path, monkeypatch):
    service = IntakeService(tmp_path / "intake")
    session = approved(service)
    rows = [
        {"编号": "101", "客户描述": "物流\u2028延迟", "类别": "物流", "处理结果": "查询"},
        {"编号": "102", "客户描述": "杯子\u2029破损", "类别": "质量", "处理结果": "补发"},
        {"编号": "103", "客户描述": "包装\u0085破损", "类别": "质量", "处理结果": "补发"},
    ]
    data = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode()
    session = service.validate_full_data(session.session_id, session.revision, "full.jsonl", data)
    assert session.full_data.status == "review"
    assert [row.original for row in session.full_data.preview.rows] == rows
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    assert (
        service.root / session.session_id / "full" / f"{session.full_data.source.digest}.jsonl"
    ).read_bytes() == data
    check = {"issues": []}
    partitions = _verified_partitions(session, check)
    assert not check["issues"]
    outputs = [row for split in ("train", "validation", "test") for row in partitions[split]]
    assert len(outputs) == len(rows)
    for row in rows:
        assert sum(row["客户描述"] in output["input"] for output in outputs) == 1
    assert evaluation_cases(session)["records"] == partitions["validation"]

    # Exercise the same Hugging Face loader used by training, with its cache
    # confined to the temporary test directory and no remote downloads.
    from src.data import loaders

    real_load = loaders.load_dataset
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    def local_load(*args, **kwargs):
        return real_load(*args, cache_dir=tmp_path / "hf-cache", **kwargs)

    monkeypatch.setattr(loaders, "load_dataset", local_load)
    for split in ("train", "validation"):
        dataset = loaders.AlpacaDataset(session.dataset.paths[split]).load()
        assert [row["input"] for row in dataset] == [row["input"] for row in partitions[split]]

    from src.workbench.business_evaluation import (
        BusinessEvaluationService,
        EvaluationModel,
        EvaluationProtocol,
        Generation,
    )

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "pytorch_model.bin").write_bytes(b"fixture model")
    prompts = []

    class Runtime:
        def __init__(self, model):
            pass

        def generate(self, prompt, protocol):
            prompts.append(prompt)
            return Generation("质量")

        def close(self):
            pass

    report = BusinessEvaluationService(tmp_path / "evaluations").compare(
        session,
        [EvaluationModel("基座", str(base)), EvaluationModel("候选", str(base))],
        EvaluationProtocol("classification_exact"),
        runtime_factory=Runtime,
    )
    assert len(prompts) == 2
    assert prompts[0] == prompts[1]
    assert partitions["validation"][0]["input"] in prompts[0]
    assert all(model["rows"][0]["prompt"] == prompts[0] for model in report.models)


def test_invalid_json_after_unicode_separator_reports_physical_line():
    good = json.dumps({"text": "甲\u2028乙"}, ensure_ascii=False)
    with pytest.raises(ValueError, match="JSONL 第 3 行无效"):
        read_source("bad.jsonl", (good + "\r\n\r\n" + '{"text": }').encode())
