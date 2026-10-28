"""Public display data stays faithful to existing evidence without execution."""

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.workbench.demo_task import demo_create_payload
from ui.showcase import (
    R148_SOURCE,
    R149_SOURCE,
    example_tasks,
    example_tasks_tolerant,
    history_catalog,
    load_showcase,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def assets(tmp_path):
    for name in (R149_SOURCE, R148_SOURCE):
        target = tmp_path / "docs/validation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "docs/validation" / name, target)
    target = tmp_path / "data/custom/examples"
    target.mkdir(parents=True)
    for variant in (None, "generation", "extraction"):
        _, content, name, _ = demo_create_payload(ROOT, variant)
        (target / name).write_bytes(content)
    return tmp_path


def source(root, name=R149_SOURCE):
    path = root / "docs/validation" / name
    return path, json.loads(path.read_bytes())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False))


def test_r149_preserves_all_raw_outputs_proofs_history_and_failure_status():
    path, original = source(ROOT)
    data = load_showcase(ROOT)
    assert data["report"] == original["actual_report"]
    assert data["history"] == original["actual_lr_and_optimizer_steps"]
    assert len(data["history"]) == data["training"]["global_step"] == 64
    assert data["report"]["status"] == "completed_with_failures"
    assert sum(len(model["rows"]) for model in data["report"]["models"]) == 32
    assert [model["metrics"]["exact_match"] for model in data["report"]["models"]] == [0, 1, 1, 1]
    assert all(row["truncated"] for row in data["report"]["models"][0]["rows"])
    assert all(row["token_proof"] for model in data["report"]["models"] for row in model["rows"])
    assert data["source"] == {
        "name": R149_SOURCE,
        "path": f"docs/validation/{R149_SOURCE}",
        "date": original["date"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert data["costs"] == original["costs"]
    assert data["costs"]["electricity_kwh"] is None
    assert data["limitations"] == original["limitations"]


def test_presentation_metadata_omits_server_paths_and_final_test_payloads():
    data = load_showcase(ROOT)
    for group in data["config"].values():
        assert (
            not {
                "output_dir",
                "train_file",
                "validation_file",
                "dataset_name",
                "mlflow_tracking_uri",
                "log_dir",
            }
            & group.keys()
        )
    assert "/" not in data["config"]["model"]["name"]
    assert all("directory" not in item for item in data["artifacts"].values())
    assert "preflight" not in data and "preparation" not in data
    assert "test_seal_from_manifest_unchanged" not in data
    assert "/Users/" not in json.dumps(data, ensure_ascii=False)
    for model in data["report"]["models"]:
        assert model["rows"]
        assert data["report"]["dataset"]["split"] == "validation"


def test_artifact_sizes_and_hashes_come_from_bound_export_receipts():
    _, original = source(ROOT)
    data = load_showcase(ROOT)
    for name, receipt_name in (("hf", "hf"), ("gguf_q4_k_m", "gguf")):
        item = data["artifacts"][name]
        for filename, record in item["weight_files"].items():
            assert record == original["export_receipts"][receipt_name]["output_files"][filename]
        assert item["source_run_id"] == data["training"]["run_id"]


def test_new_load_does_not_reuse_mutated_display_objects(assets):
    data = load_showcase(assets)
    original = deepcopy(data)
    data["report"]["models"][0]["rows"][0]["output"] = "edited by caller"
    data["history"][0]["loss"] = 999
    data["artifacts"]["hf"]["weight_files"].clear()
    assert load_showcase(assets) == original


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_model",
        "missing_question",
        "duplicate_question",
        "invented_score",
        "repaired_output",
        "missing_proof",
        "different_prompt_ids",
        "different_eos",
        "bad_stop",
        "hide_truncation",
        "short_history",
        "repeated_step",
        "changed_lr",
        "nan_loss",
        "changed_receipt_loss",
        "changed_verification_id",
        "changed_artifact_hash",
        "inherited_export_score",
    ],
)
def test_incomplete_or_contradictory_evidence_fails_instead_of_filling_values(assets, mutation):
    path, data = source(assets)
    report, logs = data["actual_report"], data["actual_lr_and_optimizer_steps"]
    row = report["models"][1]["rows"][0]
    if mutation == "missing_model":
        report["models"].pop()
    elif mutation == "missing_question":
        report["models"][1]["rows"].pop()
    elif mutation == "duplicate_question":
        report["models"][1]["rows"][1]["index"] = 0
    elif mutation == "invented_score":
        report["models"][0]["metrics"]["exact_match"] = 1
    elif mutation == "repaired_output":
        report["models"][0]["rows"][0]["output"] = "L8"
    elif mutation == "missing_proof":
        del row["token_proof"]
    elif mutation == "different_prompt_ids":
        row["token_proof"]["prompt_token_ids"][0] += 1
    elif mutation == "different_eos":
        row["token_proof"]["eos_token_ids"] = [0]
    elif mutation == "bad_stop":
        row["token_proof"]["finish_reason"] = "length"
    elif mutation == "hide_truncation":
        report["status"] = "completed"
    elif mutation == "short_history":
        logs.pop()
    elif mutation == "repeated_step":
        logs[1]["step"] = 1
    elif mutation == "changed_lr":
        logs[1]["learning_rate"] *= 10
    elif mutation == "nan_loss":
        logs[1]["loss"] = float("nan")
    elif mutation == "changed_receipt_loss":
        data["training"]["metrics"]["loss"] = 999
    elif mutation == "changed_verification_id":
        data["independent_verification"]["evaluation_id"] = "other-report"
    elif mutation == "changed_artifact_hash":
        data["artifact_inventory"]["artifacts"]["hf"]["weight_files"]["model.safetensors"][
            "sha256"
        ] = "f" * 64
    else:
        data["export_receipts"]["gguf"]["business_evaluation_status"] = "passed"
    save(path, data)
    with pytest.raises(ValueError, match="公开历史记录"):
        load_showcase(assets)


@pytest.mark.parametrize("content", [b"broken-json", b"[]", b"{}", b"\xff"])
def test_unreadable_public_json_reports_a_clear_error(assets, content):
    (assets / "docs/validation" / R149_SOURCE).write_bytes(content)
    with pytest.raises(ValueError, match="公开历史记录"):
        load_showcase(assets)


def test_missing_public_file_is_not_fabricated(tmp_path):
    with pytest.raises(ValueError, match=R149_SOURCE):
        load_showcase(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_examples_reuse_exact_existing_sample_bytes_and_goals():
    tasks = example_tasks(ROOT)
    assert [task["id"] for task in tasks] == ["classification", "extraction", "generation"]
    variants = {"classification": None, "generation": "generation", "extraction": "extraction"}
    for task in tasks:
        goal, raw, name, description = demo_create_payload(ROOT, variants[task["id"]])
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        assert (task["goal"], task["bytes"], task["name"], task["description"]) == (
            goal,
            raw,
            name,
            description,
        )
        assert task["rows"] == rows
        assert task["row_count"] == len(rows)
        assert task["target_column"] in rows[0]
        assert set(task["input_columns"]) <= rows[0].keys()
        assert task["kind"] == "fictional_example"


def test_examples_reflect_changed_csv_without_global_cache(assets):
    task = example_tasks(assets)[0]
    path = assets / "data/custom/examples" / task["name"]
    raw = path.read_bytes().replace("杯子破损".encode(), "新咨询内容".encode())
    path.write_bytes(raw)
    assert example_tasks(assets)[0]["bytes"] == raw
    assert example_tasks(assets)[0]["rows"] != task["rows"]


@pytest.mark.parametrize("mode", ["missing", "bad_columns", "bad_encoding"])
def test_missing_or_malformed_csv_is_not_replaced_with_mock_rows(assets, mode):
    name = example_tasks(assets)[0]["name"]
    path = assets / "data/custom/examples" / name
    if mode == "missing":
        path.unlink()
    elif mode == "bad_columns":
        path.write_bytes(b"unexpected,columns\nx,y\n")
    else:
        path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="示例"):
        example_tasks(assets)


@pytest.mark.parametrize("mode", ["missing", "bad_columns", "bad_encoding"])
def test_tolerant_examples_isolate_one_unreadable_source_and_keep_actual_bytes(assets, mode):
    expected = example_tasks(assets)
    damaged = expected[1]
    path = assets / "data/custom/examples" / damaged["name"]
    if mode == "missing":
        path.unlink()
    elif mode == "bad_columns":
        path.write_bytes(b"unexpected,columns\nx,y\n")
    else:
        path.write_bytes(b"\xff")
    before = {
        item.relative_to(assets): item.read_bytes() if item.is_file() else None
        for item in assets.rglob("*")
    }
    healthy, errors = example_tasks_tolerant(assets)
    assert healthy == [expected[0], expected[2]]
    assert errors == [{"id": damaged["id"], "title": damaged["title"]}]
    assert str(assets) not in repr(errors)
    assert {
        item.relative_to(assets): item.read_bytes() if item.is_file() else None
        for item in assets.rglob("*")
    } == before
    with pytest.raises(ValueError, match="工单编号抽取"):
        example_tasks(assets)


def test_tolerant_examples_report_all_missing_without_writing(tmp_path):
    healthy, errors = example_tasks_tolerant(tmp_path)
    assert healthy == []
    assert [item["id"] for item in errors] == ["classification", "extraction", "generation"]
    assert all(set(item) == {"id", "title"} for item in errors)
    assert not list(tmp_path.iterdir())


def test_catalog_has_three_studies_and_preserves_simultaneous_arms():
    catalog = history_catalog(ROOT)
    assert [item["id"] for item in catalog] == ["r149", "r148-controlled-v2", "r148-product-smoke"]
    controlled, smoke = catalog[1:]
    assert controlled["counts"]["arms"] == len(controlled["results"]) == 3
    assert [item["correct"] for item in controlled["results"]] == [12, 16, 12]
    assert all(item["total"] == 16 for item in controlled["results"])
    assert [item["correct"] for item in smoke["results"]] == [0, 12]
    assert "不是三轮" in controlled["scope"]
    assert all("rows" not in item and "report" not in item for item in catalog)


def test_catalog_rechecks_r148_counts_and_rejects_missing_arms(assets):
    path, data = source(assets, R148_SOURCE)
    arm = next(iter(data["controlled_v2"]["results"]["arms"].values()))
    arm["evaluation"]["summary"]["all"]["accepted"] = 16
    save(path, data)
    with pytest.raises(ValueError, match="历史目录"):
        history_catalog(assets)
    data["controlled_v2"]["results"]["arms"].popitem()
    save(path, data)
    with pytest.raises(ValueError, match="三臂"):
        history_catalog(assets)


def test_all_readers_make_no_stores_or_writes(assets, monkeypatch):
    before = {
        path.relative_to(assets): path.read_bytes() for path in assets.rglob("*") if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Showcase readers must never write or initialize stores")

    for name in ("write_text", "write_bytes", "mkdir", "touch"):
        monkeypatch.setattr(Path, name, forbidden)
    load_showcase(assets)
    example_tasks(assets)
    history_catalog(assets)
    after = {
        path.relative_to(assets): path.read_bytes() for path in assets.rglob("*") if path.is_file()
    }
    assert after == before


def test_cold_data_import_does_not_load_model_tracking_or_streamlit_libraries():
    code = "import sys; import ui.showcase; assert not {'torch','transformers','datasets','mlflow','streamlit'} & sys.modules.keys()"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
