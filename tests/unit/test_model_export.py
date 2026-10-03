"""model_export 单元:环节⑨交接出口的只读盘点与合并执行。

轻量盘点与执行前完整哈希校验;合并打桩,不加载真实模型。
"""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.inference.discovery import looks_merged
from src.utils.gguf_export import gguf_toolchain
from src.utils.gguf_metadata import read_gguf_metadata
from src.workbench.model_export import default_export_dir, export_model, plan_model_export
from src.workbench.training_runs import file_digest, model_identity
from tests.unit.test_gguf_export import _tools
from tests.unit.test_gguf_metadata import _fields, _write

merger = ModuleType("src.models.merger")
merger.merge_adapter_to_dir = None


def _fake_gguf(calls, *, failure=False):
    def convert(source, target, *, quantization, llama_cpp_dir):
        calls.append((source, target, quantization, llama_cpp_dir))
        if failure:
            raise ValueError("conversion failed")
        fields = [
            (key, kind, 15 if key == "general.file_type" else value)
            for key, kind, value in _fields()
        ]
        _write(Path(target), fields=fields, tensors=[("weight", [256], 12, 0)], payload=b"\0" * 144)
        return {
            "gguf_file": str(target),
            "quantization": quantization,
            "metadata": read_gguf_metadata(target),
            "toolchain": gguf_toolchain(llama_cpp_dir, quantization, verify_content=True),
            "commands": [],
            "business_evaluation_status": "not_run",
        }

    return convert


@pytest.fixture(autouse=True)
def lightweight_merger(monkeypatch):
    """The export module must be testable without installing torch/peft."""
    monkeypatch.setitem(sys.modules, "src.models.merger", merger)


def _make_run(tmp_path, *, status="succeeded", with_adapter=True, base_exists=True):
    """构造一份最小训练记录:基座目录 + 产物目录(带/不带 adapter 文件)。"""
    base = tmp_path / "base-model"
    base.mkdir(exist_ok=True)
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_text("base", encoding="utf-8")
    adapter_dir = tmp_path / "training" / "wb-x" / "model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if with_adapter:
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_text("w", encoding="utf-8")
    else:
        # 同一 tmp_path 可能已被上一个用例写进 adapter 文件,缺件场景必须真缺。
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            (adapter_dir / name).unlink(missing_ok=True)
    record = {
        "run_id": "wb-x",
        "session_id": "sess-1",
        "dataset_version": "ds-v3",
        "status": status,
        "model_path": str(base) if base_exists else str(tmp_path / "gone-base"),
        "output_dir": str(adapter_dir),
        "config_digest": "config-v1",
        "model_identity": model_identity(base),
        "artifacts": {"manifest": str(adapter_dir / "workbench_manifest.json")},
    }
    (adapter_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    manifest = {
        key: record[key] for key in ("run_id", "dataset_version", "config_digest", "model_identity")
    }
    manifest["files"] = {
        key: {"path": str(adapter_dir / name), "sha256": file_digest(adapter_dir / name)}
        for key, name in {
            "adapter_config": "adapter_config.json",
            "adapter_weights": "adapter_model.safetensors",
            "tokenizer_config": "tokenizer_config.json",
        }.items()
        if (adapter_dir / name).exists()
    }
    (adapter_dir / "workbench_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return record


def _fake_merger(calls):
    def merge(adapter_dir, output_dir, base_model_name=None, dtype="bfloat16"):
        calls.append((adapter_dir, output_dir, base_model_name))
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_text("w", encoding="utf-8")
        return str(out)

    return merge


def test_default_export_dir_is_merged_sibling_named_by_run():
    assert default_export_dir(Path("/a/workbench/training"), "wb-1") == Path(
        "/a/workbench/merged/wb-1"
    )


def test_plan_ready_copies_lineage_fields(tmp_path):
    target = tmp_path / "merged" / "wb-x"
    plan = plan_model_export(_make_run(tmp_path), target)
    assert plan["status"] == "ready"
    assert plan["run_id"] == "wb-x"
    assert plan["session_id"] == "sess-1"
    assert plan["dataset_version"] == "ds-v3"
    assert plan["base_model_path"] == str(tmp_path / "base-model")
    assert plan["adapter_dir"] == str(tmp_path / "training" / "wb-x" / "model")
    assert plan["output_dir"] == str(target)
    assert plan["reasons"] == []
    assert not target.exists(), "只读盘点不得创建导出目录"


def test_plan_refuses_foreign_model_even_when_target_looks_merged(tmp_path):
    target = tmp_path / "merged" / "wb-x"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_text("w", encoding="utf-8")
    plan = plan_model_export(_make_run(tmp_path), target)
    assert plan["status"] == "target_conflict"


def test_plan_blocked_states_name_the_cause(tmp_path):
    running = plan_model_export(_make_run(tmp_path, status="running"), tmp_path / "m")
    assert running["status"] == "not_succeeded"
    assert running["reasons"] == ["只有成功完成的训练才能合并导出；这次运行当前状态是 running。"]

    no_adapter = plan_model_export(_make_run(tmp_path, with_adapter=False), tmp_path / "m")
    assert no_adapter["status"] == "adapter_missing"
    assert "缺少完整的 adapter 文件" in no_adapter["reasons"][0]

    gone_base = plan_model_export(_make_run(tmp_path, base_exists=False), tmp_path / "m")
    assert gone_base["status"] == "base_missing"
    assert "基础模型目录当前不存在" in gone_base["reasons"][0]

    missing_base_field = _make_run(tmp_path)
    missing_base_field["model_path"] = None
    no_base = plan_model_export(missing_base_field, tmp_path / "m")
    assert no_base["status"] == "base_missing"
    assert "记录中缺失" in no_base["reasons"][0]


def test_plan_target_conflict_when_dir_holds_foreign_content(tmp_path):
    target = tmp_path / "merged" / "wb-x"
    target.mkdir(parents=True)
    (target / "someone-elses.txt").write_text("x", encoding="utf-8")
    plan = plan_model_export(_make_run(tmp_path), target)
    assert plan["status"] == "target_conflict"
    assert "无法验证为本次训练的完整模型" in plan["reasons"][0]


def test_export_model_merges_and_writes_evidence(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    plan = plan_model_export(_make_run(tmp_path), tmp_path / "merged" / "wb-x")
    result = export_model(plan)

    assert result["status"] == "exported"
    assert result["dataset_version"] == "ds-v3"
    # 框架层合并收到的基座来自训练记录,adapter/output 与 plan 一致。
    assert len(calls) == 1
    assert calls[0][0] == plan["adapter_dir"]
    assert calls[0][2] == plan["base_model_path"]
    assert calls[0][1] != plan["output_dir"], "合并先写临时目录，成功后原子发布"
    evidence_path = Path(result["output_dir"]) / "export_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["run_id"] == "wb-x"
    assert evidence["session_id"] == "sess-1"
    assert evidence["dataset_version"] == "ds-v3"
    assert evidence["exported_at"]
    # 证据文件不破坏幂等判定:目录仍是自包含已合并模型。
    assert looks_merged(Path(result["output_dir"]))


def test_export_model_already_exported_is_idempotent(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    target = tmp_path / "merged" / "wb-x"
    run = _make_run(tmp_path)
    export_model(plan_model_export(run, target))
    calls.clear()
    plan = plan_model_export(run, target)
    assert plan["status"] == "verification_required"
    assert plan["verification"] == "pending"

    result = export_model(plan)
    assert result == {"status": "already_exported", "run_id": "wb-x", "output_dir": str(target)}
    assert calls == [], "幂等返回不得再次触发合并"


def test_export_model_refuses_blocked_plans(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    plan = plan_model_export(_make_run(tmp_path, status="failed"), tmp_path / "m")
    with pytest.raises(ValueError, match="只有成功完成的训练才能合并导出"):
        export_model(plan)
    assert calls == [], "阻塞盘点不得触发合并"


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors"])
def test_export_rejects_changed_output_after_plan(tmp_path, monkeypatch, filename):
    calls = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    export_model(plan_model_export(run, target))
    plan = plan_model_export(run, target)
    (target / filename).write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="导出计划已失效"):
        export_model(plan)
    assert len(calls) == 1


@pytest.mark.parametrize("source", ["base", "adapter", "manifest"])
def test_export_rejects_changed_sources_before_merge(tmp_path, monkeypatch, source):
    calls = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    plan = plan_model_export(run, target)
    path = {
        "base": Path(run["model_path"]) / "model.safetensors",
        "adapter": Path(run["output_dir"]) / "adapter_model.safetensors",
        "manifest": Path(run["artifacts"]["manifest"]),
    }[source]
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="导出计划已失效"):
        export_model(plan)
    assert calls == []
    assert not target.exists()


@pytest.mark.parametrize(
    "filename",
    [
        "chat_template.jinja",
        "chat_templates/tool_use.jinja",
        "additional_chat_templates/tool_use.jinja",
    ],
)
def test_fresh_export_rejects_templates_changed_since_training(tmp_path, monkeypatch, filename):
    """A new plan must not legitimize a template changed after its training identity was saved."""
    calls = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    run = _make_run(tmp_path)
    template = Path(run["model_path"]) / filename
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("training template")
    run["model_identity"] = model_identity(Path(run["model_path"]))
    manifest_path = Path(run["artifacts"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["model_identity"] = run["model_identity"]
    manifest_path.write_text(json.dumps(manifest))
    target = tmp_path / "merged"
    assert plan_model_export(run, target, _verify_content=True)["status"] == "ready"

    template.write_text("modified template")
    fresh = plan_model_export(run, target)
    with pytest.raises(ValueError, match="导出计划已失效.*基础模型或 tokenizer 内容已变化"):
        export_model(fresh)
    assert plan_model_export(run, target, _verify_content=True)["status"] == "source_mismatch"
    assert calls == []
    assert not target.exists()


def test_export_refuses_legacy_identity_that_did_not_record_existing_template(tmp_path):
    run = _make_run(tmp_path)
    (Path(run["model_path"]) / "chat_template.jinja").write_text("unrecorded training input")
    full = plan_model_export(run, tmp_path / "merged", _verify_content=True)
    assert full["status"] == "source_mismatch"
    assert "基础模型或 tokenizer 内容已变化" in full["reasons"][0]


def test_export_rechecks_target_before_side_effects(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    target = tmp_path / "merged"
    plan = plan_model_export(_make_run(tmp_path), target)
    target.mkdir()
    (target / "foreign.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="导出计划已失效"):
        export_model(plan)
    assert calls == []
    assert (target / "foreign.txt").read_text() == "mine"


@pytest.mark.parametrize("source", ["model_path", "output_dir"])
@pytest.mark.parametrize("relation", ["same", "child", "parent", "symlink"])
def test_export_protects_overlapping_sources(tmp_path, monkeypatch, source, relation):
    calls = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    run = _make_run(tmp_path)
    directory = Path(run[source])
    target = directory
    if relation == "child":
        target = directory / "new-export"
    elif relation == "parent":
        target = directory.parent
    elif relation == "symlink":
        target = tmp_path / "source-link"
        target.symlink_to(directory, target_is_directory=True)
    plan = plan_model_export(run, target)
    assert plan["status"] == "path_overlap"
    with pytest.raises(ValueError, match="来源目录重叠"):
        export_model(plan)
    assert calls == []


def test_export_binds_evidence_to_run_and_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger([]))
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    export_model(plan_model_export(run, target))
    evidence_path = target / "export_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["run_id"] = "other-run"
    evidence_path.write_text(json.dumps(evidence))
    assert plan_model_export(run, target)["status"] == "target_conflict"


def test_export_revalidates_current_run_record(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger(calls))
    run = _make_run(tmp_path)
    record_path = Path(run["output_dir"]).parent / "run.json"
    record_path.write_text(json.dumps(run))
    plan = plan_model_export(run, tmp_path / "merged")
    record_path.write_text(json.dumps({**run, "status": "failed"}))
    with pytest.raises(ValueError, match="运行记录已经变化"):
        export_model(plan)
    assert calls == []


def test_source_edit_during_merge_never_publishes(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    fake = _fake_merger([])

    def merge(*args, **kwargs):
        output = fake(*args, **kwargs)
        (Path(run["output_dir"]) / "adapter_model.safetensors").write_text("changed")
        return output

    monkeypatch.setattr(merger, "merge_adapter_to_dir", merge)
    with pytest.raises(ValueError, match="未发布"):
        export_model(plan_model_export(run, target))
    assert not target.exists()
    assert not list(tmp_path.glob(".merged-export-*"))


def test_plan_does_not_hash_model_weights(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    from src.workbench import training_runs

    original = training_runs.file_digest
    hashes = []

    def digest(path):
        hashes.append(path.name)
        return original(path)

    monkeypatch.setattr(training_runs, "file_digest", digest)
    assert plan_model_export(run, tmp_path / "merged")["status"] == "ready"
    assert hashes == ["workbench_manifest.json"]


def test_existing_export_plan_defers_weight_hashing(tmp_path, monkeypatch):
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger([]))
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    export_model(plan_model_export(run, target))
    from src.workbench import training_runs

    original = training_runs.file_digest
    hashes = []

    def digest(path):
        hashes.append(path.name)
        return original(path)

    monkeypatch.setattr(training_runs, "file_digest", digest)
    assert plan_model_export(run, target)["status"] == "verification_required"
    assert hashes == ["workbench_manifest.json"]


def test_failed_merge_keeps_target_and_removes_staging(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    target.mkdir()

    def merge(adapter, output, **kwargs):
        output = Path(output)
        output.mkdir()
        (output / "model.safetensors").write_text("partial")
        raise RuntimeError("out of memory")

    monkeypatch.setattr(merger, "merge_adapter_to_dir", merge)
    with pytest.raises(RuntimeError, match="out of memory"):
        export_model(plan_model_export(run, target))
    assert list(target.iterdir()) == []
    assert not list(tmp_path.glob(".merged-export-*"))


def test_concurrent_foreign_target_is_never_overwritten(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    fake = _fake_merger([])

    def merge(*args, **kwargs):
        output = fake(*args, **kwargs)
        target.mkdir()
        (target / "foreign.txt").write_text("existing")
        return output

    monkeypatch.setattr(merger, "merge_adapter_to_dir", merge)
    with pytest.raises(ValueError, match="未发布"):
        export_model(plan_model_export(run, target))
    assert (target / "foreign.txt").read_text() == "existing"
    assert not (target / "export_evidence.json").exists()


@pytest.mark.parametrize("value", [[], None, {"files": []}])
def test_plan_refuses_malformed_training_manifest(tmp_path, value):
    run = _make_run(tmp_path)
    Path(run["artifacts"]["manifest"]).write_text(json.dumps(value))
    assert plan_model_export(run, tmp_path / "merged")["status"] == "source_mismatch"


def test_same_size_same_timestamp_output_change_still_fails_hash(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger([]))
    run = _make_run(tmp_path)
    target = tmp_path / "merged"
    export_model(plan_model_export(run, target))
    plan = plan_model_export(run, target)
    weight = target / "model.safetensors"
    original = weight.stat()
    weight.write_text("z")  # one byte, just like the original
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="文件摘要不匹配"):
        export_model(plan)


def test_export_to_empty_directory_is_published_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger([]))
    target = tmp_path / "merged"
    target.mkdir()
    result = export_model(plan_model_export(_make_run(tmp_path), target))
    assert result["status"] == "exported"
    assert looks_merged(target)
    assert not list(tmp_path.glob(".merged-export-*"))


def test_gguf_default_directory_keeps_formats_and_precisions_separate():
    assert default_export_dir(Path("/a/training"), "wb-x", export_format="gguf") == Path(
        "/a/gguf/wb-x-q4_k_m"
    )


def test_gguf_tool_absence_blocks_before_merging(tmp_path, monkeypatch):
    monkeypatch.delenv("TUNESMITH_LLAMA_CPP", raising=False)
    run = _make_run(tmp_path)
    plan = plan_model_export(run, tmp_path / "gguf", export_format="gguf")
    assert plan["status"] == "toolchain_missing"
    assert "setup_gguf_tools" in plan["reasons"][0]
    with pytest.raises(ValueError):
        export_model(plan)
    assert not (tmp_path / "gguf").exists()


def test_gguf_handoff_is_content_bound_and_repeat_needs_no_conversion_tools(tmp_path, monkeypatch):
    run, tools, calls = _make_run(tmp_path), _tools(tmp_path), []
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger([]))
    monkeypatch.setattr("src.utils.gguf_export.convert_to_gguf", _fake_gguf(calls))
    target = tmp_path / "gguf"
    plan = plan_model_export(run, target, export_format="gguf", llama_cpp_dir=tools)
    result = export_model(plan)
    assert result["status"] == "exported" and result["export_format"] == "gguf"
    assert result["business_evaluation_status"] == "not_run"
    assert result["conversion"]["gguf_file"] == result["gguf_file"]
    assert Path(result["gguf_file"]).is_file() and Path(result["modelfile"]).is_file()
    assert 'TEMPLATE """{{ .Prompt }}"""' in Path(result["modelfile"]).read_text()
    assert "not_run" in (target / "USAGE.md").read_text()
    assert not list(tmp_path.glob(".gguf-export-*"))
    repeat = export_model(plan_model_export(run, target, export_format="gguf"))
    assert repeat["status"] == "already_exported" and repeat["gguf_file"] == result["gguf_file"]
    assert len(calls) == 1
    assert plan_model_export(run, target, export_format="hf")["status"] == "target_conflict"
    assert (
        plan_model_export(run, target, export_format="gguf", quantization="q8_0")["status"]
        == "target_conflict"
    )
    path = Path(result["gguf_file"])
    path.write_bytes(path.read_bytes()[:-1] + b"\x01")
    with pytest.raises(ValueError, match="文件摘要不匹配"):
        export_model(plan_model_export(run, target, export_format="gguf"))


def test_gguf_conversion_failure_does_not_publish_or_keep_partial_directory(tmp_path, monkeypatch):
    run, tools = _make_run(tmp_path), _tools(tmp_path)
    monkeypatch.setattr(merger, "merge_adapter_to_dir", _fake_merger([]))
    monkeypatch.setattr("src.utils.gguf_export.convert_to_gguf", _fake_gguf([], failure=True))
    target = tmp_path / "gguf"
    with pytest.raises(ValueError, match="conversion failed"):
        export_model(plan_model_export(run, target, export_format="gguf", llama_cpp_dir=tools))
    assert not target.exists() and not list(tmp_path.glob(".gguf-export-*"))


def test_tools_changed_during_merge_are_refused_before_conversion(tmp_path, monkeypatch):
    run, tools, calls = _make_run(tmp_path), _tools(tmp_path), []
    original = _fake_merger([])

    def merge(*args, **kwargs):
        result = original(*args, **kwargs)
        (tools / "convert_hf_to_gguf.py").write_text("# replaced tool")
        return result

    monkeypatch.setattr(merger, "merge_adapter_to_dir", merge)
    monkeypatch.setattr("src.utils.gguf_export.convert_to_gguf", _fake_gguf(calls))
    target = tmp_path / "gguf"
    with pytest.raises(ValueError, match="合并期间 GGUF 工具"):
        export_model(plan_model_export(run, target, export_format="gguf", llama_cpp_dir=tools))
    assert not target.exists() and calls == []
