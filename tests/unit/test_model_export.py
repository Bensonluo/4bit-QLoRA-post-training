"""model_export 单元:环节⑨交接出口的只读盘点与合并执行。

plan_model_export 六态全覆盖(ready/already_exported/not_succeeded/
adapter_missing/base_missing/target_conflict);export_model 只在 ready 动手,
already_exported 幂等返回,其余逐条拒绝;合并复用框架层 merge_adapter_to_dir
(测试内打桩,不加载真实模型)。
"""

import json
from pathlib import Path

import pytest

import src.models.merger
from src.inference.discovery import looks_merged
from src.workbench.model_export import default_export_dir, export_model, plan_model_export


def _make_run(tmp_path, *, status="succeeded", with_adapter=True, base_exists=True):
    """构造一份最小训练记录:基座目录 + 产物目录(带/不带 adapter 文件)。"""
    base = tmp_path / "base-model"
    base.mkdir(exist_ok=True)
    adapter_dir = tmp_path / "training" / "wb-x" / "model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if with_adapter:
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_text("w", encoding="utf-8")
    else:
        # 同一 tmp_path 可能已被上一个用例写进 adapter 文件,缺件场景必须真缺。
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            (adapter_dir / name).unlink(missing_ok=True)
    return {
        "run_id": "wb-x",
        "session_id": "sess-1",
        "dataset_version": "ds-v3",
        "status": status,
        "model_path": str(base) if base_exists else str(tmp_path / "gone-base"),
        "output_dir": str(adapter_dir),
    }


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


def test_plan_already_exported_when_target_looks_merged(tmp_path):
    target = tmp_path / "merged" / "wb-x"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_text("w", encoding="utf-8")
    plan = plan_model_export(_make_run(tmp_path), target)
    assert plan["status"] == "already_exported"


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
    assert "导出目录已存在且不是完整的已合并模型" in plan["reasons"][0]


def test_export_model_merges_and_writes_evidence(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(src.models.merger, "merge_adapter_to_dir", _fake_merger(calls))
    plan = plan_model_export(_make_run(tmp_path), tmp_path / "merged" / "wb-x")
    result = export_model(plan)

    assert result["status"] == "exported"
    assert result["dataset_version"] == "ds-v3"
    # 框架层合并收到的基座来自训练记录,adapter/output 与 plan 一致。
    assert calls == [(plan["adapter_dir"], plan["output_dir"], plan["base_model_path"])]
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
    monkeypatch.setattr(src.models.merger, "merge_adapter_to_dir", _fake_merger(calls))
    target = tmp_path / "merged" / "wb-x"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_text("w", encoding="utf-8")
    plan = plan_model_export(_make_run(tmp_path), target)

    result = export_model(plan)
    assert result == {"status": "already_exported", "run_id": "wb-x", "output_dir": str(target)}
    assert calls == [], "幂等返回不得再次触发合并"


def test_export_model_refuses_blocked_plans(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(src.models.merger, "merge_adapter_to_dir", _fake_merger(calls))
    plan = plan_model_export(_make_run(tmp_path, status="failed"), tmp_path / "m")
    with pytest.raises(ValueError, match="只有成功完成的训练才能合并导出"):
        export_model(plan)
    assert calls == [], "阻塞盘点不得触发合并"
