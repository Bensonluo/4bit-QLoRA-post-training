"""Export verified training adapters with portable, content-bound evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.inference.discovery import looks_merged

_EVIDENCE = "export_evidence.json"
_LINEAGE = ("run_id", "session_id", "dataset_version", "config_digest")


def default_export_dir(training_root: Path, run_id: str) -> Path:
    """Place exports outside the training tree, named by their originating run."""
    return Path(training_root).parent / "merged" / run_id


def _files(
    directory: Path,
    *,
    exclude: set[str] | None = None,
    hash_content: bool = True,
    recursive: bool = True,
) -> dict:
    # Reuse training's streamed hashes; never load model weights into memory.
    from src.workbench.training_runs import file_digest

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*") if recursive else directory.iterdir()):
        name = path.relative_to(directory).as_posix()
        if name in (exclude or set()):
            continue
        if path.is_symlink():
            raise ValueError(f"产物包含符号链接，无法作为独立交接模型：{path}")
        if path.is_file():
            stat = path.stat()
            files[name] = {"size": stat.st_size}
            if hash_content:
                files[name]["sha256"] = file_digest(path)
            else:
                files[name]["mtime_ns"] = stat.st_mtime_ns
    return files


def _source_identity(record: dict, adapter: Path, base: Path, *, verify_content: bool) -> dict:
    from src.workbench.training_runs import file_digest, model_identity

    if any(not record.get(key) for key in _LINEAGE):
        raise ValueError("训练记录缺少运行、会话、数据版本或配置摘要，不能验证导出血缘。")
    run_directory = adapter.parent
    if (run_directory / "run.json").exists():
        current = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            raise ValueError("训练运行记录无效。")
        if (run_directory / "result.json").exists():
            result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
            if not isinstance(result, dict) or result.get("run_id") != record["run_id"]:
                raise ValueError("运行结果身份不匹配。")
            current.update(result)
        if any(
            current.get(key) != record.get(key)
            for key in (
                *_LINEAGE,
                "status",
                "model_path",
                "output_dir",
                "model_identity",
                "artifacts",
            )
        ):
            raise ValueError("训练运行记录已经变化，请重新获取当前运行。")
    identity = record.get("model_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("path") != str(base)
        or not (base / "config.json").is_file()
        or (verify_content and model_identity(base) != identity)
    ):
        raise ValueError("基础模型或 tokenizer 内容已变化，与训练时的身份不匹配。")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("训练记录缺少产物清单。")
    manifest_path = Path(artifacts.get("manifest") or "")
    if manifest_path.resolve() != adapter / "workbench_manifest.json":
        raise ValueError("训练记录缺少产物清单，或清单不在本次 adapter 目录中。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != record[key] for key in ("run_id", "dataset_version", "config_digest")
    ):
        raise ValueError("训练产物清单血缘与运行记录不匹配。")
    if manifest.get("model_identity") != identity:
        raise ValueError("训练产物清单中的基础模型身份不匹配。")
    entries = manifest.get("files")
    if not isinstance(entries, dict):
        raise ValueError("训练产物清单缺少文件内容摘要。")
    required = {
        "adapter_config": "adapter_config.json",
        "adapter_weights": (
            "adapter_model.safetensors"
            if (adapter / "adapter_model.safetensors").is_file()
            else "adapter_model.bin"
        ),
        "tokenizer_config": "tokenizer_config.json",
    }
    for key, name in required.items():
        entry = entries.get(key)
        if not isinstance(entry, dict) or Path(entry.get("path") or "").resolve() != adapter / name:
            raise ValueError(f"训练产物清单缺少或错误绑定必要文件：{name}")
    # The merger reads the final adapter at the root, not historical checkpoints.
    adapter_files = _files(adapter, hash_content=verify_content, recursive=False)
    for entry in entries.values():
        if not isinstance(entry, dict):
            raise ValueError("训练产物清单中的文件记录无效。")
        path = Path(entry.get("path") or "").resolve()
        if (
            not path.is_relative_to(adapter)
            or not path.is_file()
            or not isinstance(entry.get("sha256"), str)
            or (
                verify_content
                and adapter_files[path.relative_to(adapter).as_posix()]["sha256"]
                != entry.get("sha256")
            )
        ):
            raise ValueError("训练产物已修改，内容哈希或路径不匹配。")
    return {
        "base_model": identity,
        "adapter_files": adapter_files,
        "training_manifest_sha256": file_digest(manifest_path),
    }


def _verified_export(target: Path, plan: dict, *, verify_content: bool = True) -> None:
    if not looks_merged(target):
        raise ValueError("目录不是完整的已合并模型。")
    evidence = json.loads((target / _EVIDENCE).read_text(encoding="utf-8"))
    expected = {key: plan.get(key) for key in (*_LINEAGE, "base_model_path", "adapter_dir")}
    if (
        not isinstance(evidence, dict)
        or evidence.get("format_version") != 1
        or any(evidence.get(key) != value for key, value in expected.items())
        or not isinstance(evidence.get("source_identity"), dict)
        or evidence["source_identity"].get("base_model") != plan["source_identity"]["base_model"]
        or evidence["source_identity"].get("training_manifest_sha256")
        != plan["source_identity"]["training_manifest_sha256"]
        or (verify_content and evidence.get("source_identity") != plan.get("source_identity"))
    ):
        raise ValueError("导出证据不属于本次训练或来源内容已变化。")
    if not isinstance(evidence.get("output_files"), dict) or (
        verify_content and evidence["output_files"] != _files(target, exclude={_EVIDENCE})
    ):
        raise ValueError("已导出模型内容已修改，文件摘要不匹配。")


def plan_model_export(
    run_record: dict, output_dir: str | Path, *, _verify_content: bool = False
) -> dict:
    """Read-only export preflight; directory shape alone never proves provenance."""
    target_input = Path(output_dir).expanduser()
    target = target_input.resolve()
    adapter = Path(run_record.get("output_dir") or "").expanduser().resolve()
    base_value = run_record.get("model_path")
    base = Path(str(base_value)).expanduser().resolve() if base_value else None
    plan: dict[str, Any] = {
        **{key: run_record.get(key) for key in _LINEAGE},
        "base_model_path": str(base) if base else None,
        "adapter_dir": str(adapter),
        "output_dir": str(target),
        "reasons": [],
        "verification": "full" if _verify_content else "pending",
        # Snapshot only the fields needed to revalidate. No mutable caller references.
        "run_record": deepcopy(run_record),
    }
    has_adapter = (adapter / "adapter_config.json").is_file() and any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if run_record.get("status") != "succeeded":
        plan["status"] = "not_succeeded"
        plan["reasons"].append(
            f"只有成功完成的训练才能合并导出；这次运行当前状态是 {run_record.get('status') or '未知'}。"
        )
    elif not has_adapter:
        plan["status"] = "adapter_missing"
        plan["reasons"].append(f"训练记录为成功，但产物目录缺少完整的 adapter 文件：{adapter}")
    elif base is None or not base.is_dir():
        plan["status"] = "base_missing"
        plan["reasons"].append(f"训练使用的基础模型目录当前不存在：{base_value or '记录中缺失'}")
    elif any(
        target.is_relative_to(source) or source.is_relative_to(target) for source in (adapter, base)
    ):
        plan["status"] = "path_overlap"
        plan["reasons"].append("导出目录与基础模型或 adapter 来源目录重叠，请选择独立目录。")
    elif target_input.is_symlink() or (target.exists() and not target.is_dir()):
        plan["status"] = "target_conflict"
        plan["reasons"].append(f"导出路径是符号链接或普通文件，请选择独立目录：{target_input}")
    else:
        try:
            plan["source_identity"] = _source_identity(
                run_record, adapter, base, verify_content=_verify_content
            )
            plan["source_inventory"] = {
                "adapter": _files(adapter, hash_content=False, recursive=False),
                "base": {
                    path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                    for path in sorted(base.iterdir())
                    if path.is_file()
                },
            }
        except (OSError, ValueError, TypeError, KeyError) as exc:
            plan["status"] = "source_mismatch"
            plan["reasons"].append(f"训练来源完整性校验未通过：{exc}")
        else:
            if target.exists() and any(target.iterdir()):
                try:
                    _verified_export(target, plan, verify_content=_verify_content)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    plan["status"] = "target_conflict"
                    plan["reasons"].append(f"导出目录已存在但无法验证为本次训练的完整模型：{exc}")
                else:
                    plan["status"] = (
                        "already_exported" if _verify_content else "verification_required"
                    )
            else:
                plan["status"] = "ready"
    return plan


def export_model(plan: dict) -> dict:
    """Revalidate sources and destination immediately before merging or returning."""
    state = plan.get("status")
    if state not in {"ready", "already_exported", "verification_required"}:
        raise ValueError("；".join(plan.get("reasons") or [f"当前状态 {state} 不能合并导出。"]))
    if not isinstance(plan.get("run_record"), dict):
        raise ValueError("导出计划缺少来源快照，请重新盘点。")
    fresh = plan_model_export(plan["run_record"], plan["output_dir"], _verify_content=True)
    fields = (*_LINEAGE, "base_model_path", "adapter_dir", "output_dir", "source_inventory")
    if any(plan.get(key) != fresh.get(key) for key in fields) or fresh["status"] not in {
        "ready",
        "already_exported",
    }:
        reason = "；".join(fresh["reasons"]) or "来源身份与盘点时不一致。"
        raise ValueError(f"导出计划已失效，请重新盘点：{reason}")
    if fresh["status"] == "already_exported":
        return {
            "status": "already_exported",
            "run_id": fresh["run_id"],
            "output_dir": fresh["output_dir"],
        }
    if state in {"already_exported", "verification_required"}:
        raise ValueError("已导出模型被移走，导出计划已失效，请重新盘点。")
    # Keep torch/peft/transformers out of the planning and refusal paths.
    from src.models.merger import merge_adapter_to_dir
    from src.workbench.training_runs import write_json

    target = Path(fresh["output_dir"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{target.name}-export-", dir=target.parent) as temporary:
        staged = Path(temporary) / "model"
        merged_dir = Path(
            merge_adapter_to_dir(
                fresh["adapter_dir"], str(staged), base_model_name=fresh["base_model_path"]
            )
        ).resolve()
        if merged_dir != staged or not looks_merged(merged_dir):
            raise ValueError("合并未生成预期目录中的完整模型，未写入导出证据。")
        # A long merge can race with source edits. Never attest to changed inputs.
        after = plan_model_export(fresh["run_record"], fresh["output_dir"], _verify_content=True)
        if after.get("source_identity") != fresh["source_identity"] or after["status"] not in {
            "ready",
            "already_exported",
        }:
            raise ValueError("合并过程中训练来源或目标发生变化，未发布导出模型。")
        evidence = {
            "format_version": 1,
            **{key: fresh.get(key) for key in (*_LINEAGE, "base_model_path", "adapter_dir")},
            "source_identity": fresh["source_identity"],
            "output_files": _files(merged_dir, exclude={_EVIDENCE}),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(merged_dir / _EVIDENCE, evidence)
        try:
            # Atomic publication; an existing nonempty directory is never overwritten.
            merged_dir.rename(target)
        except OSError as exc:
            try:
                _verified_export(target, fresh)
            except (OSError, ValueError, TypeError, KeyError):
                raise ValueError("导出目标已变化，未覆盖现有目录；请重新盘点。") from exc
            return {
                "status": "already_exported",
                "run_id": fresh["run_id"],
                "output_dir": str(target),
            }
    return {"status": "exported", **evidence, "output_dir": str(target)}
