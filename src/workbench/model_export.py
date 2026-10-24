"""Export verified training adapters with portable, content-bound evidence."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.inference.discovery import looks_merged
from src.utils.gguf_export import GGUF_QUANTIZATIONS, gguf_toolchain

_EVIDENCE = "export_evidence.json"
_LINEAGE = ("run_id", "session_id", "dataset_version", "config_digest")


def _verified_training_protocol(config: dict, preflight: dict) -> dict:
    """Check objective evidence, preserving pre-protocol full-prompt receipts."""
    from src.data.sft_protocol import ANSWER_PROTOCOL, LEGACY_PROTOCOL, validate_protocol

    if not isinstance(config, dict) or not isinstance(preflight, dict):
        raise ValueError("训练配置或监督预检不是有效对象。")
    data = config.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("训练数据配置不是有效对象。")
    configured = validate_protocol(data.get("sft_protocol", LEGACY_PROTOCOL))
    inspected = validate_protocol(preflight.get("sft_protocol", LEGACY_PROTOCOL))
    expected_strategy = (
        "full_prompt_padding_masked" if configured == LEGACY_PROTOCOL else "answer_and_eos_only"
    )
    strategy = preflight.get("supervision_strategy", "full_prompt_padding_masked")
    if configured not in {LEGACY_PROTOCOL, ANSWER_PROTOCOL} or (
        inspected != configured or strategy != expected_strategy
    ):
        raise ValueError("训练配置与预检的 SFT 监督协议不一致或尚不受证据链支持。")
    verified: dict[str, Any] = {
        "sft_protocol": configured,
        "supervision_strategy": expected_strategy,
    }
    if configured == LEGACY_PROTOCOL:
        return verified
    if preflight.get("status") not in {"passed", "warnings"}:
        raise ValueError("答案专属监督没有通过训练预检。")
    rows, splits = preflight.get("rows"), preflight.get("splits")
    if not isinstance(rows, list) or not rows or not isinstance(splits, dict) or not splits:
        raise ValueError("答案专属监督缺少完整逐行预检证据。")
    counts: Counter[str] = Counter()
    seen = set()
    answer_total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("答案专属监督的逐行预检记录无效。")
        answer_count, total = row.get("answer_supervised_tokens"), row.get("supervised_tokens")
        if (
            any(
                row.get(key) is not True
                for key in (
                    "answer_boundary_verified",
                    "answer_supervision_complete",
                    "terminal_eos_supervised",
                )
            )
            or type(row.get("prompt_supervised_tokens")) is not int
            or row["prompt_supervised_tokens"] != 0
            or type(answer_count) is not int
            or answer_count <= 0
            or type(total) is not int
            or total != answer_count + 1
            or not isinstance(row.get("split"), str)
            or not isinstance(row.get("row_id"), str)
            or not row["row_id"]
        ):
            raise ValueError("答案专属监督的答案边界、提示屏蔽、完整答案或原始 EOS 证据不一致。")
        key = (row["split"], row["row_id"])
        if key in seen:
            raise ValueError("答案专属监督的逐行预检记录重复。")
        seen.add(key)
        counts[row["split"]] += 1
        answer_total += answer_count
    if any(
        not isinstance(summary, dict)
        or type(summary.get("rows")) is not int
        or summary["rows"] < 0
        or counts[split] != summary["rows"]
        for split, summary in splits.items()
    ) or set(counts) - set(splits):
        raise ValueError("答案专属监督的逐行证据与分区行数不一致。")
    verified["supervision_totals"] = {
        "verified_rows": len(rows),
        "prompt_supervised_tokens": 0,
        "answer_supervised_tokens": answer_total,
        "terminal_eos_supervised_tokens": len(rows),
        "supervised_tokens": answer_total + len(rows),
    }
    return verified


def default_export_dir(
    training_root: Path, run_id: str, *, export_format: str = "hf", quantization: str = "q4_k_m"
) -> Path:
    """Place exports outside the training tree, named by their originating run."""
    if export_format == "gguf":
        return Path(training_root).parent / "gguf" / f"{run_id}-{quantization}"
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
    from src.workbench.sources import content_digest
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
                "config",
                "preflight",
            )
        ):
            raise ValueError("训练运行记录已经变化，请重新获取当前运行。")
    config_path = run_directory / "config.json"
    config = record.get("config")
    if config is not None or config_path.exists():
        if (
            not isinstance(config, dict)
            or content_digest(config) != record["config_digest"]
            or (
                config_path.exists()
                and json.loads(config_path.read_text(encoding="utf-8")) != config
            )
        ):
            raise ValueError("训练配置内容与运行或产物清单的配置摘要不一致。")
        training_protocol = _verified_training_protocol(config, record.get("preflight") or {})
    else:
        training_protocol = _verified_training_protocol({}, record.get("preflight") or {})
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
    source: dict[str, Any] = {
        "base_model": identity,
        "adapter_files": adapter_files,
        "training_manifest_sha256": file_digest(manifest_path),
    }
    # Old receipts had no explicit protocol. Preserve their existing identities
    # so already exported historical models remain idempotently verifiable.
    if isinstance(config, dict) and "sft_protocol" in config.get("data", {}):
        source["training_protocol"] = training_protocol
    return source


def _verified_export(target: Path, plan: dict, *, verify_content: bool = True) -> None:
    export_format = plan.get("export_format", "hf")
    if export_format == "hf" and not looks_merged(target):
        raise ValueError("目录不是完整的已合并模型。")
    if export_format == "gguf":
        gguf = target / f"model-{plan['quantization']}.gguf"
        if not gguf.is_file() or not (target / "Modelfile").is_file():
            raise ValueError("目录缺少本次 GGUF 模型或交接文件。")
    evidence = json.loads((target / _EVIDENCE).read_text(encoding="utf-8"))
    expected = {key: plan.get(key) for key in (*_LINEAGE, "base_model_path", "adapter_dir")}
    if (
        not isinstance(evidence, dict)
        or evidence.get("format_version") != 1
        or evidence.get("export_format", "hf") != export_format
        or (export_format == "gguf" and evidence.get("quantization") != plan["quantization"])
        or any(evidence.get(key) != value for key, value in expected.items())
        or not isinstance(evidence.get("source_identity"), dict)
        or evidence["source_identity"].get("base_model") != plan["source_identity"]["base_model"]
        or evidence["source_identity"].get("training_manifest_sha256")
        != plan["source_identity"]["training_manifest_sha256"]
        or (
            "training_protocol" in plan["source_identity"]
            and evidence.get("training_protocol") != plan["source_identity"]["training_protocol"]
        )
        or (verify_content and evidence.get("source_identity") != plan.get("source_identity"))
    ):
        raise ValueError("导出证据不属于本次训练或来源内容已变化。")
    if not isinstance(evidence.get("output_files"), dict) or (
        verify_content and evidence["output_files"] != _files(target, exclude={_EVIDENCE})
    ):
        raise ValueError("已导出模型内容已修改，文件摘要不匹配。")
    if export_format == "gguf" and verify_content:
        from src.utils.gguf_metadata import read_gguf_metadata

        if read_gguf_metadata(gguf) != evidence.get("conversion", {}).get("metadata"):
            raise ValueError("GGUF 实际元数据与转换证据不同。")
        if evidence.get("tokenizer_directory") is not None and (
            evidence["tokenizer_directory"] != "hf_tokenizer"
            or not (target / "hf_tokenizer").is_dir()
        ):
            raise ValueError("GGUF 的可携带 tokenizer 目录缺失或路径无效。")


def plan_model_export(
    run_record: dict,
    output_dir: str | Path,
    *,
    export_format: str = "hf",
    llama_cpp_dir: str | Path | None = None,
    quantization: str = "q4_k_m",
    _verify_content: bool = False,
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
        "export_format": export_format,
        "quantization": quantization if export_format == "gguf" else None,
        "llama_cpp_dir": str(Path(llama_cpp_dir).expanduser().resolve()) if llama_cpp_dir else None,
        # Snapshot only the fields needed to revalidate. No mutable caller references.
        "run_record": deepcopy(run_record),
    }
    has_adapter = (adapter / "adapter_config.json").is_file() and any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )
    if export_format not in {"hf", "gguf"} or (
        export_format == "gguf" and quantization not in GGUF_QUANTIZATIONS
    ):
        plan["status"] = "invalid_format"
        plan["reasons"].append("请选择 HF 或 GGUF 格式，以及支持的 GGUF 精度。")
    elif run_record.get("status") != "succeeded":
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
                if export_format == "gguf":
                    try:
                        plan["toolchain_inventory"] = gguf_toolchain(llama_cpp_dir, quantization)
                        plan["llama_cpp_dir"] = plan["toolchain_inventory"]["directory"]
                        if _verify_content:
                            plan["toolchain_identity"] = gguf_toolchain(
                                llama_cpp_dir, quantization, verify_content=True
                            )
                    except (OSError, ValueError) as exc:
                        plan["status"] = "toolchain_missing"
                        plan["reasons"].append(str(exc))
    return plan


def export_model(plan: dict) -> dict:
    """Revalidate sources and destination immediately before merging or returning."""
    state = plan.get("status")
    if state not in {"ready", "already_exported", "verification_required"}:
        raise ValueError("；".join(plan.get("reasons") or [f"当前状态 {state} 不能合并导出。"]))
    if not isinstance(plan.get("run_record"), dict):
        raise ValueError("导出计划缺少来源快照，请重新盘点。")
    options = {
        "export_format": plan.get("export_format", "hf"),
        "quantization": plan.get("quantization") or "q4_k_m",
        "llama_cpp_dir": plan.get("llama_cpp_dir"),
    }
    fresh = plan_model_export(
        plan["run_record"], plan["output_dir"], **options, _verify_content=True
    )
    fields = (
        *_LINEAGE,
        "base_model_path",
        "adapter_dir",
        "output_dir",
        "source_inventory",
        "export_format",
        "quantization",
        "llama_cpp_dir",
        "toolchain_inventory",
    )
    if any(plan.get(key) != fresh.get(key) for key in fields) or fresh["status"] not in {
        "ready",
        "already_exported",
    }:
        reason = "；".join(fresh["reasons"]) or "来源身份与盘点时不一致。"
        raise ValueError(f"导出计划已失效，请重新盘点：{reason}")
    if fresh["status"] == "already_exported":
        return _completed_result(fresh, "already_exported")
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
        published = merged_dir
        conversion = None
        merged_source_files = None
        if fresh["export_format"] == "gguf":
            from src.utils.gguf_export import convert_to_gguf

            if (
                gguf_toolchain(fresh["llama_cpp_dir"], fresh["quantization"])
                != fresh["toolchain_inventory"]
                or gguf_toolchain(
                    fresh["llama_cpp_dir"], fresh["quantization"], verify_content=True
                )
                != fresh["toolchain_identity"]
            ):
                raise ValueError("合并期间 GGUF 工具发生变化，未转换或发布模型。")
            merged_source_files = _files(merged_dir)
            published = Path(temporary) / "gguf"
            published.mkdir()
            conversion = convert_to_gguf(
                merged_dir,
                published / f"model-{fresh['quantization']}.gguf",
                quantization=fresh["quantization"],
                llama_cpp_dir=fresh["llama_cpp_dir"],
            )
            conversion["gguf_file"] = str(target / f"model-{fresh['quantization']}.gguf")
            if conversion["toolchain"] != fresh["toolchain_identity"]:
                raise ValueError("GGUF 实际转换工具与已核验工具不同，未发布模型。")
            if _files(merged_dir) != merged_source_files:
                raise ValueError("转换期间合并模型发生变化，未发布 GGUF。")
            # Use exactly the tokenizer saved by the merger/converter, including
            # Jinja templates. No weights or executable Python code are copied.
            from src.utils.model_files import config_and_tokenizer_files

            for source in config_and_tokenizer_files(merged_dir):
                destination = published / "hf_tokenizer" / source.relative_to(merged_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            if any(
                merged_source_files.get(name) != entry
                for name, entry in _files(published / "hf_tokenizer").items()
            ):
                raise ValueError("复制 tokenizer 时文件发生变化，未发布 GGUF。")
            _write_gguf_handoff(published, fresh)
        after = plan_model_export(
            fresh["run_record"], fresh["output_dir"], **options, _verify_content=True
        )
        if after.get("source_identity") != fresh["source_identity"] or after["status"] not in {
            "ready",
            "already_exported",
        }:
            raise ValueError("合并过程中训练来源或目标发生变化，未发布导出模型。")
        evidence = {
            "format_version": 1,
            "export_format": fresh["export_format"],
            "quantization": fresh["quantization"],
            **{key: fresh.get(key) for key in (*_LINEAGE, "base_model_path", "adapter_dir")},
            "source_identity": fresh["source_identity"],
            "output_files": _files(published, exclude={_EVIDENCE}),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        if "training_protocol" in fresh["source_identity"]:
            evidence["training_protocol"] = fresh["source_identity"]["training_protocol"]
        if conversion is not None:
            evidence.update(
                conversion=conversion,
                merged_source_files=merged_source_files,
                business_evaluation_status="not_run",
                prompt_protocol="render_alpaca_prompt_without_answer",
                tokenizer_directory="hf_tokenizer",
            )
        write_json(published / _EVIDENCE, evidence)
        try:
            # Atomic publication; an existing nonempty directory is never overwritten.
            published.rename(target)
        except OSError as exc:
            try:
                _verified_export(target, fresh)
            except (OSError, ValueError, TypeError, KeyError):
                raise ValueError("导出目标已变化，未覆盖现有目录；请重新盘点。") from exc
            return _completed_result(fresh, "already_exported")
    return {**evidence, **_completed_result(fresh, "exported")}


def _completed_result(plan: dict, status: str) -> dict:
    result = {
        "status": status,
        "run_id": plan["run_id"],
        "output_dir": plan["output_dir"],
    }
    if plan["export_format"] == "gguf":
        result.update(
            export_format="gguf",
            quantization=plan["quantization"],
            gguf_file=str(Path(plan["output_dir"]) / f"model-{plan['quantization']}.gguf"),
            modelfile=str(Path(plan["output_dir"]) / "Modelfile"),
            business_evaluation_status="not_run",
        )
    return result


def _write_gguf_handoff(directory: Path, plan: dict) -> None:
    filename = f"model-{plan['quantization']}.gguf"
    (directory / "Modelfile").write_text(
        f'FROM ./{filename}\nTEMPLATE """{{{{ .Prompt }}}}"""\n', encoding="utf-8"
    )
    (directory / "USAGE.md").write_text(
        "# 本地模型交接\n\n"
        f"来源训练：{plan['run_id']}；精度：{plan['quantization']}。\n\n"
        "模型经过格式转换或量化，源 HF 评测分数不能视为当前 GGUF 的业务成绩。"
        "请用同题、同生成与评分协议重新验证，当前业务验证状态为 not_run。\n\n"
        "训练和工作台评测使用下列原始提示（删除答案，保留 Response 后的换行）：\n\n"
        "```text\n### Instruction:\n[训练时的业务指令]\n\n### Input:\n[业务输入]\n\n### Response:\n```\n\n"
        "没有输入时省略整个 Input 段。不要把占位文本当作实际业务指令。"
        "Modelfile 的 TEMPLATE 直接使用传入的完整提示；Ollama 调用须提供上述完整文本。"
        "LM Studio 聊天界面可能使用基座聊天模板，与原始提示不同；"
        "业务对照应使用不附加聊天模板的原始文本生成。\n\n"
        "GGUF 内保存源 tokenizer 的聊天模板供兼容工具使用；"
        "保留它不代表训练采用了该聊天模板。hf_tokenizer/ 保存实际转换的 HF 分词文件，"
        "用于工作台的输入 token 核对与统一解码。可在成功训练记录下选择 GGUF 同题对照；"
        "评测结果保存在独立报告中，不回写导出时的 not_run 状态。"
        "文件和来源摘要见 export_evidence.json。\n",
        encoding="utf-8",
    )
