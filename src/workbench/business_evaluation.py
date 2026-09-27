"""Separate development comparison and single-model final acceptance evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from src.workbench.intake_models import IntakeSession
from src.workbench.materialize import dataset_is_current
from src.workbench.sources import canonical, content_digest


@dataclass(frozen=True)
class EvaluationModel:
    label: str
    base_model: str
    adapter_path: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class EvaluationProtocol:
    scorer: Literal["classification_exact", "json_fields_exact", "open_review", "custom_rules"]
    fields: tuple[str, ...] = ()
    max_new_tokens: int = 256
    strip_whitespace: bool = True
    custom_scoring: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.scorer not in {
            "classification_exact",
            "json_fields_exact",
            "open_review",
            "custom_rules",
        }:
            raise ValueError("当前评分规则不受支持；自定义业务评分需先实现并验证。")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须为正整数。")
        if type(self.strip_whitespace) is not bool:
            raise ValueError("空白处理规则必须明确为布尔值。")
        if self.scorer == "json_fields_exact" and (
            not self.fields or len(set(self.fields)) != len(self.fields)
        ):
            raise ValueError("结构化评分必须声明不重复的业务字段。")
        if self.scorer != "json_fields_exact" and self.fields:
            raise ValueError("仅结构化评分可以指定字段。")
        if (self.scorer == "custom_rules") != (
            isinstance(self.custom_scoring, dict) and bool(self.custom_scoring)
        ):
            raise ValueError("自定义评分必须绑定已确认规则，其他评分方式不能携带自定义规则。")
        if self.scorer != "custom_rules" and self.custom_scoring is not None:
            raise ValueError("非自定义评分不能携带自定义规则。")


@dataclass(frozen=True)
class Generation:
    text: str
    truncated: bool = False
    generated_tokens: int | None = None


def protocol_settings(protocol: EvaluationProtocol) -> dict:
    """Preserve existing built-in protocol identities when no custom rule is attached."""
    value = asdict(protocol)
    if value["custom_scoring"] is None:
        value.pop("custom_scoring")
    return value


class EvaluationRuntime(Protocol):
    def generate(self, prompt: str, protocol: EvaluationProtocol) -> Generation: ...
    def close(self) -> None: ...


@dataclass
class EvaluationReport:
    evaluation_id: str
    created_at: str
    dataset: dict[str, Any]
    protocol: dict[str, Any]
    comparison_key: str
    status: str = "running"
    models: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def assert_comparable(left: EvaluationReport, right: EvaluationReport) -> None:
    if any(report.dataset.get("purpose") != "development_only" for report in (left, right)):
        raise ValueError("最终验收报告不能用于开发对照、模型筛选或调参。")
    if any(
        report.status not in {"completed", "completed_with_failures"} for report in (left, right)
    ):
        raise ValueError("评测尚未完整执行，不能作为已完成对照比较。")
    if any(
        report.comparison_key != comparison_key(report.dataset, report.protocol)
        for report in (left, right)
    ):
        raise ValueError("评测报告的比较身份与实际数据/协议不一致。")
    if left.comparison_key != right.comparison_key:
        raise ValueError("开发评测套件、提示、答案或评分/生成协议不同，不能直接比较这些结果。")


def comparison_key(dataset: dict[str, Any], protocol: dict[str, Any]) -> str:
    """Fixed-suite scoring identity excludes mutable training-data provenance."""
    suite = dataset.get("evaluation_suite")
    if suite:
        if not all(suite.get(key) for key in ("suite_id", "cases_digest")):
            raise ValueError("固定评测套件缺少内容身份。")
        if not protocol.get("evaluation_key") or not protocol.get("answers_digest"):
            raise ValueError("固定评测套件缺少开发集与标准答案指纹。")
        return content_digest(
            {
                "evaluation_suite": suite,
                "split": dataset["split"],
                "purpose": dataset["purpose"],
                "row_count": dataset["row_count"],
                "protocol": protocol,
            }
        )
    return content_digest({"dataset": dataset, "protocol": protocol})


def _model_directory(reference: str, revision: str | None = None) -> Path:
    path = Path(reference).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(reference, revision=revision, local_files_only=True)
        ).resolve()
    except Exception as exc:
        raise ValueError(
            f"找不到本地模型或已缓存快照：{reference}；评测不会自动下载模型。"
        ) from exc


def _model_identity(directory: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not (
            path.suffix in {".json", ".safetensors", ".model", ".txt"}
            or path.name.startswith(("pytorch_model", "adapter_model"))
            and path.suffix == ".bin"
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        files[path.name] = digest.hexdigest()
    if not files or not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError(f"模型目录没有可识别的权重文件：{directory}")
    return {"directory": str(directory), "files": files, "content_digest": content_digest(files)}


class LocalEvaluationRuntime:
    """Reuse the platform-aware loader; no background generation thread survives close."""

    def __init__(self, model: EvaluationModel):
        self.model = None
        self.tokenizer = None
        try:
            from config.base import ModelConfig
            from src.models.loader import load_model_and_tokenizer

            self.model, self.tokenizer = load_model_and_tokenizer(
                ModelConfig(
                    name=model.base_model,
                    quantization_bits=None,
                    trust_remote_code=False,
                    use_flash_attention=False,
                )
            )
            if model.adapter_path:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, model.adapter_path)
            self.model.eval()
        except Exception:
            self.close()
            raise

    def generate(self, prompt: str, protocol: EvaluationProtocol) -> Generation:
        import torch
        from transformers import GenerationConfig

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_length = encoded["input_ids"].shape[-1]
        context = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context, int) and input_length + protocol.max_new_tokens > context:
            raise ValueError("样本与生成长度超过模型上下文；没有静默截断业务输入。")
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        eos = self.tokenizer.eos_token_id
        config = GenerationConfig(
            max_new_tokens=protocol.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            result = self.model.generate(**encoded, generation_config=config)
        tokens = result[0, input_length:]
        ended = bool(
            len(tokens)
            and eos is not None
            and int(tokens[-1]) in (eos if isinstance(eos, list) else [eos])
        )
        return Generation(
            text=self.tokenizer.decode(tokens, skip_special_tokens=True),
            truncated=len(tokens) >= protocol.max_new_tokens and not ended,
            generated_tokens=len(tokens),
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _strict_object(text: str) -> dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON 答案含重复字段。")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    canonical(value)  # Reject NaN/Infinity and preserve typed equality.
    if not isinstance(value, dict):
        raise ValueError("答案必须是 JSON 对象。")
    return value


def _score(
    output: str, expected: str, protocol: EvaluationProtocol
) -> tuple[bool | None, dict[str, bool]]:
    if protocol.scorer == "open_review":
        return None, {}
    if protocol.scorer == "custom_rules":
        raise ValueError("自定义规则必须通过隔离批量评分执行，不能在宿主逐行执行。")
    if protocol.scorer == "classification_exact":
        return (
            output.strip() == expected.strip() if protocol.strip_whitespace else output == expected
        ), {}
    actual, target = _strict_object(output), _strict_object(expected)
    fields = {
        name: name in actual and canonical(actual[name]) == canonical(target[name])
        for name in protocol.fields
    }
    return all(fields.values()), fields


def confirmed_scoring_identity(protocol: EvaluationProtocol, session: IntakeSession):
    """Load the confirmed rule as data; executable code runs only inside the sandbox."""
    if protocol.scorer != "custom_rules":
        return None, {}
    from src.workbench.business_scoring import load_confirmed

    recipe = load_confirmed(protocol.custom_scoring, session)
    payload = recipe.model_dump()
    return recipe, {
        "custom_scoring_recipe": payload,
        "custom_scoring_digest": content_digest(payload),
    }


def custom_score_observations(recipe, rows, records):
    from src.workbench.business_scoring import score_outputs

    inputs = [
        {
            "__row_id": str(row["index"]),
            "input": records[row["index"]]["input"],
            "expected": row["expected"],
            "output": row["output"],
        }
        for row in rows
        if row["status"] in {"scored", "awaiting_custom_score"} and not row["truncated"]
    ]
    if not inputs:
        return {}
    outputs = score_outputs(recipe, inputs)
    if not isinstance(outputs, list) or len(outputs) != len(inputs):
        raise ValueError("自定义评分没有返回全部原始记录。")
    scored = {}
    for source, output in zip(inputs, outputs):
        if any(output.get(key) != value for key, value in source.items()):
            raise ValueError("自定义评分修改了原始记录或顺序。")
        score, reason = output.get("score"), output.get("reason")
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("自定义评分必须返回 0 到 1 的有限分数及明确理由。")
        scored[int(source["__row_id"])] = {"score": score, "reason": reason}
    return scored


class BusinessEvaluationService:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_report(self, evaluation_id: str) -> EvaluationReport:
        if not re.fullmatch(r"[0-9a-f]{32}", evaluation_id):
            raise ValueError("无效业务评测 ID。")
        try:
            data = json.loads((self.root / f"{evaluation_id}.json").read_text(encoding="utf-8"))
            report = EvaluationReport(**data)
            if report.evaluation_id != evaluation_id or report.comparison_key != comparison_key(
                report.dataset, report.protocol
            ):
                raise ValueError("评测报告身份与数据/协议内容不一致。")
        except FileNotFoundError as exc:
            raise ValueError("找不到这个业务评测报告。") from exc
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("业务评测报告格式无效。") from exc
        return report

    def list_reports(
        self,
        dataset_version: str | None = None,
        *,
        suite_id: str | None = None,
        purpose: Literal["development_only", "final_acceptance"] | None = "development_only",
    ) -> list[EvaluationReport]:
        if purpose not in (None, "development_only", "final_acceptance"):
            raise ValueError("评测报告用途必须为开发对照或最终验收。")
        reports = [
            self.get_report(path.stem)
            for path in self.root.glob("*.json")
            if re.fullmatch(r"[0-9a-f]{32}", path.stem)
        ]
        return sorted(
            (
                report
                for report in reports
                if (dataset_version is None or report.dataset["version"] == dataset_version)
                and (
                    purpose is None or report.dataset.get("purpose", "development_only") == purpose
                )
                and (
                    suite_id is None
                    or report.dataset.get("evaluation_suite", {}).get("suite_id") == suite_id
                )
            ),
            key=lambda report: report.created_at,
            reverse=True,
        )

    def _save(self, report: EvaluationReport) -> None:
        path = self.root / f"{report.evaluation_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def compare(
        self,
        session: IntakeSession,
        models: list[EvaluationModel],
        protocol: EvaluationProtocol,
        *,
        runtime_factory: Callable[[EvaluationModel], EvaluationRuntime] | None = None,
    ) -> EvaluationReport:
        """Compare only the fixed development partition; final-test evaluation is separate."""
        protocol.validate()
        if (
            len(models) < 2
            or len({model.label for model in models}) != len(models)
            or any(not model.label.strip() for model in models)
        ):
            raise ValueError("至少提供两个名称不重复的模型，通常为基座与微调版本。")
        return self._evaluate(
            session,
            models,
            protocol,
            split="validation",
            purpose="development_only",
            runtime_factory=runtime_factory,
        )

    def evaluate_final(
        self,
        session: IntakeSession,
        model: EvaluationModel,
        protocol: EvaluationProtocol,
        *,
        runtime_factory: Callable[[EvaluationModel], EvaluationRuntime] | None = None,
    ) -> EvaluationReport:
        """Evaluate one previously selected model on fixed final-test cases only."""
        if not isinstance(model, EvaluationModel) or not model.label.strip():
            raise ValueError("最终验收只接受一个已选定且名称明确的模型。")
        if not dataset_is_current(session) or not session.dataset.evaluation_suite:
            raise ValueError("最终验收需要当前有效数据绑定已冻结的固定测试套件。")
        return self._evaluate(
            session,
            [model],
            protocol,
            split="test",
            purpose="final_acceptance",
            runtime_factory=runtime_factory,
        )

    def _evaluate(
        self,
        session: IntakeSession,
        models: list[EvaluationModel],
        protocol: EvaluationProtocol,
        *,
        split: str,
        purpose: str,
        runtime_factory: Callable[[EvaluationModel], EvaluationRuntime] | None,
    ) -> EvaluationReport:
        protocol.validate()
        if not dataset_is_current(session):
            raise ValueError("请先确认当前全量资料并生成有效的数据版本。")
        recipe = session.analysis.recipe
        scoring_recipe, scoring_identity = confirmed_scoring_identity(protocol, session)
        if protocol.scorer == "classification_exact" and (
            recipe.output_format != "text"
            or len(recipe.targets) != 1
            or recipe.targets[0].value_kind != "categorical"
        ):
            raise ValueError("分类精确评分仅用于已确认的单字段类别任务；开放任务需要业务评分规则。")
        if protocol.scorer == "json_fields_exact" and (
            recipe.output_format != "json"
            or not set(protocol.fields) <= {target.label for target in recipe.targets}
        ):
            raise ValueError("结构化评分字段必须属于已确认的 JSON 答案方案。")
        from src.data.loaders import render_alpaca_prompt
        from src.data_flywheel.dataset_registry import LocalDatasetRegistry
        from src.workbench.training_preflight import _verified_partitions

        verification = {"issues": []}
        records = _verified_partitions(session, verification)[split]
        if any(issue["severity"] == "blocking" for issue in verification["issues"]):
            raise ValueError("当前分区存在交叉泄漏，不能用于独立评测。")
        suite = None
        evaluation_key = None
        if getattr(session.dataset, "evaluation_suite", None):
            from src.workbench.evaluation_suites import evaluation_cases

            selected = evaluation_cases(session, split=split)
            records = selected["records"]
            # The report carries the full suite reference (manifest location included)
            # so the lineage is auditable from the report alone; the per-split
            # question identity lives in the protocol, where the comparison key
            # already binds it.
            suite = dict(session.dataset.evaluation_suite)
            evaluation_key = selected["evaluation_key"]
        if not records:
            raise ValueError("当前评测套件没有可评分样本。")
        if protocol.scorer == "json_fields_exact":
            for row in records:
                target = _strict_object(row["output"])
                if not set(protocol.fields) <= set(target):
                    raise ValueError("评测集标准答案缺少已声明业务评分字段。")
        artifact = session.dataset
        manifest = LocalDatasetRegistry(artifact.registry_root).get_split_manifest(
            artifact.name, artifact.version
        )
        prompts = [render_alpaca_prompt({**row, "output": ""}) for row in records]
        protocol_identity = {
            **protocol_settings(protocol),
            **scoring_identity,
            "generation": "greedy_v1",
            "prompt_renderer": "render_alpaca_prompt_without_answer",
            "prompt_digest": content_digest(prompts),
        }
        if evaluation_key is not None:
            protocol_identity["evaluation_key"] = evaluation_key
        dataset = {
            "name": artifact.name,
            "version": artifact.version,
            "source_digest": artifact.source_digest,
            "recipe_digest": artifact.recipe_digest,
            "split": split,
            "purpose": purpose,
            "split_sha256": manifest["splits"][split]["sha256"],
            "row_count": len(records),
        }
        if suite:
            dataset["evaluation_suite"] = suite
            protocol_identity["answers_digest"] = content_digest([row["output"] for row in records])
        report = EvaluationReport(
            uuid4().hex,
            datetime.now(timezone.utc).isoformat(),
            dataset,
            protocol_identity,
            comparison_key(dataset, protocol_identity),
            notes=["全部开发集行纳入分母，包括生成失败与截断；独立测试集不用于此对照或自动选参。"],
        )
        if suite:
            report.notes[0] = (
                "全部固定开发评测题纳入分母，包括生成失败与截断；独立测试集不用于此对照或自动选参。"
            )
            report.notes.append("仅固定套件的评分题纳入分母；同组新增保留行不悄悄扩充评分题。")
        if purpose == "final_acceptance":
            report.notes[0] = (
                "仅验收已选定模型的固定最终测试题；全部题纳入分母，包括生成失败与截断，不用于模型筛选或调参。"
            )
        if protocol.scorer == "open_review":
            report.notes.append(
                "开放任务尚无可执行业务评分规则，仅保留输出供核对，不宣称质量达标。"
            )
        self._save(report)
        identity_cache = {}
        for model in models:
            outcome = {
                "label": model.label,
                "requested_model": asdict(model),
                "identity": {},
                "rows": [],
                "errors": [],
            }
            runtime = None
            try:
                base = _model_directory(model.base_model, model.revision)
                adapter = _model_directory(model.adapter_path) if model.adapter_path else None
                for path in (base, adapter):
                    if path is not None:
                        current_identity = _model_identity(path)
                        previous = identity_cache.get(path)
                        if (
                            previous
                            and previous["content_digest"] != current_identity["content_digest"]
                        ):
                            raise ValueError(
                                "同一对照运行期间模型文件发生变化，不能复用旧模型身份继续比较。"
                            )
                        identity_cache[path] = current_identity
                outcome["identity"] = {
                    "base": identity_cache[base],
                    "adapter": identity_cache.get(adapter),
                }
                resolved = EvaluationModel(
                    model.label, str(base), str(adapter) if adapter else None, model.revision
                )
                runtime = (runtime_factory or LocalEvaluationRuntime)(resolved)
                for index, (row, prompt) in enumerate(zip(records, prompts)):
                    result = {
                        "index": index,
                        "source": row["metadata"],
                        "prompt": prompt,
                        "expected": row["output"],
                        "output": None,
                        "status": "failed",
                        "correct": False if protocol.scorer != "open_review" else None,
                        "field_scores": {},
                        "truncated": False,
                        "generated_tokens": None,
                        "error": "",
                    }
                    try:
                        generation = runtime.generate(prompt, protocol)
                        if not isinstance(generation, Generation) or not isinstance(
                            generation.text, str
                        ):
                            raise ValueError("模型生成器没有返回有效 Generation。")
                        result.update(
                            output=generation.text,
                            truncated=generation.truncated,
                            generated_tokens=generation.generated_tokens,
                        )
                        if generation.truncated:
                            result.update(
                                status="truncated",
                                error="达到生成长度限制，输出可能不完整；未作为成功答案。",
                            )
                        else:
                            if scoring_recipe is not None:
                                result["status"] = "awaiting_custom_score"
                            else:
                                result["correct"], result["field_scores"] = _score(
                                    generation.text, row["output"], protocol
                                )
                                result["status"] = (
                                    "needs_business_review"
                                    if protocol.scorer == "open_review"
                                    else "scored"
                                )
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                    outcome["rows"].append(result)
            except Exception as exc:
                outcome["errors"].append(f"{type(exc).__name__}: {exc}")
                outcome["rows"] = [
                    {
                        "index": index,
                        "source": row["metadata"],
                        "prompt": prompt,
                        "expected": row["output"],
                        "output": None,
                        "status": "failed",
                        "correct": False if protocol.scorer != "open_review" else None,
                        "field_scores": {},
                        "truncated": False,
                        "generated_tokens": None,
                        "error": outcome["errors"][-1],
                    }
                    for index, (row, prompt) in enumerate(zip(records, prompts))
                ]
            finally:
                if runtime is not None:
                    try:
                        runtime.close()
                    except Exception as exc:
                        outcome["errors"].append(f"模型释放失败：{exc}")
                runtime = None
                gc.collect()
            if scoring_recipe is not None:
                for row in outcome["rows"]:
                    row.update(
                        business_score=0.0, scoring_reason=row.get("error") or "尚未完成业务评分。"
                    )
                try:
                    scored = custom_score_observations(scoring_recipe, outcome["rows"], records)
                    for row in outcome["rows"]:
                        if row["index"] in scored:
                            observation = scored[row["index"]]
                            row.update(
                                business_score=observation["score"],
                                scoring_reason=observation["reason"],
                                correct=observation["score"] >= scoring_recipe.pass_threshold,
                                status="scored",
                            )
                except (ValueError, OSError, TypeError) as exc:
                    for row in outcome["rows"]:
                        if row["status"] == "awaiting_custom_score":
                            row.update(
                                status="failed",
                                correct=False,
                                error=f"自定义评分失败：{exc}",
                                scoring_reason=f"自定义评分失败：{exc}",
                            )
            total = len(records)
            valid = sum(row["status"] == "scored" for row in outcome["rows"])
            generated = sum(row["output"] is not None for row in outcome["rows"])
            outcome["metrics"] = {
                "total": total,
                "scored": valid,
                "generated": generated,
                "scorable_coverage": valid / total,
                "generation_coverage": generated / total,
                "exact_match": sum(row["correct"] is True for row in outcome["rows"]) / total
                if protocol.scorer not in {"open_review", "custom_rules"}
                else None,
                "field_accuracy": {
                    name: sum(row["field_scores"].get(name, False) for row in outcome["rows"])
                    / total
                    for name in protocol.fields
                },
            }
            if scoring_recipe is not None:
                outcome["metrics"].update(
                    business_score=sum(row["business_score"] for row in outcome["rows"]) / total,
                    pass_rate=sum(row["correct"] is True for row in outcome["rows"]) / total,
                )
            report.models.append(outcome)
            self._save(report)
            if any(error.startswith("模型释放失败") for error in outcome["errors"]):
                report.status = "release_failed"
                self._save(report)
                return report  # Do not load another model while release is uncertain.
        report.status = (
            "needs_business_review"
            if purpose == "final_acceptance" and protocol.scorer == "open_review"
            else "completed_with_failures"
            if any(
                row["status"] in {"failed", "truncated"}
                for model in report.models
                for row in model["rows"]
            )
            else "completed"
        )
        self._save(report)
        return report
