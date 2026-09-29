"""Grounded, paginated development-set evidence for an Agent's next diagnosis."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.workbench.business_evaluation import (
    EvaluationProtocol,
    EvaluationReport,
    _model_identity,
    _score,
    _strict_object,
    comparison_key,
)
from src.workbench.intake_models import DatasetArtifact, IntakeSession
from src.workbench.materialize import dataset_is_current
from src.workbench.sources import canonical, content_digest

ERROR_TYPES = (
    "correct",
    "mismatch",
    "truncated",
    "generation_error",
    "invalid_output",
    "scoring_error",
    "needs_business_review",
)

# 指令回声判定：输出与提示共享≥12字连续片段即视为复述（改述式复述也命中）；
# 短标签（如「上涨」）与短语引用的共享长度远低于此阈值。
ECHO_MIN_SHARED_CHARS = 12
ECHO_MIN_OUTPUT_CHARS = 12


def output_echoes_prompt(output: object, prompt: object) -> bool:
    """输出是否在复述提示文本（逐字前缀或改述式长片段共享）。"""

    def longest_common_span(left: str, right: str) -> int:
        best = 0
        previous = [0] * (len(right) + 1)
        for i in range(1, len(left) + 1):
            current = [0] * (len(right) + 1)
            for j in range(1, len(right) + 1):
                if left[i - 1] == right[j - 1]:
                    current[j] = previous[j - 1] + 1
                    if current[j] > best:
                        best = current[j]
            previous = current
        return best

    if not isinstance(output, str) or not isinstance(prompt, str):
        return False
    if len(output) < ECHO_MIN_OUTPUT_CHARS:
        return False
    # 提示可能远长于输出；先用输出片段探测，避免整段动态规划无谓开销。
    window = output[: max(64, len(output))]
    return longest_common_span(window, prompt) >= ECHO_MIN_SHARED_CHARS


def count_instruction_echo(rows: list[dict[str, Any]]) -> int:
    """统计一个模型评测行中复述提示的行数。"""
    return sum(output_echoes_prompt(row.get("output"), row.get("prompt")) for row in rows)


# 截断提示阈值：单个模型的截断行占比达到 20% 即视为高比例，提示核查 max_new_tokens；
# 与回声提示同构——只陈述观察事实与核查方向，不认定原因（触及上限不等于只需加长）。
TRUNCATION_HINT_RATIO = 0.2


def high_truncation_models(models: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """截断占比达到提示阈值的（模型标签, 截断题数）列表，供对照区与语言化摘要共用口径。"""
    hints = []
    for model in models:
        rows = model.get("rows") or []
        if not rows:
            continue
        truncated = sum(row.get("status") == "truncated" for row in rows)
        if truncated / len(rows) >= TRUNCATION_HINT_RATIO:
            hints.append((model["label"], truncated))
    return hints


# 输出坍缩提示阈值：单个模型的非空输出中，同一内容占比 ≥80% 且非空输出至少 4 条才提示。
# 与回声/截断提示同构——只陈述观察事实与核查方向，不认定原因：答案分布本身集中的任务
# （如恒定目标或严重不均衡）里，模型复述多数类也会呈现同一形态，提示须指向对照答案分布。
DOMINANT_OUTPUT_RATIO = 0.8
DOMINANT_MIN_OUTPUTS = 4


def dominant_output_models(
    models: list[dict[str, Any]],
) -> list[tuple[str, int, int, str]]:
    """输出高度重复的（模型标签, 该内容条数, 非空输出条数, 重复内容）列表。

    None/非字符串输出不计入分母（生成失败没有输出，不参与占比）；供对照区
    警告与语言化摘要共用口径。
    """
    collapsed = []
    for model in models:
        rows = model.get("rows") or []
        outputs = [row.get("output") for row in rows if isinstance(row.get("output"), str)]
        if len(outputs) < DOMINANT_MIN_OUTPUTS:
            continue
        top_output, top_count = Counter(outputs).most_common(1)[0]
        if top_count / len(outputs) >= DOMINANT_OUTPUT_RATIO:
            collapsed.append((model["label"], top_count, len(outputs), top_output))
    return collapsed


# 回声分流长度比：回声题输入平均长度达到其余题 1.5 倍时，才提示「过长内容
# 淹没答案信号」方向——比例是粗略启发，行文同时给出两个平均值供人自行核对。
ECHO_TRIAGE_LENGTH_RATIO = 1.5


def echo_triage_lines(
    label: str, rows: list[dict[str, Any]], protocol: dict[str, Any] | None = None
) -> list[str]:
    """回声预警的观察事实分流（单一来源，页面警告与对照摘要同词汇）。

    三个核查方向里，「max_new_tokens 过小」与「过长内容淹没答案信号」能用
    报告内可观察事实分流：回声题是否同时截断、回声题输入是否更长。「模板
    不匹配」依赖基座是否对话型，报告内没有该证据，只陈述已知模板事实。
    只给观察与核查顺序，不认定原因；改一项后须同题复测。
    """
    flags = [output_echoes_prompt(row.get("output"), row.get("prompt")) for row in rows]
    echo_rows = [row for row, flag in zip(rows, flags) if flag]
    if not echo_rows:
        return []
    protocol = protocol or {}
    limit = protocol.get("max_new_tokens")
    limit_note = f"（当前 max_new_tokens={limit}）" if limit is not None else ""
    overlap = sum(row.get("status") == "truncated" for row in echo_rows)
    lines: list[str] = []
    if overlap:
        lines.append(
            f"检测到指令回声——{label} 有 {len(echo_rows)} 题输出在复述提示而非作答，"
            f"其中 {overlap} 题同时触及生成长度上限{limit_note}：优先核查 max_new_tokens "
            "是否小于最短合法答案——模型可能把生成额度先花在了复述指令上，加长后同题复测。"
        )
    else:
        lines.append(
            f"检测到指令回声——{label} 有 {len(echo_rows)} 题输出在复述提示而非作答，"
            f"且没有一题触及生成长度上限{limit_note}：长度上限不是第一嫌疑，"
            "优先核查输入长度与提示模板两个方向。"
        )
    rest_rows = [row for row, flag in zip(rows, flags) if not flag]
    if rest_rows:
        echo_mean = sum(len(row.get("prompt") or "") for row in echo_rows) / len(echo_rows)
        rest_mean = sum(len(row.get("prompt") or "") for row in rest_rows) / len(rest_rows)
        if echo_mean >= rest_mean * ECHO_TRIAGE_LENGTH_RATIO:
            lines.append(
                f"回声题的模型输入平均 {echo_mean:.0f} 字符，其余题平均 {rest_mean:.0f} 字符——"
                "输入更长的题更易回声，支持「过长内容淹没答案信号」方向："
                "压缩指令或精简字段呈现后同题复测。"
            )
        else:
            lines.append(
                f"回声题与其余题的输入长度相近（平均 {echo_mean:.0f} vs {rest_mean:.0f} 字符）——"
                "「内容过长」方向证据不足，不作为优先核查项。"
            )
    renderer = protocol.get("prompt_renderer")
    if renderer == "render_alpaca_prompt_without_answer":
        lines.append(
            "本次评测使用补全式（Alpaca）提示模板；对话型基座（Instruct/Chat 类）与补全模板"
            "不匹配是回声的已知形态——若基座为对话型，改用 messages 对话格式后同题复测。"
        )
    else:
        lines.append(
            "「提示模板与基座不匹配」方向仍需人工核查：报告未记录本次模板类型，"
            "无法用报告内事实分流。"
        )
    lines.append("以上是按报告内事实排出的核查顺序，不认定原因；每改一项后用同一题集复测一次。")
    return lines


class EvaluationDiagnostics:
    def __init__(self, report: EvaluationReport, session: IntakeSession):
        if not dataset_is_current(session):
            raise ValueError("诊断必须关联这份评测所用的已确认数据版本；当前任务快照已失效。")
        report, session = deepcopy(report), session.model_copy(deep=True)
        self.report, self.session = report, session
        if (
            report.status not in {"completed", "completed_with_failures", "release_failed"}
            or not report.models
        ):
            raise ValueError("评测尚未产生完整模型结果，不能把未完成报告当作坏例诊断。")
        artifact = session.dataset
        expected_identity = {
            "name": artifact.name,
            "version": artifact.version,
            "source_digest": artifact.source_digest,
            "recipe_digest": artifact.recipe_digest,
            "split": "validation",
            "purpose": "development_only",
        }
        if any(report.dataset.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(
                "报告与当前确认的数据版本或开发集边界不一致；不能拿独立测试集作迭代诊断。"
            )
        if report.comparison_key != comparison_key(report.dataset, report.protocol):
            raise ValueError("评测的数据/协议指纹不一致。")
        protocol = EvaluationProtocol(
            scorer=report.protocol["scorer"],
            fields=tuple(report.protocol.get("fields", [])),
            max_new_tokens=report.protocol["max_new_tokens"],
            strip_whitespace=report.protocol["strip_whitespace"],
            custom_scoring=report.protocol.get("custom_scoring"),
        )
        protocol.validate()
        from src.workbench.business_evaluation import (
            confirmed_scoring_identity,
            custom_score_observations,
        )

        scoring_recipe, scoring_identity = confirmed_scoring_identity(protocol, session)
        if any(report.protocol.get(key) != value for key, value in scoring_identity.items()):
            raise ValueError("报告自定义评分规则与已确认完整规则内容不一致。")
        if (
            report.protocol.get("generation") != "greedy_v1"
            or report.protocol.get("prompt_renderer") != "render_alpaca_prompt_without_answer"
        ):
            raise ValueError("当前诊断不支持这份生成或提示协议。")
        from src.data.loaders import render_alpaca_prompt
        from src.data_flywheel.dataset_registry import LocalDatasetRegistry
        from src.workbench.training_preflight import _verified_partitions

        issues = {"issues": []}
        records = _verified_partitions(session, issues)["validation"]
        if any(issue["severity"] == "blocking" for issue in issues["issues"]):
            raise ValueError("开发集分区存在泄漏，不能继续归因或迭代。")
        if report.dataset.get("evaluation_suite"):
            from src.workbench.evaluation_suites import evaluation_cases

            selected = evaluation_cases(session, split="validation")
            if report.dataset["evaluation_suite"] != getattr(
                session.dataset, "evaluation_suite", None
            ):
                raise ValueError("报告与当前固定评测套件身份不一致。")
            if report.protocol.get("evaluation_key") != selected["evaluation_key"]:
                raise ValueError("报告固定开发题身份与当前分区不一致。")
            records = selected["records"]
            if report.protocol.get("answers_digest") != content_digest(
                [row["output"] for row in records]
            ):
                raise ValueError("报告标准答案与固定评测套件内容不一致。")
        manifest = LocalDatasetRegistry(artifact.registry_root).get_split_manifest(
            artifact.name, artifact.version
        )
        prompts = [render_alpaca_prompt({**row, "output": ""}) for row in records]
        if (
            report.dataset.get("split_sha256") != manifest["splits"]["validation"]["sha256"]
            or report.dataset.get("row_count") != len(records)
            or report.protocol.get("prompt_digest") != content_digest(prompts)
        ):
            raise ValueError("报告中的开发集内容或提示指纹与实际分区不一致。")
        self.report_digest = content_digest(asdict(report))
        self._cases = []
        self._counts = {}
        source_lookup = {}
        raw_sources = session.full_data.sources or {"main": session.full_data.source}
        for alias, source in raw_sources.items():
            for row in source.rows:
                source_lookup[(source.digest, row.row_id)] = {
                    "alias": alias,
                    "name": source.name,
                    "source_digest": source.digest,
                    "row_id": row.row_id,
                    "line": row.line,
                    "values": row.values,
                }
        processed = {row.row_id: row for row in session.full_data.source.rows}
        labels = [model["label"] for model in report.models]
        if len(set(labels)) != len(labels):
            raise ValueError("报告模型名称重复，不能唯一定位证据。")
        for model in report.models:
            rows = model["rows"]
            indices = [row.get("index") for row in rows]
            if (
                len(rows) != len(records)
                or any(type(index) is not int for index in indices)
                or set(indices) != set(range(len(records)))
            ):
                raise ValueError("评测行缺失、重复或索引不正确；不能删除失败行后生成诊断。")
            counts = dict.fromkeys(ERROR_TYPES, 0)
            rescored = (
                custom_score_observations(scoring_recipe, rows, records) if scoring_recipe else {}
            )
            for row in sorted(rows, key=lambda item: item["index"]):
                index = row["index"]
                expected = records[index]
                if (
                    row.get("expected") != expected["output"]
                    or row.get("prompt") != prompts[index]
                    or canonical(row.get("source")) != canonical(expected["metadata"])
                ):
                    raise ValueError("评测行的来源、期望或实际提示与开发集不匹配。")
                if scoring_recipe is not None:
                    score, reason = row.get("business_score"), row.get("scoring_reason")
                    if (
                        type(score) not in (int, float)
                        or not math.isfinite(score)
                        or not 0 <= score <= 1
                        or not isinstance(reason, str)
                        or not reason.strip()
                    ):
                        raise ValueError("报告业务分数或评分理由无效。")
                    if row["status"] == "scored":
                        observed = rescored.get(index)
                        if (
                            observed != {"score": score, "reason": reason}
                            or row.get("correct") is not (score >= scoring_recipe.pass_threshold)
                            or row.get("field_scores") != {}
                        ):
                            raise ValueError("保存的自定义业务分数与隔离重算结果不一致。")
                    elif score != 0:
                        raise ValueError("生成失败、截断或评分失败必须按业务零分保留。")
                kind = self._classify(row, protocol)
                counts[kind] += 1
                echo = output_echoes_prompt(row.get("output"), row.get("prompt"))
                if echo:
                    counts["instruction_echo"] = counts.get("instruction_echo", 0) + 1
                if kind == "correct":
                    continue
                origins = expected["metadata"].get("origins") or [
                    {
                        "source_digest": expected["metadata"]["source_digest"],
                        "row_id": expected["metadata"]["source_row_id"],
                    }
                ]
                original_rows = []
                for origin in origins:
                    original = source_lookup.get((origin["source_digest"], origin["row_id"]))
                    if original is None:
                        raise ValueError("坏例无法定位到对应的原始全量资料，不能编造原数据证据。")
                    original_rows.append(original)
                case = {
                    "evidence_id": f"{model['label']}:{index}",
                    "model": model["label"],
                    "index": index,
                    "error_type": kind,
                    "task": session.analysis.task.model_dump(),
                    "processing_plan": {
                        "recipe": session.analysis.recipe.model_dump(),
                        "composition": session.analysis.composition,
                        "adapter": session.analysis.adapter,
                    },
                    "original_rows": original_rows,
                    "processed_record": processed[
                        expected["metadata"]["source_row_id"]
                    ].model_dump(),
                    "source": row["source"],
                    "prompt": row["prompt"],
                    "expected": row["expected"],
                    "output": row["output"],
                    "status": row["status"],
                    "truncated": row["truncated"],
                    "generated_tokens": row["generated_tokens"],
                    "error": row["error"],
                    "correct": row["correct"],
                    "field_scores": row["field_scores"],
                    "echoes_prompt": echo,
                    "evidence_scope": "observed_development_evaluation",
                }
                if scoring_recipe is not None:
                    case.update(
                        business_score=row["business_score"], scoring_reason=row["scoring_reason"]
                    )
                self._cases.append(case)
            if scoring_recipe is not None:
                expected_metrics = {
                    "business_score": sum(row["business_score"] for row in rows) / len(rows),
                    "pass_rate": sum(row["correct"] is True for row in rows) / len(rows),
                    "exact_match": None,
                }
                if any(
                    model["metrics"].get(key) != value for key, value in expected_metrics.items()
                ):
                    raise ValueError("自定义业务指标与全量评测行不一致。")
            self._counts[model["label"]] = {
                "total": len(records),
                "counts": counts,
                "model_errors": model.get("errors", []),
            }

    def training_evidence(self) -> dict[str, Any]:
        """Read local training receipts bound to the evaluated adapter, never raw train/test rows."""
        models = []
        for model in self.report.models:
            entry = {"model": model["label"]}
            if not model.get("requested_model", {}).get("adapter_path"):
                entry.update(status="not_applicable", reason="该评测对象没有微调适配器。")
            else:
                try:
                    entry.update(self._training_evidence(model))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    entry.update(status="invalid", reason=str(exc))
            models.append(entry)
        return {
            "evaluation_id": self.report.evaluation_id,
            "report_digest": self.report_digest,
            "models": models,
            "boundaries": [
                "仅返回配置、训练指标和预检汇总，不发送训练集或独立测试集原文。",
                "训练损失与预检事实不能替代开发集业务评分，也不能证明业务目标达标。",
            ],
        }

    def _training_evidence(self, model: dict[str, Any]) -> dict[str, Any]:
        return inspect_model_training_evidence(model, self.session, self.report.dataset)

    @staticmethod
    def _classify(row: dict[str, Any], protocol: EvaluationProtocol) -> str:
        status, output = row.get("status"), row.get("output")
        if status == "truncated":
            if (
                not row.get("truncated")
                or not isinstance(output, str)
                or row.get("correct") is True
            ):
                raise ValueError("截断标记与评测结果不一致。")
            return "truncated"
        if row.get("truncated"):
            raise ValueError("截断输出不能伪装为正常评分。")
        if status == "failed":
            if row.get("correct") is True:
                raise ValueError("失败输出不能记为正确。")
            if output is None:
                return "generation_error"
            if not isinstance(output, str):
                raise ValueError("评测输出必须保留原始文本。")
            if protocol.scorer == "json_fields_exact":
                try:
                    _strict_object(output)
                except (ValueError, TypeError):
                    return "invalid_output"
            return "scoring_error"
        if not isinstance(output, str):
            raise ValueError("已完成输出缺少真实文本。")
        if (
            status == "needs_business_review"
            and protocol.scorer == "open_review"
            and row.get("correct") is None
        ):
            return "needs_business_review"
        if status != "scored" or protocol.scorer == "open_review":
            raise ValueError("评测行状态与业务评分协议不一致。")
        if protocol.scorer == "custom_rules":
            return "correct" if row["correct"] else "mismatch"
        correct, fields = _score(output, row["expected"], protocol)
        if row.get("correct") is not correct or row.get("field_scores") != fields:
            raise ValueError("保存的业务分数与真实输出不一致。")
        return "correct" if correct else "mismatch"

    def summary(self) -> dict[str, Any]:
        total_counts = Counter()
        for model in self._counts.values():
            total_counts.update(model["counts"])
        facts = [
            {"kind": "observed", "error_type": kind, "count": total_counts[kind]}
            for kind in ERROR_TYPES
        ]
        hypotheses = []
        echo_labels = [
            label
            for label, model in self._counts.items()
            if model["counts"].get("instruction_echo")
        ]
        if echo_labels:
            hypotheses.append(
                {
                    "kind": "needs_verification",
                    "based_on": "instruction_echo",
                    "statement": (
                        "部分输出以提示文本的长前缀开头（复述指令而非作答）。核查方向："
                        "①指令模板过长、答案信号被淹没，考虑压缩指令或调整字段呈现；"
                        "②补全式提示与对话型基座的模板不匹配，考虑改用 messages 对话格式；"
                        "③max_new_tokens 是否小于最短合法答案。回声是观察事实，原因仍待核查。"
                    ),
                    "models": echo_labels,
                }
            )
        if total_counts["truncated"]:
            hypotheses.append(
                {
                    "kind": "needs_verification",
                    "based_on": "truncated",
                    "statement": "触及长度上限不等于只需增加长度；重复生成、停止标记、训练提示与任务真实输出长度均需核查。",
                }
            )
        if total_counts["mismatch"] or total_counts["invalid_output"]:
            hypotheses.append(
                {
                    "kind": "needs_verification",
                    "based_on": "mismatch_or_invalid_output",
                    "statement": "输出与已确认答案或结构不符是事实；标签含义、样本覆盖、数据处理及模型行为谁是原因仍待核查。",
                }
            )
        if total_counts["generation_error"] or total_counts["scoring_error"]:
            hypotheses.append(
                {
                    "kind": "needs_verification",
                    "based_on": "execution_failure",
                    "statement": "先核对所保留的具体执行错误；执行失败本身不能归因为模型能力不足或训练轮数不足。",
                }
            )
        return deepcopy(
            {
                "evaluation_id": self.report.evaluation_id,
                "report_digest": self.report_digest,
                "report_status": self.report.status,
                "comparison_complete": self.report.status
                in {"completed", "completed_with_failures"},
                "dataset": self.report.dataset,
                "protocol": self.report.protocol,
                "models": self._counts,
                "bad_case_count": len(self._cases),
                "error_count": len(self._cases) - total_counts["needs_business_review"],
                "facts": facts,
                "hypotheses": hypotheses,
                "boundaries": [
                    "只使用本次固定开发集；没有读取独立测试集输出或把测试集用于自动选参。",
                    "跨训练版本仅在固定套件、实际提示/答案及评分生成协议一致时可比；无套件的旧报告仍要求相同数据版本。",
                    "诊断提供观察与待核查假设，不自动修改目标、标签、方案或训练轮数。",
                ],
            }
        )

    def read_cases(
        self,
        model: str | None = None,
        error_type: str | None = None,
        offset: int = 0,
        limit: int = 20,
        max_bytes: int = 128 * 1024,
    ) -> dict[str, Any]:
        if model is not None and model not in self._counts:
            raise ValueError("未知评测模型。")
        if error_type is not None and error_type not in ERROR_TYPES[1:]:
            raise ValueError("未知需核查错误类型。")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("分页 offset 必须非负，limit 必须为 1 至 50。")
        if type(max_bytes) is not int or not 4096 <= max_bytes <= 256 * 1024:
            raise ValueError("每页内容边界必须在 4 KiB 至 256 KiB。")
        selected = [
            case
            for case in self._cases
            if (model is None or case["model"] == model)
            and (error_type is None or case["error_type"] == error_type)
        ]
        cases, used = [], 0
        for case in selected[offset : offset + limit]:
            item = {**case, "content_status": "complete"}
            size = len(canonical(item).encode())
            if size > max_bytes:
                item = {
                    "evidence_id": case["evidence_id"],
                    "model": case["model"],
                    "index": case["index"],
                    "error_type": case["error_type"],
                    "content_status": "requires_chunks",
                    "content_bytes": size,
                    "read_with": "read_case_content",
                }
                size = len(canonical(item).encode())
            if cases and used + size > max_bytes:
                break
            cases.append(item)
            used += size
        next_offset = offset + len(cases)
        return deepcopy(
            {
                "evaluation_id": self.report.evaluation_id,
                "report_digest": self.report_digest,
                "total": len(selected),
                "offset": offset,
                "cases": cases,
                "next_offset": next_offset if next_offset < len(selected) else None,
                "has_more": next_offset < len(selected),
            }
        )

    def read_case_content(
        self, evidence_id: str, offset: int = 0, limit: int = 16384
    ) -> dict[str, Any]:
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 16384
        ):
            raise ValueError("内容分块 offset 必须非负，limit 必须为 1 至 16384 个字符。")
        case = next((case for case in self._cases if case["evidence_id"] == evidence_id), None)
        if case is None:
            raise ValueError("找不到这个已定位的开发集坏例。")
        text = canonical(case)
        next_offset = min(offset + limit, len(text))
        return {
            "evidence_id": evidence_id,
            "report_digest": self.report_digest,
            "format": "json",
            "total_characters": len(text),
            "offset": offset,
            "content": text[offset:next_offset],
            "next_offset": next_offset if next_offset < len(text) else None,
            "has_more": next_offset < len(text),
        }


def inspect_model_training_evidence(
    model: dict[str, Any], session: IntakeSession, evaluation_dataset: dict[str, Any]
) -> dict[str, Any]:
    """Verify local training provenance without reading development or final outputs."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def read(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(value, dict), f"训练记录不是对象：{path.name}")
        return value

    def same_path(left: str, right: str | Path) -> bool:
        return Path(left).resolve() == Path(right).resolve()

    identity = model.get("identity") or {}
    recorded_adapter = identity.get("adapter")
    if not recorded_adapter:
        return {"status": "not_available", "reason": "评测未记录可验证的适配器身份。"}
    adapter = Path(recorded_adapter["directory"]).resolve()
    require(
        same_path(model["requested_model"]["adapter_path"], adapter),
        "评测请求与已记录的适配器目录不一致。",
    )
    current_adapter = _model_identity(adapter)
    require(
        current_adapter["content_digest"] == recorded_adapter["content_digest"]
        and current_adapter["files"] == recorded_adapter["files"],
        "适配器或其工作台清单在评测之后已改变，不能绑定当前训练证据。",
    )
    manifest_path = adapter / "workbench_manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_available", "reason": "该适配器未提供工作台训练清单。"}
    run_dir = adapter.parent
    if not all((run_dir / name).is_file() for name in ("run.json", "result.json", "config.json")):
        return {
            "status": "not_available",
            "reason": "工作台训练回执未保留在适配器的所属运行目录。",
        }
    manifest = read(manifest_path)
    run, result, config = (
        read(run_dir / name) for name in ("run.json", "result.json", "config.json")
    )
    run_id = run["run_id"]
    require(
        bool(re.fullmatch(r"wb-[0-9a-f]{32}", run_id))
        and run_dir.name == run_id
        and manifest["run_id"] == result["run_id"] == run_id,
        "适配器清单与训练运行身份不一致。",
    )
    require(result["status"] == "succeeded", "训练没有成功产物回执。")
    training_session = session
    snapshot_path = run_dir / "session.json"
    if snapshot_path.is_file():
        training_session = IntakeSession.model_validate_json(snapshot_path.read_text())
        require(dataset_is_current(training_session), "训练时保存的已确认任务快照已失效。")
    elif run["dataset_version"] != session.dataset.version:
        return {
            "status": "not_available",
            "reason": "历史训练未保留自身任务快照，无法核验跨版本关联。",
        }
    training_dataset = training_session.dataset
    require(
        run["session_id"] == training_session.session_id
        and DatasetArtifact.model_validate(run["dataset"]).model_dump()
        == training_dataset.model_dump()
        and run["dataset_version"] == manifest["dataset_version"] == training_dataset.version,
        "训练运行、训练时快照与产物数据版本不一致。",
    )
    if evaluation_dataset.get("evaluation_suite"):
        from src.workbench.evaluation_suites import verify_training_membership

        verify_training_membership(session.dataset.evaluation_suite, training_session)
    else:
        require(
            training_session.session_id == session.session_id
            and training_dataset.model_dump() == session.dataset.model_dump(),
            "无固定评测套件的历史报告不能跨训练数据版本关联。",
        )
    require(
        config == run["config"]
        and content_digest(config) == run["config_digest"] == manifest["config_digest"],
        "训练配置与产物清单指纹不一致。",
    )
    require(
        same_path(run["output_dir"], adapter)
        and same_path(config["training"]["output_dir"], adapter),
        "训练输出目录与评测适配器不一致。",
    )
    data = config["data"]
    require(
        same_path(data["train_file"], training_dataset.paths["train"])
        and same_path(data["validation_file"], training_dataset.paths["validation"])
        and data.get("validation_split") == 0,
        "训练配置未沿用已确认训练/开发分区。",
    )
    base = identity["base"]
    training_base = run["model_identity"]
    require(
        training_base == manifest["model_identity"]
        and all(
            same_path(path, base["directory"])
            for path in (run["model_path"], config["model"]["name"], training_base["path"])
        )
        and all(
            base["files"].get(name) == digest
            for name, digest in training_base["config_and_tokenizer_hashes"].items()
        )
        and all(
            base["files"].get(weight["name"]) == weight["sha256"]
            for weight in training_base["weights"]
            if "sha256" in weight
        ),
        "训练基座身份与评测记录不一致。",
    )
    expected_files = {
        "adapter_config": {"adapter_config.json"},
        "adapter_weights": {"adapter_model.safetensors", "adapter_model.bin"},
        "tokenizer_config": {"tokenizer_config.json"},
        "metrics": {"workbench_metrics.json"},
    }
    for key, names in expected_files.items():
        artifact = manifest["files"][key]
        path = Path(artifact["path"]).resolve()
        require(
            path.parent == adapter
            and path.name in names
            and same_path(result["artifacts"][key], path)
            and current_adapter["files"].get(path.name) == artifact["sha256"],
            f"训练产物 {key} 的路径或内容身份不一致。",
        )
    require(same_path(result["artifacts"]["manifest"], manifest_path), "训练清单路径不一致。")
    metrics = read(adapter / "workbench_metrics.json")
    require(metrics == result["metrics"], "训练指标与成功回执不一致。")
    preflight = run["preflight"]
    require(
        preflight["version"] == training_dataset.version
        and preflight["source_digest"] == training_dataset.source_digest
        and preflight["recipe_digest"] == training_dataset.recipe_digest
        and preflight["full_confirmed_revision"] == training_dataset.full_confirmed_revision
        and preflight["max_length"] == config["model"]["max_length"]
        and preflight["status"] in {"passed", "warnings"},
        "训练预检与已确认数据或训练长度不一致。",
    )
    logging = config.get("logging", {})
    for key in ("mlflow_run_id", "mlflow_tracking_uri"):
        require(
            not manifest.get(key) or manifest[key] == result.get(key),
            "MLflow 关联与成功回执不一致。",
        )
    return {
        "status": "available",
        "training_run_id": run_id,
        "training_dataset": training_dataset.model_dump(),
        "training_session_digest": content_digest(training_session.model_dump()),
        "config": config,
        "metrics": metrics,
        "preflight": {
            key: preflight[key]
            for key in (
                "status",
                "max_length",
                "supervision_strategy",
                "tokenizer",
                "splits",
                "scope_note",
                "issues",
            )
            if key in preflight
        },
        "mlflow": {
            "run_id": result.get("mlflow_run_id"),
            "status": result.get("mlflow_status")
            or ("recorded" if result.get("mlflow_run_id") else "not_recorded"),
            "tracking_uri": result.get("mlflow_tracking_uri") or logging.get("mlflow_tracking_uri"),
            "experiment_name": logging.get("mlflow_experiment_name"),
            "run_name": logging.get("mlflow_run_name"),
        },
        "identity": {
            "evaluation": identity,
            "training_base": training_base,
            "base_weight_verification": (
                "content_verified"
                if training_base["weights"]
                and all(weight.get("sha256") for weight in training_base["weights"])
                else "legacy_identity_unverified"
            ),
        },
        "facts": ["适配器与清单内容匹配评测时身份；训练配置、数据版本及成功回执关联已核验。"],
    }
