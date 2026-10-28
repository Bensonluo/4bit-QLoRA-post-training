"""Read two recorded R148 studies without loading models or creating stores."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from ui.components.format import finite_number

SOURCE = "sft-answer-supervision-live-2026-10-22.json"
ARCHIVE = "sft-answer-supervision-evidence-2026-10-22.tar.gz"
PREFIX = "r148-review-evidence/"
PROTOCOLS = ("alpaca_full_prompt_v1", "alpaca_answer_only_v1", "native_chat_answer_only_v1")
LOGS = {protocol: f"outputs/validation/r148/v2/training-{protocol}.json" for protocol in PROTOCOLS}
STATE = "outputs/validation/r148/product-sft/adapter/trainer_state.json"
LABELS = {
    PROTOCOLS[0]: "Alpaca · 全提示监督",
    PROTOCOLS[1]: "Alpaca · 仅答案监督",
    PROTOCOLS[2]: "原生对话 · 仅答案监督（实验）",
}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _history(points: list[dict], rate: float) -> list[dict]:
    _require(type(rate) in (int, float) and finite_number(rate) and rate > 0, "学习率记录无效。")
    history = [
        {
            "step": point["step"],
            "loss": point["loss"],
            "learning_rate": point.get("learning_rate", point.get("lr")),
        }
        for point in points
        if "loss" in point
    ]
    _require(
        len(history) == 64
        and all(type(point["step"]) is int for point in history)
        and [point["step"] for point in history] == list(range(1, 65))
        and all(
            type(point["loss"]) in (int, float)
            and finite_number(point["loss"])
            and point["loss"] >= 0
            and point["learning_rate"] == rate
            for point in history
        ),
        "64 次真实训练更新或学习率日志不完整。",
    )
    return history


def _raw_score(row: dict, labels: set[str]) -> tuple[bool, bool]:
    """The original R148 judge: stripped class code or unique-key JSON object."""
    if row["form"] == "classification":
        return row["output"].strip() == row["expected"], row["output"].strip() in labels

    def unique(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON 重复键")
            result[key] = value
        return result

    try:
        value = json.loads(row["output"], object_pairs_hook=unique)
    except (ValueError, TypeError):
        return False, False
    valid = (
        isinstance(value, dict)
        and set(value) == {"route", "ticket"}
        and all(type(item) is str for item in value.values())
    )
    return valid and value == json.loads(row["expected"]), valid


def _verify_scores(rows: list[dict], summary: dict) -> None:
    _require(
        len(rows) == 16 and len({row["case_id"] for row in rows}) == 16, "16 道开发题缺失或重复。"
    )
    labels = {row["expected"] for row in rows if row["form"] == "classification"}
    for row in rows:
        _require(
            row["form"] in {"classification", "json"}
            and all(isinstance(row[key], str) for key in ("prompt", "output", "expected")),
            "开发题的原始文本或类型缺失。",
        )
        proof = row.get("token_proof")
        ids = proof["generated_token_ids"] if proof else row["generated_ids"]
        eos = proof["eos_token_ids"] if proof else [row["eos_token_id"]]
        finish = proof["finish_reason"] if proof else row["finish_reason"]
        _require(
            isinstance(ids, list)
            and ids
            and isinstance(eos, list)
            and eos
            and all(type(token) is int and token >= 0 for token in [*ids, *eos])
            and row["generated_tokens"] == len(ids),
            "原始生成 token 或长度证据不完整。",
        )
        ended = ids[-1] in eos
        _require(
            row["eos_observed"] is ended
            and row["truncated"] is (not ended)
            and finish == ("eos" if ended else "length"),
            "原始 token 的 EOS 或停止原因与已记录状态矛盾。",
        )
        matches, valid = _raw_score(row, labels)
        _require(
            all(
                type(row[key]) is bool
                for key in (
                    "raw_output_matches",
                    "format_valid",
                    "eos_observed",
                    "truncated",
                    "accepted",
                )
            ),
            "原始评分状态不是布尔值。",
        )
        accepted = matches and row["format_valid"] and row["eos_observed"] and not row["truncated"]
        _require(
            row["raw_output_matches"] == matches
            and row["format_valid"] == valid
            and row["accepted"] == accepted,
            "原始输出与已记录的逐题得分矛盾。",
        )
    for form in ("classification", "json", "all"):
        selected = rows if form == "all" else [row for row in rows if row["form"] == form]
        expected = {
            "total": len(selected),
            "accepted": sum(row["accepted"] for row in selected),
            "truncated": sum(row["truncated"] for row in selected),
            "format_valid": sum(row["format_valid"] for row in selected),
        }
        _require(
            len(selected) == (16 if form == "all" else 8)
            and all(summary[form][key] == value for key, value in expected.items()),
            "汇总分数与完整开发题记录不一致。",
        )
        if "acceptance_rate" in summary[form]:
            _require(
                summary[form]["acceptance_rate"] == expected["accepted"] / expected["total"],
                "接受率与实际题目分母不一致。",
            )


def _read_logs(root: Path, data: dict, references: dict[str, dict]) -> tuple[dict[str, Any], dict]:
    """Read only the exact manifest and requested log members, never extract()."""
    archive_info = data["detailed_evidence_archive"]
    _require(archive_info["path"] == ARCHIVE, "公开归档文件名不受支持。")
    raw = (root / "docs/validation" / ARCHIVE).read_bytes()
    _require(
        len(raw) == archive_info["bytes"] and _sha(raw) == archive_info["sha256"],
        "公开归档大小或 SHA256 与证据不一致。",
    )
    members, values = [], {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as bundle:
        names = [member.name for member in bundle.getmembers()]

        def read_member(relative: str) -> bytes:
            full = PREFIX + relative
            _require(names.count(full) == 1, "归档中所需日志缺失或名称重复。")
            member = bundle.getmember(full)
            _require(
                member.isfile() and 0 < member.size <= 100_000,
                "所需归档成员不是有效的小型日志文件。",
            )
            stream = bundle.extractfile(member)
            _require(stream is not None, "所需归档日志无法读取。")
            assert stream is not None
            content = stream.read(member.size + 1)
            _require(len(content) == member.size, "归档成员实际大小不一致。")
            members.append({"name": relative, "size": len(content), "sha256": _sha(content)})
            return content

        manifest = json.loads(read_member("manifest.json"))
        _require(
            isinstance(manifest, dict) and len(manifest) == archive_info["manifest_entries"],
            "归档清单条目数不一致。",
        )
        for relative, reference in references.items():
            _require(
                relative in {*LOGS.values(), STATE}
                and reference["archive_relative_path"] == relative,
                "拒绝读取未列入白名单的归档成员。",
            )
            content = read_member(relative)
            expected = manifest[relative]
            _require(
                len(content) == expected["size"]
                and _sha(content) == expected["sha256"] == reference["sanitized_sha256"],
                "训练日志与归档清单或源成员摘要不一致。",
            )
            values[relative] = json.loads(content)
    return values, {
        "name": ARCHIVE,
        "sha256": archive_info["sha256"],
        "bytes": archive_info["bytes"],
        "members": members,
    }


def _variant(
    variant_id: str,
    label: str,
    protocol: str,
    evaluation: dict,
    train_precision: str | None,
    eval_precision: str,
    history: list[dict],
) -> dict:
    _verify_scores(evaluation["rows"], evaluation["summary"])
    return {
        "id": variant_id,
        "label": label,
        "role": "baseline" if train_precision is None else "trained",
        "protocol": protocol,
        "training_precision": train_precision,
        "evaluation_precision": eval_precision,
        "precision": (
            f"训练 {train_precision} / 推理 {eval_precision}"
            if train_precision
            else f"推理 {eval_precision}"
        ),
        "training_history": history,
        "rows": evaluation["rows"],
        "summary": evaluation["summary"],
        "baseline_id": None,
    }


def _bind_baselines(variants: list[dict]) -> None:
    def prompts(variant: dict) -> dict:
        return {(row["case_id"], row["form"]): row["prompt"] for row in variant["rows"]}

    questions = None
    for variant in variants:
        current = {(row["case_id"], row["form"]): row["expected"] for row in variant["rows"]}
        _require(questions is None or questions == current, "各版本的题目身份或标准答案不同。")
        questions = current
        if variant["role"] == "trained":
            matches = [
                base["id"]
                for base in variants
                if base["role"] == "baseline" and prompts(base) == prompts(variant)
            ]
            _require(len(matches) == 1, "训练版本没有唯一匹配其真实提示的基座记录。")
            variant["baseline_id"] = matches[0]


def _verify_case_prompts(cases: list[dict], variants: list[dict]) -> None:
    """Bind the displayed business text to every recorded generation prompt."""
    from src.data.sft_protocol import render_sft_prompt

    by_id = {(case["case_id"], case["form"]): case for case in cases}
    for variant in variants:
        protocol = variant["protocol"]
        for row in variant["rows"]:
            case = by_id[(row["case_id"], row["form"])]
            if protocol == PROTOCOLS[2]:
                # R148 records Qwen's actual non-thinking generation wrapper;
                # no tokenizer or regenerated model input is needed to verify it.
                content = case["instruction"] + ("\n\n" + case["input"] if case["input"] else "")
                prompt = (
                    "<|im_start|>user\n"
                    + content
                    + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
            else:
                prompt = render_sft_prompt(case, None, protocol)
            _require(row["prompt"] == prompt, "业务输入或任务要求与实际模型提示不一致。")


def load_study(root: Path, study_id: str) -> dict:
    """Return one immutable study view, not a fabricated BusinessEvaluationReport."""
    if study_id not in {"r148-controlled-v2", "r148-product-smoke"}:
        raise ValueError("未收录这份公开历史研究。")
    try:
        root = Path(root)
        raw = (root / "docs/validation" / SOURCE).read_bytes()
        data = json.loads(raw)
        if study_id == "r148-controlled-v2":
            controlled = data["controlled_v2"]
            settings, results = controlled["settings"], controlled["results"]
            _require(
                settings["steps"] == 64 and set(results["arms"]) == set(PROTOCOLS),
                "同次三臂受控实验不完整。",
            )
            references = {
                LOGS[protocol]: results["arms"][protocol]["optimizer_update_evidence"]["full_log"]
                for protocol in PROTOCOLS
            }
            logs, archive = _read_logs(root, data, references)
            variants = [
                _variant(
                    "base-alpaca",
                    "未训练基座 · Alpaca",
                    PROTOCOLS[0],
                    results["baseline"][PROTOCOLS[0]],
                    None,
                    settings["dtype"],
                    [],
                ),
                _variant(
                    "base-native",
                    "未训练基座 · 原生对话",
                    PROTOCOLS[2],
                    results["baseline"][PROTOCOLS[2]],
                    None,
                    settings["dtype"],
                    [],
                ),
            ]
            for protocol in PROTOCOLS:
                arm = results["arms"][protocol]
                _require(arm["protocol_identity"]["protocol"] == protocol, "训练臂协议身份不一致。")
                history = _history(logs[LOGS[protocol]], settings["learning_rate"])
                evidence = arm["optimizer_update_evidence"]
                _require(
                    evidence["count"] == 64
                    and evidence["first_step"] == 1
                    and evidence["last_step"] == 64,
                    "训练更新证据与完整日志不一致。",
                )
                variants.append(
                    _variant(
                        protocol,
                        LABELS[protocol],
                        protocol,
                        arm["evaluation"],
                        settings["dtype"],
                        settings["dtype"],
                        history,
                    )
                )
            title, scope, pair = (
                "监督协议受控对照 · 同次三臂",
                "合成机制实验：同次三臂，不是三轮业务迭代；原生对话仅实验，不代表业务验收。",
                [PROTOCOLS[0], PROTOCOLS[1]],
            )
        else:
            smoke = data["product_smoke"]
            settings, results = smoke["settings"], smoke["results"]
            logs, archive = _read_logs(root, data, {STATE: smoke["trainer_state"]})
            state = logs[STATE]
            _require(
                state["global_step"] == results["training"]["global_step"] == 64,
                "产品训练回执与原始 Trainer 状态不一致。",
            )
            precision = settings["actual_training_dtype"]
            _require(
                precision == "torch.float32"
                and settings["configs"]["model"]["torch_dtype"] == "float32"
                and not settings["configs"]["training"]["fp16"]
                and not settings["configs"]["training"]["bf16"],
                "产品训练与推理精度不能混淆。",
            )
            history = _history(
                state["log_history"], settings["configs"]["training"]["learning_rate"]
            )
            evaluations = results["evaluations"]
            variants = [
                _variant(
                    "base",
                    "未训练基座",
                    PROTOCOLS[1],
                    evaluations["base"],
                    None,
                    evaluations["base"]["actual_dtype"],
                    [],
                ),
                _variant(
                    "trained",
                    "仅答案监督 · 产品模型",
                    PROTOCOLS[1],
                    evaluations["trained"],
                    precision,
                    evaluations["trained"]["actual_dtype"],
                    history,
                ),
            ]
            title, scope, pair = (
                "仅答案监督 · 产品链路验证",
                "独立产品 smoke；训练 FP32、推理 BF16，不是前述受控实验第四臂，也不代表真实业务验收。",
                ["trained", "base"],
            )
        _bind_baselines(variants)
        cases = data["controlled_v2"]["data"]["development"]
        expected = {(case["case_id"], case["form"]): case["output"] for case in cases}
        _require(
            len(cases) == len(expected) == 16
            and all(
                isinstance(case["input"], str) and isinstance(case["instruction"], str)
                for case in cases
            ),
            "共同业务输入缺失或重复。",
        )
        if study_id == "r148-product-smoke":
            _require(
                data["product_smoke"]["data_source"] == "controlled_v2.data",
                "产品 smoke 的业务数据来源不一致。",
            )
        for item in variants:
            _require(
                {(row["case_id"], row["form"]): row["expected"] for row in item["rows"]}
                == expected,
                "业务输入与版本的题目或标准答案不一致。",
            )
        _verify_case_prompts(cases, variants)
        return {
            "id": study_id,
            "title": title,
            "date": data["date"],
            "scope": scope,
            "source": {
                "name": SOURCE,
                "path": f"docs/validation/{SOURCE}",
                "date": data["date"],
                "sha256": _sha(raw),
                "archive": {key: value for key, value in archive.items() if key != "members"},
                "members": archive["members"],
            },
            "limitations": data["limitations"],
            "variants": variants,
            "cases": cases,
            "default_pair": pair,
        }
    except (OSError, KeyError, TypeError, ValueError, tarfile.TarError, EOFError) as exc:
        raise ValueError(f"公开历史研究 {study_id} 无法读取或核验：{exc}") from exc
