"""Immutable business evaluation cases and held-out entity reservations."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from src.workbench.full_data import full_data_is_current
from src.workbench.intake_models import IntakeSession
from src.workbench.sources import canonical, content_digest

EVALUATION_SPLITS = ("validation", "test")
ANCHOR_FIELDS = ("name", "version", "registry_root", "source_digest", "recipe_digest")


def _semantic(record):
    value = {
        "instruction": record["instruction"],
        "input": record["input"],
        "output": record["output"],
        "group": record["metadata"]["group"],
    }
    if "temporal" in record["metadata"]:
        value["temporal"] = record["metadata"]["temporal"]
    return value


def _keys(record, columns):
    return [("input", record["input"])] + [
        (f"group:{column}", canonical(record["metadata"]["group"][column])) for column in columns
    ]


def _reference(root, manifest):
    return {
        "suite_id": manifest["suite_id"],
        "root": str(root),
        "manifest_path": str(root / f"{manifest['suite_id']}.json"),
        "case_counts": manifest["case_counts"],
        "cases_digest": manifest["cases_digest"],
    }


def _case_digest(cases):
    return content_digest(
        {
            split: [{"case_id": case["case_id"], **_semantic(case)} for case in cases[split]]
            for split in EVALUATION_SPLITS
        }
    )


def verify_suite(reference: dict) -> dict:
    suite_id = reference.get("suite_id")
    if not isinstance(suite_id, str) or not re.fullmatch(r"[0-9a-f]{64}", suite_id):
        raise ValueError("无效固定评测套件 ID。")
    root = Path(reference["root"]).resolve()
    path = root / f"{suite_id}.json"
    if Path(reference.get("manifest_path", path)).resolve() != path or path.is_symlink():
        raise ValueError("固定评测套件路径与其身份不一致。")
    try:
        manifest = json.loads(path.read_text())
        identity = {key: value for key, value in manifest.items() if key != "suite_id"}
        if (
            manifest["suite_id"] != suite_id
            or content_digest(identity) != suite_id
            or manifest["format_version"] != 1
        ):
            raise ValueError("固定评测套件内容哈希不匹配；原评分题不能修改。")
        if set(manifest["cases"]) != set(EVALUATION_SPLITS) or any(
            not manifest["cases"][split] for split in EVALUATION_SPLITS
        ):
            raise ValueError("固定评测套件必须同时含非空开发集与最终测试集。")
        if _case_digest(manifest["cases"]) != manifest["cases_digest"]:
            raise ValueError("固定评测题身份不匹配。")
        if set(manifest["anchor_dataset"]) != set(ANCHOR_FIELDS) or manifest["case_counts"] != {
            split: len(manifest["cases"][split]) for split in EVALUATION_SPLITS
        }:
            raise ValueError("固定评测套件的锚定版本或题数无效。")
        identifiers = [case["case_id"] for cases in manifest["cases"].values() for case in cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("固定评测题 ID 必须唯一。")
        expected_ref = _reference(root, manifest)
        if any(
            reference.get(key, expected_ref[key]) != expected_ref[key]
            for key in ("case_counts", "cases_digest")
        ):
            raise ValueError("套件引用的题数或内容摘要与实际套件不一致。")
        return manifest
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("固定评测套件结构无效。") from exc


class EvalSuiteService:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, suite_id: str) -> dict:
        return _reference(self.root, verify_suite({"root": str(self.root), "suite_id": suite_id}))

    def verify(self, reference: dict) -> dict:
        return verify_suite(reference)

    def load(self, reference: dict) -> dict:
        return verify_suite(reference)

    def list_suites(self) -> list[dict]:
        return [
            self.get(path.stem)
            for path in sorted(self.root.glob("*.json"))
            if re.fullmatch(r"[0-9a-f]{64}", path.stem)
        ]

    def assert_compatible(self, session: IntakeSession, reference: dict) -> dict:
        return assert_compatible(session, reference)

    def freeze(self, session: IntakeSession, *, new_suite: bool = False) -> dict:
        from src.workbench.materialize import dataset_is_current
        from src.workbench.training_preflight import _verified_partitions

        if not dataset_is_current(session):
            raise ValueError("请先物化当前确认的全量数据，再固定开发与最终测试题。")
        artifact = session.dataset
        report = {"issues": []}
        splits = _verified_partitions(session, report)
        if report["issues"]:
            raise ValueError("当前分区存在业务对象或输入泄漏，不能冻结为评测套件。")
        if artifact.evaluation_suite and not new_suite:
            assert_compatible(session, artifact.evaluation_suite)
            return _reference(
                Path(artifact.evaluation_suite["root"]).resolve(),
                verify_suite(artifact.evaluation_suite),
            )
        cases = {}
        for split in EVALUATION_SPLITS:
            occurrences = defaultdict(int)
            cases[split] = []
            for record in splits[split]:
                case = copy.deepcopy(record)
                signature = content_digest(_semantic(case))
                occurrence = occurrences[signature]
                occurrences[signature] += 1
                case["case_id"] = content_digest(
                    {"split": split, "content": signature, "occurrence": occurrence}
                )
                case["metadata"]["evaluation_case_id"] = case["case_id"]
                cases[split].append(case)
        group_columns = sorted(session.analysis.recipe.group_columns)
        identity = {
            "format_version": 1,
            "anchor_dataset": {key: getattr(artifact, key) for key in ANCHOR_FIELDS},
            "instruction": session.analysis.recipe.instruction,
            "group_columns": group_columns,
            "cases": cases,
            "case_counts": {split: len(cases[split]) for split in EVALUATION_SPLITS},
            "cases_digest": _case_digest(cases),
            "anchor_train_keys": sorted(
                {key for row in splits["train"] for key in _keys(row, group_columns)}
            ),
            "scope_note": "原开发/最终测试评分题固定；同对象新增行保留在对应分区但不自动扩充评分题。",
        }
        temporal_policy = getattr(session.analysis.recipe, "temporal_split", None)
        if temporal_policy is not None:
            identity["temporal_policy"] = temporal_policy.model_dump()
        manifest = {**identity, "suite_id": content_digest(identity)}
        path = self.root / f"{manifest['suite_id']}.json"
        handle, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.root)
        try:
            with os.fdopen(handle, "w") as output:
                output.write(canonical(manifest))
            try:
                os.link(temporary, path)
            except FileExistsError:
                verify_suite(_reference(self.root, manifest))
        finally:
            Path(temporary).unlink(missing_ok=True)
        return _reference(self.root, manifest)


def assert_compatible(session: IntakeSession, reference: dict) -> dict:
    from src.workbench.materialize import _connected_groups
    from src.workbench.recipes import preview_recipe

    suite = verify_suite(reference)
    report = session.full_data
    recipe = session.analysis.recipe if session.analysis else None
    if (
        not full_data_is_current(session)
        or not report
        or report.status != "confirmed"
        or report.preview is None
        or recipe is None
    ):
        raise ValueError("请先确认本轮全量数据，再与固定评测套件匹配。")
    if (
        recipe.instruction != suite["instruction"]
        or sorted(recipe.group_columns) != suite["group_columns"]
    ):
        raise ValueError("指令或业务分组含义已改变；请建立新套件并统一重新评测，不能沿用旧评分题。")
    if report.preview != preview_recipe(report.source, recipe):
        raise ValueError("当前全量预览与实际方案不一致，请重新验证。")
    temporal_policy = getattr(recipe, "temporal_split", None)
    if suite.get("temporal_policy") != (
        temporal_policy.model_dump() if temporal_policy is not None else None
    ):
        raise ValueError("固定套件的时间策略、边界或观察截止时间已改变，请建立新套件统一重评。")
    rows = report.preview.rows
    temporal = None
    if temporal_policy is not None:
        from src.workbench.temporal_split import temporal_assignment

        temporal = temporal_assignment(rows, temporal_policy, recipe.group_columns)
    records = [
        {
            "instruction": recipe.instruction,
            "input": row.input,
            "output": row.target,
            "metadata": {"group": row.group},
        }
        for row in rows
    ]
    if temporal is not None:
        for row, record in zip(rows, records):
            record["metadata"]["temporal"] = temporal["times_by_row"][row.row_id]
    candidates = defaultdict(list)
    for index, record in enumerate(records):
        candidates[content_digest(_semantic(record))].append(index)
    case_ids_by_row, case_bindings, heldout_keys = {}, {}, defaultdict(set)
    for split in EVALUATION_SPLITS:
        for case in suite["cases"][split]:
            matches = candidates.get(content_digest(_semantic(case)), [])
            if not matches:
                raise ValueError(
                    f"固定{split}题 {case['case_id'][:12]} 缺失，或输入、答案、分组、时间已改变；不能默默替题，请恢复原题或新建套件统一重评。"
                )
            index = matches.pop(0)
            if temporal is not None and index not in temporal["assignments"][split]:
                raise ValueError("固定评分题不再属于原时间分区，不能沿用旧套件。")
            case_ids_by_row[rows[index].row_id] = case["case_id"]
            case_bindings[case["case_id"]] = rows[index].row_id
            for key in _keys(case, suite["group_columns"]):
                heldout_keys[key].add(split)
    assigned = (
        temporal["assignments"]
        if temporal is not None
        else {split: [] for split in ("train", *EVALUATION_SPLITS)}
    )
    group_counts = temporal["group_counts"] if temporal is not None else dict.fromkeys(assigned, 0)
    trained = {tuple(key) for key in suite["anchor_train_keys"]}
    for group in _connected_groups(rows, recipe.group_columns):
        keys = {key for index in group for key in _keys(records[index], suite["group_columns"])}
        destinations = set().union(*(heldout_keys.get(key, set()) for key in keys))
        if len(destinations) > 1:
            raise ValueError(
                "本轮资料把固定开发集与最终测试集连接为同一业务对象，无法维持隔离；请澄清关系或新建套件。"
            )
        if destinations and keys & trained:
            raise ValueError(
                "新增关系把原训练对象连接到固定评测对象，旧模型已有训练暴露；请新建套件统一重评。"
            )
        if temporal is None:
            destination = next(iter(destinations)) if destinations else "train"
            assigned[destination].extend(group)
            group_counts[destination] += 1
    if any(not assigned[split] for split in assigned):
        raise ValueError(
            "固定评测对象隔离后不能形成非空训练/开发/最终测试分区，请补充独立训练数据。"
        )
    reserved = {
        split: [
            rows[index].row_id
            for index in assigned[split]
            if rows[index].row_id not in case_ids_by_row
        ]
        for split in EVALUATION_SPLITS
    }
    assignment = {
        "suite_id": suite["suite_id"],
        "assignments": assigned,
        "group_counts": group_counts,
        "case_ids_by_row": case_ids_by_row,
        "case_bindings": case_bindings,
        "reserved_rows": reserved,
    }
    if temporal is not None:
        assignment.update(
            excluded_rows=temporal["excluded_rows"], times_by_row=temporal["times_by_row"]
        )
    return assignment


def verify_training_membership(reference: dict, training_session: IntakeSession) -> bool:
    from src.workbench.materialize import dataset_is_current
    from src.workbench.training_preflight import _verified_partitions

    suite = verify_suite(reference)
    artifact = training_session.dataset
    if not dataset_is_current(training_session):
        raise ValueError("训练记录不含有效的已确认数据版本。")
    if artifact.evaluation_suite:
        if artifact.evaluation_suite["suite_id"] != suite["suite_id"]:
            raise ValueError("训练记录绑定的是另一套固定评测题。")
    elif any(getattr(artifact, key) != value for key, value in suite["anchor_dataset"].items()):
        raise ValueError("历史训练版本不是此套件的原始锚定版本，不能假定未见过固定评测题。")
    else:
        # An original training snapshot predates suite creation. Bind only its exact
        # anchor copy so case membership and frozen temporal facts are checked too.
        training_session = training_session.model_copy(deep=True)
        training_session.dataset.evaluation_suite = reference
    assert_compatible(training_session, reference)
    report = {"issues": []}
    _verified_partitions(training_session, report)
    if report["issues"]:
        raise ValueError("历史训练分区存在固定评测资料泄漏。")
    return True


def evaluation_cases(session: IntakeSession, split: str = "validation") -> dict:
    from src.workbench.materialize import dataset_is_current
    from src.workbench.training_preflight import _verified_partitions

    if split not in EVALUATION_SPLITS or not dataset_is_current(session):
        raise ValueError("评测必须选择当前有效数据的开发集或最终测试集。")
    artifact = session.dataset
    report = {"issues": []}
    records = _verified_partitions(session, report)[split]
    if report["issues"]:
        raise ValueError("评测资料存在跨分区泄漏。")
    reference = artifact.evaluation_suite
    if reference is None:
        digest = content_digest([_semantic(row) for row in records])
        return {
            "suite_id": None,
            "cases_digest": digest,
            "evaluation_key": content_digest(
                {"dataset_version": artifact.version, "split": split, "cases_digest": digest}
            ),
            "records": records,
        }
    suite = verify_suite(reference)
    assignment = assert_compatible(session, reference)
    by_id = {row["metadata"]["source_row_id"]: row for row in records}
    selected = []
    for case in suite["cases"][split]:
        row_id = assignment["case_bindings"][case["case_id"]]
        if row_id not in by_id:
            raise ValueError("固定评分题没有保留在约定分区，禁止使用此数据版本评测。")
        row = copy.deepcopy(by_id[row_id])
        row["metadata"]["evaluation_case_id"] = case["case_id"]
        selected.append(row)
    digest = content_digest(
        [{"case_id": case["case_id"], **_semantic(case)} for case in suite["cases"][split]]
    )
    return {
        "suite_id": suite["suite_id"],
        "cases_digest": digest,
        "evaluation_key": content_digest(
            {"suite_id": suite["suite_id"], "split": split, "cases_digest": digest}
        ),
        "records": selected,
    }
