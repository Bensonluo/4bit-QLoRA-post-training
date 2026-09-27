"""Shared UI/CLI service for sample analysis and business-confirmed previews."""

from __future__ import annotations

import json
import random
import re
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from src.agent.intake import ChatClient, analyze_intake
from src.workbench.full_data import full_data_is_current, validate_full_source
from src.workbench.intake_models import BusinessExample, IntakeAnalysis, IntakeSession
from src.workbench.materialize import dataset_is_current, materialize_dataset
from src.workbench.recipes import check_examples, preview_recipe, validate_analysis
from src.workbench.sources import profile_source, read_source
from src.workbench.temporal_split import is_pending_label, parse_timestamp, temporal_row_times


def _mature_ready_ids(preview, recipe) -> set[str]:
    policy = recipe.temporal_split if recipe else None
    eligible = set()
    for row in preview.rows:
        if row.status != "ready" or row.target is None:
            continue
        if policy:
            try:
                if temporal_row_times(row, policy)["label_end_at"] > parse_timestamp(
                    policy.observation_end
                ):
                    continue
            except ValueError:
                continue
        eligible.add(row.row_id)
    return eligible


def next_action(session: IntakeSession) -> str:
    if session.analysis is None:
        return "awaiting_analysis"
    if any(q.blocks_confirmation for q in session.analysis.questions):
        return "needs_business_answers"
    if session.analysis.capability_gaps:
        return "needs_capability"
    if session.preview is None:
        return "needs_recipe"
    if session.preview.counts["invalid"] or session.preview.counts["conflict"]:
        return "needs_data_revision"
    recipe = session.analysis.recipe
    policy = recipe.temporal_split if recipe else None
    if policy:
        try:
            for row in session.preview.rows:
                temporal_row_times(row, policy)
        except ValueError:
            return "needs_data_revision"
    if any(
        row.status == "needs_label" and not is_pending_label(row, policy)
        for row in session.preview.rows
    ) or not _mature_ready_ids(session.preview, recipe):
        return "needs_labels"
    if session.confirmed_revision is None:
        return "review_preview"
    if full_data_is_current(session):
        if session.full_data.preview and not _mature_ready_ids(session.full_data.preview, recipe):
            return "needs_full_data_revision"
        if session.full_data.status == "needs_revision":
            return "needs_full_data_revision"
        if session.full_data.status == "review":
            return "review_full_data"
        if session.full_data.status == "confirmed":
            if dataset_is_current(session):
                return "ready_for_training_preflight"
            return "awaiting_dataset_split"
    return "awaiting_full_data" if session.source.scope == "sample" else "awaiting_full_validation"


def _stratified_sample(rows: list, size: int, seed: int) -> list[int]:
    """分层抽样:样本少于类别数时,按标签轮转保证每个类别至少一条被抽到。

    稀有类恰恰是核验最关键的——随机抽样可能整轮都抽不到它们。
    """
    if size >= len(rows):
        return list(range(len(rows)))
    generator = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_label.setdefault(row.target, []).append(index)
    for members in by_label.values():
        generator.shuffle(members)
    labels = sorted(by_label, key=lambda label: (len(by_label[label]), label))
    picked: list[int] = []
    cursor = 0
    while len(picked) < size:
        label = labels[cursor % len(labels)]
        members = by_label[label]
        slot = len(picked) // len(labels)
        if slot < len(members):
            picked.append(members[slot])
        if all(
            len(by_label[other])
            <= len(picked) // len(labels) + (0 if i >= cursor % len(labels) else 1)
            for i, other in enumerate(labels)
        ) and slot >= min(len(by_label[other]) for other in labels):
            # 轮转一圈都取满时退回纯随机补足,避免死循环
            remaining = [i for i in range(len(rows)) if i not in set(picked)]
            picked.extend(generator.sample(remaining, size - len(picked)))
            break
        cursor += 1
    return sorted(set(picked))


class IntakeService:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "intake.sqlite"
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, snapshot TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS revisions (id TEXT NOT NULL, revision INTEGER NOT NULL, snapshot TEXT NOT NULL, PRIMARY KEY(id, revision))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS label_verifications ("
                "verification_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                "full_source_digest TEXT NOT NULL, recipe_digest TEXT NOT NULL, "
                "sample_size INTEGER NOT NULL, seed INTEGER NOT NULL, row_ids TEXT NOT NULL, "
                "status TEXT NOT NULL, verdict TEXT, result TEXT, created_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS contrast_checks ("
                "check_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
                "binding TEXT NOT NULL, row_ids TEXT NOT NULL, options TEXT NOT NULL, "
                "status TEXT NOT NULL, verdict TEXT, result TEXT, created_at TEXT NOT NULL)"
            )

    def _save(self, session: IntakeSession, expected_revision: int | None) -> IntakeSession:
        updated = session.model_copy(deep=True)
        updated.revision = 0 if expected_revision is None else expected_revision + 1
        updated.updated_at = datetime.now(timezone.utc).isoformat()
        payload = updated.model_dump_json()
        with sqlite3.connect(self.database) as connection:
            if expected_revision is None:
                connection.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?)",
                    (updated.session_id, updated.revision, payload),
                )
            else:
                result = connection.execute(
                    "UPDATE sessions SET revision=?, snapshot=? WHERE id=? AND revision=?",
                    (updated.revision, payload, updated.session_id, expected_revision),
                )
                if result.rowcount != 1:
                    raise ValueError("任务已更新，请重新读取后继续；没有覆盖其他修改。")
            connection.execute(
                "INSERT INTO revisions VALUES (?, ?, ?)",
                (updated.session_id, updated.revision, payload),
            )
        return updated

    def create(
        self,
        goal: str,
        name: str,
        data: bytes,
        *,
        data_description: str = "",
        scope: Literal["sample", "full"] = "sample",
        encoding: str | None = None,
        delimiter: str | None = None,
    ) -> IntakeSession:
        if not goal.strip():
            raise ValueError("请先描述希望模型完成的业务任务。")
        source = read_source(name, data, scope=scope, encoding=encoding, delimiter=delimiter)
        session = IntakeSession(
            session_id=uuid4().hex,
            goal=goal.strip(),
            data_description=data_description,
            source=source,
            sources={"main": source},
            profile=profile_source(source),
        )
        directory = self.root / session.session_id
        directory.mkdir()
        (directory / f"source.{source.format}").write_bytes(data)
        return self._save(session, None)

    def load(self, session_id: str) -> IntakeSession:
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("无效任务 ID。")
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT snapshot FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if row is None:
            raise ValueError("找不到这个任务。")
        session = IntakeSession.model_validate_json(row[0])
        return self._attach_label_verification(session)

    def _attach_label_verification(self, session: IntakeSession) -> IntakeSession:
        """Attach the latest verified-or-not blind check matching current supervision digests.

        数据或方案修订后,旧核验不再匹配当前摘要;此时附加 ``stale`` 标记而非静默清空,
        让页面能明确告知「此前的核验已失效,需重新完成」。
        """
        report = session.full_data
        recipe = session.analysis.recipe if session.analysis else None
        if report is None or recipe is None:
            session.label_verification = None
            return session
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT verification_id, status, verdict, result, created_at FROM label_verifications "
                "WHERE session_id=? AND full_source_digest=? AND recipe_digest=? "
                "ORDER BY created_at DESC LIMIT 1",
                (session.session_id, report.source.digest, report.approved_recipe_digest),
            ).fetchone()
            if row is None:
                stale = connection.execute(
                    "SELECT verdict, created_at FROM label_verifications "
                    "WHERE session_id=? AND status='completed' "
                    "ORDER BY created_at DESC LIMIT 1",
                    (session.session_id,),
                ).fetchone()
            else:
                stale = None
        if row is None:
            if stale is not None:
                session.label_verification = {
                    "stale": True,
                    "previous_verdict": stale[0],
                    "previous_created_at": stale[1],
                }
            else:
                session.label_verification = None
            return session
        verification_id, status, verdict, result, created_at = row
        session.label_verification = {
            "verification_id": verification_id,
            "status": status,
            "verdict": verdict,
            "created_at": created_at,
            **(json.loads(result) if result else {}),
        }
        return session

    def list_sessions(self) -> list[IntakeSession]:
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                "SELECT snapshot FROM sessions ORDER BY rowid DESC"
            ).fetchall()
        return [IntakeSession.model_validate_json(row[0]) for row in rows]

    def dataset_snapshot(self, session_id: str, dataset_version: str) -> IntakeSession:
        """Find the recorded context used by a historical report, never relabel current rows."""
        self.load(session_id)
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                "SELECT snapshot FROM revisions WHERE id=? ORDER BY revision DESC", (session_id,)
            ).fetchall()
        for row in rows:
            snapshot = IntakeSession.model_validate_json(row[0])
            if snapshot.dataset and snapshot.dataset.version == dataset_version:
                return snapshot
        raise ValueError("找不到评测对应的历史数据快照，不能用当前数据冒充旧证据。")

    def answer(self, session_id: str, feedback: str) -> IntakeSession:
        session = self.load(session_id)
        if not feedback.strip():
            raise ValueError("请填写业务回答或修正说明。")
        pending = (
            "\n".join(q.question for q in session.analysis.questions)
            if session.analysis
            else "补充或修正业务理解"
        )
        session.answers.append({"question": pending, "answer": feedback.strip()})
        session.previous_analysis = session.analysis or session.previous_analysis
        session.analysis = None
        session.preview = None
        session.confirmed_revision = None
        session.dataset = None
        session.training_preflight = None
        if session.full_data:
            session.full_data.status = "stale"
            session.full_data.confirmed_revision = None
        return self._save(session, session.revision)

    def apply_analysis(
        self,
        session: IntakeSession,
        analysis: IntakeAnalysis,
        *,
        model: str = "",
        trace: list | None = None,
    ) -> IntakeSession:
        raw_sources = session.sources or {"main": session.source}
        source = raw_sources["main"]
        composition_report = None
        if analysis.composition is not None:
            from src.workbench.composition import CompositionRecipe, compose_sources

            composition = CompositionRecipe.model_validate(analysis.composition)
            composed = compose_sources(raw_sources, composition)
            if not composed.can_confirm:
                raise ValueError("组合方案仍有阻断问题，请修正关联、展开或会话规则后重新预览。")
            source = composed.source
            composition_report = composed.model_dump(exclude={"source"})
        adapter_report = None
        if analysis.adapter is not None:
            from src.workbench.adapters import AdapterRecipe, apply_adapter

            adapted = apply_adapter(source, AdapterRecipe.model_validate(analysis.adapter))
            source = adapted.source
            adapter_report = adapted.model_dump(exclude={"source"})
        validate_analysis(
            source,
            analysis,
            full_source=session.full_data.source if session.full_data else None,
            other_sources=raw_sources,
        )
        session.sources = raw_sources
        session.source = source
        session.profile = profile_source(source)
        session.composition_report = composition_report
        session.adapter_report = adapter_report
        session.previous_analysis = session.analysis or session.previous_analysis
        session.analysis = analysis
        session.preview = (
            preview_recipe(session.source, analysis.recipe) if analysis.recipe else None
        )
        if session.preview and session.confirmed_examples:
            failures = check_examples(session.preview, session.confirmed_examples)
            if failures:
                # Revisions remain reviewable; old business confirmations are not silently reused.
                session.tool_trace = [
                    {"tool": "validate_business_examples", "ok": False, "errors": failures}
                ]
            else:
                session.tool_trace = []
        else:
            session.tool_trace = []
        session.tool_trace.extend(trace or [])
        session.confirmed_revision = None
        session.dataset = None
        session.training_preflight = None
        if session.full_data:
            session.full_data.status = "stale"
            session.full_data.confirmed_revision = None
        session.agent_model = model
        return self._save(session, session.revision)

    def analyze(self, session_id: str, client: ChatClient) -> IntakeSession:
        session = self.load(session_id)
        analysis, trace = analyze_intake(session, client)
        return self.apply_analysis(session, analysis, model=client.model, trace=trace)

    def start_contrast_check(self, session_id: str, expected_revision: int) -> dict:
        """配对对比:两行输入+打乱的两个答案,用户配对——确认不再是盲点头。"""
        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("任务已更新，请读取最新预览后再开始对比核验。")
        preview = session.preview
        if preview is None:
            raise ValueError("请先完成分析并生成真实预览，再进行对比核验。")
        labelled = [row for row in preview.rows if row.target is not None]
        distinct = {row.target for row in labelled}
        if len(labelled) < 2 or len(distinct) < 2:
            raise ValueError("对比核验需要至少两条答案不同的已标注行。")
        binding = session.source.digest + (
            session.analysis.recipe.model_dump_json()
            if session.analysis and session.analysis.recipe
            else ""
        )
        # 轮次感知抽样:已完成轮数计入种子,每轮换一组题,二连对防瞎蒙。
        with sqlite3.connect(self.database) as connection:
            round_row = connection.execute(
                "SELECT COUNT(*) FROM contrast_checks "
                "WHERE session_id=? AND binding=? AND status='completed'",
                (session.session_id, binding),
            ).fetchone()
        round_number = (round_row[0] if round_row else 0) + 1
        seed = zlib.crc32(f"{binding}:{round_number}".encode())
        first, second = sorted(random.Random(seed).sample(range(len(labelled)), 2))
        rows = [labelled[first], labelled[second]]
        if rows[0].target == rows[1].target:  # 确定性兜底:答案必须不同
            for candidate in labelled:
                if candidate.target != rows[0].target:
                    rows[1] = candidate
                    break
        options = [rows[0].target, rows[1].target]
        random.Random(seed + 1).shuffle(options)
        check_id = uuid4().hex
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT INTO contrast_checks VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    check_id,
                    session.session_id,
                    binding,
                    json.dumps([row.row_id for row in rows]),
                    json.dumps(options),
                    "pending",
                    None,
                    None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return {
            "check_id": check_id,
            "items": [{"row_id": row.row_id, "input": row.input} for row in rows],
            "options": options,
            "note": "请把每个答案配到正确的输入上；配对正确才说明已看清业务含义。",
        }

    def submit_contrast_check(
        self, session_id: str, check_id: str, mapping: dict[str, str]
    ) -> dict:
        """判定配对并留档;配错不会静默——确认前必须有人真正读懂了转换。"""
        session = self.load(session_id)
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT session_id, binding, row_ids, options, status FROM contrast_checks "
                "WHERE check_id=?",
                (check_id,),
            ).fetchone()
        if row is None or row[0] != session.session_id:
            raise ValueError("找不到这个任务的对比核验。")
        stored_session, binding, row_ids_json, options_json, status = row
        if status != "pending":
            raise ValueError("该对比核验已提交过结论，请开始新的核验。")
        current_binding = session.source.digest + (
            session.analysis.recipe.model_dump_json()
            if session.analysis and session.analysis.recipe
            else ""
        )
        if binding != current_binding:
            with sqlite3.connect(self.database) as connection:
                connection.execute(
                    "UPDATE contrast_checks SET status='stale' WHERE check_id=?", (check_id,)
                )
            raise ValueError("预览或方案已变化，本次对比核验失效；请重新开始。")
        row_ids = json.loads(row_ids_json)
        options = json.loads(options_json)
        if not isinstance(mapping, dict) or set(mapping) != set(row_ids):
            raise ValueError(f"请恰好为 {len(row_ids)} 条输入各选一个答案。")
        if any(value not in options for value in mapping.values()):
            raise ValueError("答案必须来自给出的选项。")
        by_id = {preview_row.row_id: preview_row for preview_row in session.preview.rows}
        items, matched = [], 0
        for row_id in row_ids:
            correct = by_id[row_id].target
            choice = mapping[row_id]
            ok = choice == correct
            matched += ok
            items.append(
                {
                    "row_id": row_id,
                    "chosen": choice,
                    "correct_answer": correct,
                    "match": ok,
                }
            )
        verdict = "verified" if matched == len(row_ids) else "mismatch"
        result = {
            "verdict": verdict,
            "matched": matched,
            "total": len(row_ids),
            "items": items,
            "verdict_note": (
                "配对正确：转换的业务含义已被真正核对。"
                if verdict == "verified"
                else "配对错误：此前的确认可能是盲点头；请重新查看预览后再确认。"
            ),
        }
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE contrast_checks SET status='completed', verdict=?, result=? "
                "WHERE check_id=?",
                (verdict, json.dumps(result, ensure_ascii=False), check_id),
            )
        return result

    def contrast_check_status(self, session_id: str) -> dict | None:
        """当前绑定下最近结论、连胜轮数与逐轮历史(二连对才算真正看清,防瞎蒙)。

        连胜是真实的连续 verified 轮数:用户可选继续第三轮及以后,
        needs_second_round 语义不变(连胜不足两轮即需要再核验)。
        history 按轮次升序记录每一轮的题目、选择与对错——「二连对」不是
        口号,页面上每一轮都可回查核对。
        """
        session = self.load(session_id)
        if session.preview is None:
            return None
        binding = session.source.digest + (
            session.analysis.recipe.model_dump_json()
            if session.analysis and session.analysis.recipe
            else ""
        )
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute(
                "SELECT verdict, result FROM contrast_checks "
                "WHERE session_id=? AND binding=? AND status='completed' "
                "ORDER BY created_at DESC",
                (session.session_id, binding),
            ).fetchall()
        if not rows:
            return None
        streak = 0
        for verdict, _ in rows:
            if verdict == "verified":
                streak += 1
            else:
                break
        # 题目原文从当前预览行取:绑定一致说明预览行未变,按 row_id 回查;
        # 万一预览行已被替换,保留 row_id 本身,不编造题目。
        by_id = {row.row_id: row.input for row in session.preview.rows}
        history = []
        for round_number, (verdict, result_json) in enumerate(reversed(rows), start=1):
            parsed = json.loads(result_json) if result_json else {}
            history.append(
                {
                    "round": round_number,
                    "verdict": verdict,
                    "items": [
                        {
                            "row_id": item.get("row_id", ""),
                            "input": by_id.get(item.get("row_id"), item.get("row_id", "")),
                            "chosen": item.get("chosen"),
                            "correct_answer": item.get("correct_answer"),
                            "match": bool(item.get("match")),
                        }
                        for item in (parsed.get("items") or [])
                        if isinstance(item, dict)
                    ],
                }
            )
        return {
            "verdict": rows[0][0],
            "streak": streak,
            "needs_second_round": streak < 2,
            "history": history,
            **(json.loads(rows[0][1]) if rows[0][1] else {}),
        }

    def confirm(
        self, session_id: str, expected_revision: int, row_ids: list[str] | None = None
    ) -> IntakeSession:
        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("方案已变化，请先查看最新预览。")
        if next_action(session) != "review_preview" or session.preview is None:
            raise ValueError("当前方案仍有业务问题、缺标签或转换问题，不能确认数据就绪。")
        chosen = set(row_ids) if row_ids is not None else {r.row_id for r in session.preview.rows}
        if not chosen or chosen - {r.row_id for r in session.preview.rows}:
            raise ValueError("请指定已经查看的有效预览行。")
        mature_ids = _mature_ready_ids(session.preview, session.analysis.recipe)
        if not chosen & mature_ids:
            raise ValueError(
                "请至少核对一条已有真实答案且标签窗口成熟的样例；不能只确认尚未观察到答案的记录。"
            )
        session.confirmed_examples = [
            BusinessExample(row_id=r.row_id, expected_input=r.input, expected_target=r.target)
            for r in session.preview.rows
            if r.row_id in chosen & mature_ids
        ]
        session.confirmed_revision = session.revision
        return self._save(session, session.revision)

    def validate_full_data(
        self,
        session_id: str,
        expected_revision: int,
        name: str | None = None,
        data: bytes | None = None,
        *,
        encoding: str | None = None,
        delimiter: str | None = None,
    ) -> IntakeSession:
        """Validate an upload, or the original source when it was declared full data."""
        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("方案已变化，请先查看最新预览与全量报告。")
        if session.confirmed_revision is None:
            raise ValueError("请先确认样例的业务含义与转换预览，再验证全量数据。")
        if session.analysis and session.analysis.composition:
            if name is not None or data is not None:
                raise ValueError(
                    "当前方案使用组合步骤，请按资料别名提供全量来源，不能把组合结果当原始来源上传。"
                )
            return self.validate_full_sources(session_id, expected_revision)
        if name is None and data is None and session.source.scope == "full":
            raw_source = (session.sources or {"main": session.source})["main"]
            name = raw_source.name
            original = (
                self.root / session_id / "sources" / f"{raw_source.digest}.{raw_source.format}"
            )
            if not original.exists():
                original = self.root / session_id / f"source.{raw_source.format}"
            data = original.read_bytes()
            encoding = encoding or raw_source.encoding or None
            delimiter = delimiter or raw_source.delimiter or None
        if name is None or data is None:
            raise ValueError("请提供本次任务的全量文件；原文件只声明为样例，不能自动当作全量。")
        source = read_source(name, data, scope="full", encoding=encoding, delimiter=delimiter)
        original_source = source
        adapted = None
        adapter_error = None
        if session.analysis and session.analysis.adapter:
            from src.workbench.adapters import AdapterRecipe, apply_adapter

            try:
                adapted = apply_adapter(
                    source,
                    AdapterRecipe.model_validate(session.analysis.adapter),
                    require_source_example=False,
                )
                source = adapted.source
            except ValueError as exc:
                adapter_error = str(exc)
        session.full_data = validate_full_source(session, source)
        session.full_data.sources = {"main": original_source}
        if adapted:
            session.full_data.adapter_report = adapted.model_dump(exclude={"source"})
        if adapter_error:
            from src.workbench.intake_models import FullDataIssue

            session.full_data.adapter_report = {
                "validation": {"status": "failed", "error": adapter_error}
            }
            session.full_data.issues.append(
                FullDataIssue(code="adapter_failed", severity="blocking", message=adapter_error)
            )
            session.full_data.status = "needs_revision"
        session.dataset = None
        session.training_preflight = None
        directory = self.root / session_id / "full"
        directory.mkdir(exist_ok=True)
        try:
            with (directory / f"{original_source.digest}.{original_source.format}").open(
                "xb"
            ) as handle:
                handle.write(data)
        except FileExistsError:
            pass  # Content-addressed original bytes are shared by repeated validation.
        return self._save(session, session.revision)

    def confirm_full_data(
        self, session_id: str, expected_revision: int, row_ids: list[str] | None = None
    ) -> IntakeSession:
        """Record review of this full-data report; this does not authorize training."""
        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("全量报告或方案已变化，请先查看最新结果。")
        report = session.full_data
        if next_action(session) != "review_full_data" or report is None or report.preview is None:
            raise ValueError("全量方案仍有未解决的问题或已失效，不能确认；请先处理问题并重新验证。")
        known = {r.row_id for r in report.preview.rows}
        chosen = set(row_ids) if row_ids is not None else known
        if not chosen or chosen - known:
            raise ValueError("请指定已经查看的有效全量预览行。")
        mature_ids = _mature_ready_ids(report.preview, session.analysis.recipe)
        if not chosen & mature_ids:
            raise ValueError("请至少核对一条已有真实答案且标签窗口成熟的全量记录。")
        report.confirmed_examples = [
            BusinessExample(row_id=r.row_id, expected_input=r.input, expected_target=r.target)
            for r in report.preview.rows
            if r.row_id in chosen & mature_ids
        ]
        report.confirmed_revision = session.revision
        report.status = "confirmed"
        return self._save(session, session.revision)

    def materialize_dataset(
        self,
        session_id: str,
        expected_revision: int,
        *,
        name: str | None = None,
        registry_root: str | Path | None = None,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        independent_rows_confirmed: bool = False,
        evaluation_suite: dict | None = None,
    ) -> IntakeSession:
        """Publish confirmed full data; the resulting config still needs preflight."""
        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("方案或全量报告已变化，请查看最新结果后再生成分区。")
        session.dataset = materialize_dataset(
            session,
            registry_root=registry_root if registry_root is not None else self.root / "datasets",
            name=name if name is not None else f"intake-{session_id}",
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            independent_rows_confirmed=independent_rows_confirmed,
            evaluation_suite=evaluation_suite,
        )
        session.training_preflight = None
        saved = self._save(session, session.revision)
        return self._attach_label_verification(saved)

    def start_label_verification(
        self, session_id: str, expected_revision: int, *, sample_size: int = 5
    ) -> dict:
        """Sample labelled rows and hide their answers; the user must reproduce them."""
        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("任务已更新，请读取最新结果后再开始盲标核验。")
        report = session.full_data
        if report is None or report.status != "confirmed" or report.preview is None:
            raise ValueError("请先确认全量数据，再进行盲标核验。")
        if type(sample_size) is not int or sample_size < 1 or sample_size > 50:
            raise ValueError("抽样数量必须是 1 到 50 之间的整数。")
        labelled = [row for row in report.preview.rows if row.target is not None]
        if not labelled:
            raise ValueError("当前全量预览没有已标注的行，无法核验。")
        size = min(sample_size, len(labelled))
        binding = report.source.digest + report.approved_recipe_digest
        seed = zlib.crc32(binding.encode("utf-8"))
        picked = _stratified_sample(labelled, size, seed)
        rows = [labelled[index] for index in picked]
        verification_id = uuid4().hex
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT INTO label_verifications VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    verification_id,
                    session.session_id,
                    report.source.digest,
                    report.approved_recipe_digest,
                    size,
                    seed,
                    json.dumps([row.row_id for row in rows]),
                    "pending",
                    None,
                    None,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return {
            "verification_id": verification_id,
            "sample_size": size,
            "seed": seed,
            "items": [{"row_id": row.row_id, "input": row.input} for row in rows],
            "note": "请仅根据输入作答，不要查看数据中的现有答案；答案不会随题目显示。",
        }

    def submit_label_verification(
        self, session_id: str, verification_id: str, answers: dict[str, str]
    ) -> dict:
        """Compare the user's blind answers with the data labels; verdict gates training."""
        session = self.load(session_id)
        with sqlite3.connect(self.database) as connection:
            row = connection.execute(
                "SELECT session_id, full_source_digest, recipe_digest, row_ids, status "
                "FROM label_verifications WHERE verification_id=?",
                (verification_id,),
            ).fetchone()
        if row is None or row[0] != session.session_id:
            raise ValueError("找不到这个任务的盲标核验。")
        stored_session, full_digest, recipe_digest, row_ids_json, status = row
        if status != "pending":
            raise ValueError("该盲标核验已提交过结论，请开始新的核验。")
        report = session.full_data
        if (
            report is None
            or report.source.digest != full_digest
            or report.approved_recipe_digest != recipe_digest
        ):
            with sqlite3.connect(self.database) as connection:
                connection.execute(
                    "UPDATE label_verifications SET status='stale' WHERE verification_id=?",
                    (verification_id,),
                )
            raise ValueError("资料或处理方案已变化，本次核验失效；请重新开始盲标核验。")
        row_ids = json.loads(row_ids_json)
        if not isinstance(answers, dict) or set(answers) != set(row_ids):
            raise ValueError(f"请恰好对抽样的 {len(row_ids)} 条作答，不能多答或漏答。")
        if any(not isinstance(value, str) or not value.strip() for value in answers.values()):
            raise ValueError("每条都需要作答；无法判断时请照实填写，不要留空。")
        by_id = {preview_row.row_id: preview_row for preview_row in report.preview.rows}
        items, matched = [], 0
        for row_id in row_ids:
            target = by_id[row_id].target
            submitted = answers[row_id].strip()
            match = submitted == target
            matched += match
            items.append(
                {
                    "row_id": row_id,
                    "input": by_id[row_id].input,
                    "submitted_answer": submitted,
                    "data_label": target,
                    "match": match,
                }
            )
        verdict = "verified" if matched == len(row_ids) else "insufficient_agreement"
        result = {
            "sample_size": len(row_ids),
            "matched": matched,
            "agreement": matched / len(row_ids),
            "items": items,
            "verdict_note": (
                "抽样行全部一致：监督信号的业务含义经用户独立复现。"
                if verdict == "verified"
                else "存在不一致：数据标签与业务理解有分歧，请逐条核对原因（标签错误、歧义或任务定义不清），修正后重新核验。"
            ),
        }
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE label_verifications SET status='completed', verdict=?, result=? "
                "WHERE verification_id=?",
                (verdict, json.dumps(result, ensure_ascii=False), verification_id),
            )
        refreshed = self.load(session_id)
        return refreshed.label_verification

    def add_source(
        self,
        session_id: str,
        expected_revision: int,
        alias: str,
        name: str,
        data: bytes,
        *,
        scope: Literal["sample", "full"] = "sample",
        encoding: str | None = None,
        delimiter: str | None = None,
    ) -> IntakeSession:
        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("资料已更新，请读取最新任务后再添加。")
        if not re.fullmatch(r"[\w-]+", alias):
            raise ValueError("资料别名请使用文字、数字、下划线或短横线。")
        source = read_source(name, data, scope=scope, encoding=encoding, delimiter=delimiter)
        sources = session.sources or {"main": session.source}
        sources[alias] = source
        session.sources = sources
        session.source = sources["main"]
        session.profile = profile_source(session.source)
        session.composition_report = None
        session.adapter_report = None
        session.previous_analysis = session.analysis or session.previous_analysis
        session.analysis = None
        session.preview = None
        session.confirmed_revision = None
        session.dataset = None
        session.training_preflight = None
        if session.full_data:
            session.full_data.status = "stale"
            session.full_data.confirmed_revision = None
        directory = self.root / session_id / "sources"
        directory.mkdir(exist_ok=True)
        path = directory / f"{source.digest}.{source.format}"
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("已保存来源内容异常，未覆盖原始资料。")
        else:
            with path.open("xb") as handle:
                handle.write(data)
        return self._save(session, session.revision)

    def validate_full_sources(
        self,
        session_id: str,
        expected_revision: int,
        files: dict[str, tuple[str, bytes]] | None = None,
    ) -> IntakeSession:
        from src.workbench.composition import CompositionRecipe, compose_sources, required_sources
        from src.workbench.intake_models import FullDataIssue

        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("方案已更新，请查看最新方案后再验证全量来源。")
        if (
            not session.analysis
            or not session.analysis.composition
            or session.confirmed_revision is None
        ):
            raise ValueError("请先确认组合方案及真实样例预览。")
        recipe = CompositionRecipe.model_validate(session.analysis.composition)
        needed = required_sources(recipe)
        if files is not None:
            if needed - set(files):
                raise ValueError(f"缺少这些全量来源：{sorted(needed - set(files))}")
            sources = {
                alias: read_source(name, data, scope="full")
                for alias, (name, data) in files.items()
            }
        else:
            sources = (
                session.full_data.sources
                if session.full_data and session.full_data.sources
                else session.sources
            )
            if needed - set(sources) or any(sources[alias].scope != "full" for alias in needed):
                raise ValueError("请提供组合方案所需的每份全量资料，不能将样例自动当作全量。")
        composed = compose_sources(sources, recipe)
        full_source = composed.source
        adapted = None
        adapter_error = None
        if composed.can_confirm and session.analysis.adapter:
            from src.workbench.adapters import AdapterRecipe, apply_adapter

            try:
                adapted = apply_adapter(
                    full_source,
                    AdapterRecipe.model_validate(session.analysis.adapter),
                    require_source_example=False,
                )
                full_source = adapted.source
            except ValueError as exc:
                adapter_error = str(exc)
        report = validate_full_source(session, full_source)
        if adapted:
            report.adapter_report = adapted.model_dump(exclude={"source"})
        if adapter_error:
            report.adapter_report = {"validation": {"status": "failed", "error": adapter_error}}
            report.issues.append(
                FullDataIssue(code="adapter_failed", severity="blocking", message=adapter_error)
            )
            report.status = "needs_revision"
        report.sources = sources
        report.composition_report = composed.model_dump(exclude={"source"})
        report.issues.extend(
            FullDataIssue(
                code=f"composition_{issue.code}",
                severity=issue.severity,
                message=issue.message,
                row_ids=getattr(issue, "row_ids", []),
            )
            for issue in composed.issues
        )
        if not composed.can_confirm:
            report.status = "needs_revision"
        if files:
            directory = self.root / session_id / "sources"
            directory.mkdir(exist_ok=True)
            for alias, (_, data) in files.items():
                path = directory / f"{sources[alias].digest}.{sources[alias].format}"
                if path.exists():
                    if path.read_bytes() != data:
                        raise ValueError("已保存来源内容异常，未覆盖原始资料。")
                else:
                    with path.open("xb") as handle:
                        handle.write(data)
        session.full_data = report
        session.dataset = None
        session.training_preflight = None
        return self._save(session, session.revision)

    def preflight_training(
        self,
        session_id: str,
        expected_revision: int,
        tokenizer: object,
        max_length: int,
    ) -> IntakeSession:
        from src.workbench.training_preflight import preflight_dataset

        session = self.load(session_id)
        if session.revision != expected_revision:
            raise ValueError("数据或方案已更新，请读取最新版本后再检查。")
        session.training_preflight = preflight_dataset(session, tokenizer, max_length)
        return self._save(session, session.revision)
