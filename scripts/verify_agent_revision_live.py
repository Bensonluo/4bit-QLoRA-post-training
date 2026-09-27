#!/usr/bin/env python3
"""Single purposeful online verification: the real Agent's data-revision judgment.

Per docs/plans/development-handoff-2026-09-27.md (第二优先剩余项):
- ONE online interaction (propose/confirm are local; the single online call is
  revise_data_for_iteration). No retries on connection failure — the error is
  recorded verbatim instead.
- The API key comes ONLY from the environment (TUNESMITH_AGENT_API_KEY); it is
  never written to this script, logs, or the output record.
- Success means: connection worked AND the agent produced an analysis that the
  deterministic pipeline accepted (real preview generated). Business effect is
  not claimed.

Usage:
    TUNESMITH_AGENT_API_KEY=... venv/bin/python scripts/verify_agent_revision_live.py
"""

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SESSION_ID = "cd0ffd458e024e3da2436e8cf7267c35"
PARENT_RUN_ID = "wb-cc351445b14041f58602c99c0c79fba4"
PARENT_EVALUATION_ID = "fc1802cff3ba4fa0ac1b63c9bf5fd02f"
RECORD_PATH = Path(__file__).resolve().parents[1] / (
    "docs/validation/agent-revision-live-2026-09-27.json"
)


def main() -> int:
    from src.agent.intake import CompatibleChatClient
    from src.agent.providers import load_settings
    from src.workbench.intake_service import IntakeService
    from src.workbench.iterations import IterationService

    record = {
        "date": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "单次有目的在线验证：真实 Agent 依据坏例诊断提出数据修订方案。"
            "一次在线调用，不重试；密钥仅从环境读取，不落盘。不宣称业务效果。"
        ),
        "attempted": False,
        "status": "not_attempted",
    }

    def finish() -> int:
        RECORD_PATH.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0 if record["status"] == "verified" else 1

    api_key = os.environ.get("TUNESMITH_AGENT_API_KEY", "").strip()
    if not api_key:
        record["status"] = "not_attempted"
        record["reason"] = (
            "环境未设置 TUNESMITH_AGENT_API_KEY；按交接要求密钥仅从环境或会话取得，"
            "未尝试连接（非连接失败），不盲试。设置后重跑本脚本即可执行单次验证。"
        )
        return finish()

    settings = load_settings(Path("outputs/workbench/agent-settings.json"))
    client = CompatibleChatClient(settings.base_url, settings.model, api_key, allow_remote=True)

    intake = IntakeService("outputs/workbench/intake")
    iterations = IterationService(
        "outputs/workbench/iterations",
        "outputs/workbench/training",
        "outputs/workbench/evaluations",
    )
    session = intake.load(SESSION_ID)
    # apply_analysis mutates the session object in place — snapshot before rows first.
    before_rows = {r.row_id: r.input for r in session.preview.rows} if session.preview else {}
    record["attempted"] = True
    try:
        # Local preparation: a confirmed data-change iteration for the agent to revise.
        existing = [
            item
            for item in iterations.list_iterations(session_id=SESSION_ID)
            if item["status"] == "confirmed"
        ]
        if existing:
            iteration_id = existing[-1]["iteration_id"]
            print(f"复用已确认轮次 {iteration_id}")
        else:
            proposal = iterations.propose(
                session,
                parent_run_id=PARENT_RUN_ID,
                evaluation_id=PARENT_EVALUATION_ID,
                hypothesis=(
                    "坏例诊断：三模型在固定开发集全部复述任务定义并在 32 token 截断，零分；"
                    "把该诊断交给真实 Agent，验证其能否提出可落地的数据修订方案"
                ),
                expected_outcome=(
                    "Agent 提出的方案能通过确定性管线（真实预览生成）；"
                    "修订质量与业务效果不作预设，固定评测题不可改写"
                ),
                changes="由真实 Agent 基于坏例诊断提出数据修订（本次在线验证的对象）",
                data_change=True,
            )
            iteration_id = proposal["iteration_id"]
            iterations.confirm(iteration_id, session)
            print(f"已提出并确认轮次 {iteration_id}")

        from src.agent.revisions import revise_data_for_iteration

        updated = revise_data_for_iteration(
            intake,
            iterations,
            iteration_id,
            client,
            expected_revision=intake.load(SESSION_ID).revision,
        )
        revision = iterations.get(iteration_id)["data_revision"]
        changed = [
            r.row_id
            for r in (updated.preview.rows if updated.preview else [])
            if before_rows.get(r.row_id) != r.input
        ]
        record.update(
            status="verified",
            iteration_id=iteration_id,
            model=settings.model,
            changed_components=revision["changed_components"],
            preview_rows_changed=len(changed),
            preview_rows_total=len(updated.preview.rows if updated.preview else []),
            next_action_after="review_preview（等待用户业务确认，未启动任何训练）",
            note=(
                "Agent 方案已通过确定性管线并生成真实预览；质量与效果待用户核对，"
                "本验证只证明真实 Agent 修订判断可连接、可落地。"
            ),
        )
    except Exception as exc:  # single attempt, no retry — record verbatim
        record.update(
            status="failed",
            model=settings.model,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc()[-2000:],
            note="连接或执行失败已如实记录，未重试。",
        )
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
