"""Presentation of the existing intake state; no second workflow state machine."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from src.workbench.intake_service import next_action


@dataclass(frozen=True)
class TaskState:
    label: str
    hint: str
    step: int | None
    attention: bool = False


PREPARATION_STEPS = ("目标与样例", "监督核验", "全量验证", "独立分区", "训练预检", "准备训练")
_STATES = {
    "awaiting_analysis": TaskState("待分析目标与数据", "梳理业务输入、期望答案与样例数据。", 0),
    "needs_business_answers": TaskState(
        "待补充业务判断", "先回答影响方案确认的业务问题。", 0, True
    ),
    "needs_capability": TaskState(
        "资料或能力待补齐", "核对当前缺口，补充资料或调整业务目标。", 0, True
    ),
    "needs_recipe": TaskState(
        "转换方案待完善", "完善输入与答案的处理规则，再核对真实预览。", 0, True
    ),
    "needs_data_revision": TaskState(
        "数据问题待处理", "检查转换异常与同输入答案冲突，再修订资料或规则。", 1, True
    ),
    "needs_labels": TaskState(
        "待补充监督答案", "补齐缺失的答案，确认这些标签正是业务希望模型学习的内容。", 1, True
    ),
    "review_preview": TaskState("待核验监督语义", "核对真实转换预览，通过对比核验后确认方案。", 1),
    "awaiting_full_data": TaskState(
        "待提供全量数据", "样例方案已确认，继续核验全量数据的覆盖与质量。", 2
    ),
    "awaiting_full_validation": TaskState(
        "待验证全量数据", "检查全量覆盖、冲突、业务分组与训练消费要求。", 2
    ),
    "needs_full_data_revision": TaskState(
        "全量数据待修订", "处理报告中的阻断问题后，重新验证全量数据。", 2, True
    ),
    "review_full_data": TaskState(
        "待确认全量报告", "核对覆盖范围及问题处理结果，确认本次全量数据。", 2
    ),
    "awaiting_dataset_split": TaskState(
        "待生成独立分区", "准备版本化的训练、开发与最终测试分区。", 3
    ),
    "ready_for_training_preflight": TaskState(
        "待训练前检查", "用真实 tokenizer 检查答案保留、截断与训练协议。", 4
    ),
    "preflight_passed": TaskState(
        "数据预检已通过", "在任务中核对训练方案，并查看训练、对照与迭代记录。", 5
    ),
}
_UNKNOWN = TaskState("状态待核对", "打开任务记录核对当前状态与下一步。", None, True)


def task_state(session) -> TaskState:
    return _STATES.get(next_action(session), _UNKNOWN)


def preparation_progress(session) -> None:
    state = task_state(session)
    st.html(
        '<div class="ts-next"><span class="ts-eyebrow">当前准备阶段</span>'
        f"<h3>{escape(state.label)}</h3><p>{escape(state.hint)}</p></div>"
        '<ol class="ts-preparation" aria-label="数据准备进度">'
        + "".join(
            f'<li class="{"is-current" if i == state.step else ""}"><span>{i + 1:02d}</span>{escape(label)}</li>'
            for i, label in enumerate(PREPARATION_STEPS)
        )
        + "</ol>"
    )
