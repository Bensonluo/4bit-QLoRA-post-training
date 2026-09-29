"""Business goal + sample data → agent diagnosis and executable data preview."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src.agent.intake import CompatibleChatClient, is_local_endpoint
from src.workbench.demo_task import (
    DEMO_DESCRIPTION,
    DEMO_FULL_BUTTON,
    DEMO_GOAL,
    DEMO_SAMPLE_BUTTON,
    DEMO_SAMPLE_ENTRY_LABEL,
    demo_full,
    demo_sample,
    is_demo_session,
)
from src.workbench.evaluation_suites import EvalSuiteService
from src.workbench.intake_service import (
    IntakeService,
    agreement_evidence_note,
    next_action,
    wilson_lower_bound,
)
from src.workbench.iterations import IterationService
from src.workbench.training_guidance import LR_TIER_DISCLAIMER, learning_rate_suggestion
from src.workbench.training_runs import TrainingRunService
from ui.components.agent_settings import render_agent_settings
from ui.config import PROJECT_ROOT

st.set_page_config(page_title="目标与数据 — TuneSmith", page_icon="🧩", layout="wide")
st.title("从业务目标和数据开始")
st.caption(
    "描述希望模型完成的工作，提供一份样例。我们一起判断资料是否适合、还缺什么，并预览真实的数据处理结果。"
)
service = IntakeService(PROJECT_ROOT / "outputs" / "workbench" / "intake")


TEMPORAL_EXCLUSION_NAMES = {
    "label_not_mature": "观察截止时标签尚未成熟",
    "label_window_crosses_validation_start": "训练标签窗口跨越验证起点",
    "label_window_crosses_test_start": "验证标签窗口跨越测试起点",
    "connected_to_excluded_row": "与排除记录属于同一业务对象或相同模型输入",
}


def show_temporal_policy(
    policy, *, title: str = "**已提出的时间分区方案（随业务预览确认）**"
) -> None:
    st.write(title)
    st.dataframe(
        [
            {"时间含义": label, "来源字段": column}
            for label, column in (
                ("信息实际可获得时间", policy.available_at_column),
                ("作出预测时间", policy.prediction_at_column),
                ("未来标签窗口结束时间", policy.label_end_at_column),
            )
        ],
        hide_index=True,
        width="stretch",
    )
    st.write(
        f"验证起点：{policy.validation_start}；测试起点：{policy.test_start}；观察截止：{policy.observation_end}。"
    )
    st.caption(
        "信息可得时间不晚于预测时间，预测时间早于标签窗口结束。训练/验证标签必须在下一分区起点前成熟；跨界或截至观察期仍未成熟的行保留排除，不随机补数，不补伪标签。"
    )


def preview_answer(row) -> str:
    if row.target is not None:
        return row.target
    from src.workbench.temporal_split import is_pending_label

    policy = (
        session.analysis.recipe.temporal_split
        if session.analysis and session.analysis.recipe
        else None
    )
    if is_pending_label(row, policy):
        return "标签窗口尚未成熟，当前保留排除；没有填造监督答案。"
    return "尚缺答案，需要补充标签"


def select_preview_rows(rows, *, key: str, label: str, issue_ids=None):
    """Every row remains reachable; filtering and pagination share one UI contract."""
    issue_ids = issue_ids or set()
    options = {
        "全部记录（问题优先）": None,
        "仅需处理记录": "issues",
        "缺少答案": "needs_label",
        "转换异常": "invalid",
        "答案冲突": "conflict",
        "已生成预览": "ready",
    }
    selected = st.selectbox(f"{label}记录筛选", list(options), key=f"{key}_filter")
    status = options[selected]

    def problem(row):
        return row.status != "ready" or row.row_id in issue_ids

    filtered = [
        row
        for row in rows
        if status is None or (problem(row) if status == "issues" else row.status == status)
    ]
    filtered.sort(key=lambda row: not problem(row))
    if not filtered:
        st.caption(f"该筛选下没有记录；全部共 {len(rows)} 条。")
        return []
    pages = (len(filtered) + 19) // 20
    page = (
        int(
            st.number_input(
                f"{label}预览页码",
                min_value=1,
                max_value=pages,
                value=1,
                step=1,
                key=f"{key}_page_{selected}",
            )
        )
        if pages > 1
        else 1
    )
    start = (page - 1) * 20
    visible = filtered[start : start + 20]
    st.caption(
        f"第 {page}/{pages} 页 · 显示筛选后第 {start + 1}–{start + len(visible)} 条，共 {len(filtered)} 条；全部原始记录 {len(rows)} 条。"
    )
    return visible


def show_missing_label_next_step(rows, *, full=False) -> None:
    from src.workbench.answer_sheet import (
        answer_sheet_csv,
        answer_sheet_lines,
        missing_answer_rows,
    )

    recipe = session.analysis.recipe if session.analysis else None
    policy = recipe.temporal_split if recipe else None
    # 缺答案行筛选单一来源(missing_answer_rows):页面清单、页面提示背后的
    # needs_labels 判定与 CLI answer-sheet 命令必须是同一份行。
    missing = missing_answer_rows(rows, policy)
    if not missing:
        return
    field_names = [field.column for field in recipe.targets] if recipe and recipe.targets else []
    columns = "、".join(field_names) if field_names else "待业务确认的答案列"
    upload_step = (
        "在「全量数据验证」重新上传补齐后的对应全量文件。"
        if full
        else "在「原始资料与补充文件」替换对应原始文件后重新分析；独立标签表则上传并说明关联字段。"
    )
    st.info(
        f"需补齐 {len(missing)} 条记录的答案（字段：{columns}）。"
        "请保留原编号，由了解业务的人填写后，"
        + upload_step
        + "不确定答案含义时，先在「回答问题或修正理解」补充说明。"
    )
    # 介入点 10 编码:填写人拿到的是只含待补行的填写表(行ID+题目输入+待填列),
    # 而不是一段口头指引;交接三条规则(answer_sheet_lines 单一来源)与 CLI 同源。
    if field_names:
        st.download_button(
            "导出待补答案清单（交给填写人）",
            data=answer_sheet_csv(missing, field_names),
            file_name=f"待补答案清单-{'全量' if full else '样例'}.csv",
            mime="text/csv",
            key=f"answer_sheet_{'full' if full else 'sample'}",
        )
    for line in answer_sheet_lines(len(missing)):
        st.caption(line)


def show_temporal_exclusions(rows: list[dict], *, title: str) -> None:
    if not rows:
        return
    st.write(f"**{title}**")
    st.dataframe(
        [
            {
                "行ID": item["row_id"],
                "排除原因": TEMPORAL_EXCLUSION_NAMES.get(item["reason"], item["reason"]),
                **item.get("times", {}),
            }
            for item in rows
        ],
        hide_index=True,
        width="stretch",
    )
    with st.expander(title + "：保留的原行与输入/标签"):
        st.json(rows)


def show_fact_notes(profile, *, alias: str = "") -> None:
    """渲染来源如实标注(第 18/24/25/26/28/29 轮):sheet_note 说明读取范围(info 级),
    merged/formula/hidden/blank/dup_header 是影响数据事实的告知(warning 级)——
    blank_note 跨格式(CSV/JSONL 空行已跳过、Excel 全空行照常读入),dup_header_note
    同样跨格式(与表头完全相同的数据行,样例侧不拦、全量侧硬拦),不再只是 Excel 事实。
    profile(dict)与 SampleSource(属性)同键,统一取用;为空不渲染,顺序固定。"""
    for key in (
        "sheet_note",
        "merged_note",
        "formula_note",
        "hidden_note",
        "blank_note",
        "dup_header_note",
    ):
        note = profile.get(key) if isinstance(profile, dict) else getattr(profile, key, None)
        if not note:
            continue
        text = f"{alias}：{note}" if alias else note
        (st.info if key == "sheet_note" else st.warning)(text)


def probe_source_hints(session) -> dict[str, str]:
    """候选行 → 原始来源行提示:行号取自全量来源记录,供人工对照原始行溯源。

    探针只认识分区里的 source_row_id;溯源必须回到全量资料本身——
    这里把 row_id 映射回全量文件中的真实行号,找不到就不编造。
    """
    full = getattr(session, "full_data", None)
    if full is None:
        return {}
    return {row.row_id: f"全量资料第 {row.line} 行" for row in full.source.rows}


def render_probe_result(result: dict, *, source_hints: dict[str, str] | None = None) -> None:
    """渲染一次可学性探针结果:指标、判定与标签问题候选(证据强者在先)。"""
    delta = result["difference"]
    # 三态判定词汇与 CLI 同源(probe_verdict_phrase):页面与 stderr 不各说各话。
    from src.workbench.learnability_probe import low_baseline_triage_lines, probe_verdict_phrase

    verdict = probe_verdict_phrase(result)
    st.metric(
        f"零样本 {result['zero_shot_accuracy']:.0%} vs 基线 {result['majority_baseline']:.0%}",
        f"{delta:+.0%}",
        delta_color="normal" if delta >= 0 else "inverse",
    )
    st.info(f"{verdict}。{result['note']}")
    # 低于瞎猜基线时按记录内事实逐行给出核查方向分辨(单一来源,与 CLI stderr 同一份行):
    # 截断→先加长度重测;词汇外→模板方向;同答→两方向都要核;其余→任务定义方向。
    for triage_line in low_baseline_triage_lines(result):
        st.caption(triage_line)
    candidates = result.get("label_error_candidates") or []
    st.subheader("标签问题候选(优先人工核对)")
    if candidates:
        # 证据强者在先不依赖存盘顺序;候选多于 20 条分页展示,防止全量渲染淹没页面。
        # 分页只影响展示:计数始终如实,CSV 导出不受分页影响,始终包含全部候选。
        candidates = sorted(candidates, key=lambda item: not item.get("user_blind_answer"))
        page_size = 20
        pages = (len(candidates) + page_size - 1) // page_size
        page_number = 1
        if pages > 1:
            page_number = int(
                st.number_input(
                    "候选预览页码",
                    min_value=1,
                    max_value=pages,
                    value=1,
                    step=1,
                    key=f"probe_candidates_page_{result.get('dataset_version', 'x')}",
                )
            )
        start = (page_number - 1) * page_size
        visible = candidates[start : start + page_size]
        hints = source_hints or {}
        st.dataframe(
            [
                {
                    "行ID": item["row_id"],
                    "数据标签": item["data_label"],
                    "基座零样本输出": item["base_zero_shot"],
                    "你的盲标答案": item.get("user_blind_answer") or "—",
                    "证据": item["evidence"],
                    "来源提示": hints.get(item["row_id"], "未在全量资料中定位到该行"),
                }
                for item in visible
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"显示第 {start + 1}–{start + len(visible)} 条，共 {len(candidates)} 条候选"
            + (f"（第 {page_number}/{pages} 页，每页 {page_size} 条）" if pages > 1 else "")
            + "；下方 CSV 导出包含全部候选，不止当前页。"
        )
        st.caption(result.get("candidates_note", ""))
        from src.workbench.learnability_probe import candidates_to_csv

        st.download_button(
            "导出候选为 CSV(供人工核对)",
            data=candidates_to_csv(candidates),
            file_name="label_issue_candidates.csv",
            mime="text/csv",
            key=f"probe_candidates_csv_{result.get('dataset_version', 'x')}",
        )
    else:
        st.info("没有发现值得优先核对的行。")


def collect_current_task_spec(session_id: str) -> dict:
    """两处规约卡共用的只读投影路由参数（ADR-1：单一来源，不新增状态）。"""
    from src.workbench.task_spec_projection import collect_task_spec

    return collect_task_spec(
        session_id,
        PROJECT_ROOT / "outputs/workbench/intake",
        PROJECT_ROOT / "outputs/workbench/business-scoring",
        PROJECT_ROOT / "outputs/workbench/acceptance",
        PROJECT_ROOT / "outputs/workbench/evaluations",
        PROJECT_ROOT / "outputs/workbench/iterations",
        PROJECT_ROOT / "outputs/workbench/training",
    )


def scoring_store():
    from src.workbench.business_scoring import ScoringService

    return ScoringService(PROJECT_ROOT / "outputs/workbench/business-scoring")


def select_business_scoring(*, key: str) -> dict | None:
    try:
        choices = {
            item["scoring_id"]: item
            for item in scoring_store().list_specs(session_id=session.session_id)
            if item["status"] == "confirmed"
        }
    except (ValueError, OSError) as exc:
        st.error(f"无法读取评分规则：{exc}")
        return None
    if not choices:
        return None
    selected = st.selectbox(
        "本次使用的评分规则",
        ["", *choices],
        format_func=lambda identity: (
            "按已确认任务的默认评分"
            if not identity
            else choices[identity]["recipe"]["business_standard"][:80] + " · " + identity[:12]
        ),
        key=key,
    )
    if not selected:
        return None
    spec = choices[selected]
    st.caption(
        f"自定义业务评分：单题得分达到 {spec['recipe']['pass_threshold']} 才计为通过；均分与通过率分开展示。"
    )
    return {
        "root": str((PROJECT_ROOT / "outputs/workbench/business-scoring").resolve()),
        "scoring_id": selected,
        "spec_digest": spec["spec_digest"],
    }


def show_business_scoring() -> None:
    st.subheader("自定义业务评分（可选）")
    st.caption(
        "描述什么回答算好、什么错误不能接受。Agent 用开发样例起草可执行规则和真实正反例；先查看实际评分与理由，再明确确认。主观判断尚不清楚时先补充标准，不据此自动认定通过。"
    )
    standard = st.text_area(
        "希望怎样判断模型回答是否满足业务要求？", key=f"scoring_standard_{session.session_id}"
    )
    authorized = is_local_endpoint(base_url)
    if not authorized:
        authorized = (
            st.checkbox(
                "允许向已选 Agent 服务发送业务评分要求、处理方案及所需开发样例；不发送最终测试题。",
                key=f"scoring_consent_{session.session_id}_{base_url}",
            )
            and allow_remote
        )
    if st.button("让 Agent 拟定业务评分规则", disabled=not authorized):
        if not standard.strip():
            st.error("请先说明业务评分标准和不能接受的错误。")
        else:
            try:
                from src.agent.scoring import recommend_scoring

                client = CompatibleChatClient(base_url, model, api_key, allow_remote=authorized)
                with st.spinner("Agent 正在根据开发样例拟定规则，并执行隔离正反例验证…"):
                    scoring_response = recommend_scoring(
                        service.load(session.session_id),
                        standard.strip(),
                        client,
                        output_root=PROJECT_ROOT / "outputs/workbench/business-scoring",
                    )
                st.session_state[f"scoring_questions_{session.session_id}"] = (
                    scoring_response
                    if scoring_response.get("status") == "needs_business_input"
                    else None
                )
                st.rerun()
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                st.error(str(exc))
    pending = st.session_state.get(f"scoring_questions_{session.session_id}")
    if pending:
        st.warning(pending.get("reason") or "评分含义尚不明确，请先补充业务标准。")
        for question in pending.get("questions", []):
            st.write(
                question.get("question", str(question)) if isinstance(question, dict) else question
            )
        st.info("请补充上方业务要求后重新拟定规则；当前没有可确认的评分方案。")
    try:
        specs = scoring_store().list_specs(session_id=session.session_id)
    except (ValueError, OSError) as exc:
        st.error(f"无法读取评分草案：{exc}")
        return
    for spec in specs:
        identity = spec["scoring_id"]
        recipe = spec["recipe"]
        validation = spec.get("validation") or {}
        with st.expander(
            f"评分规则 {identity[:12]} · {'已确认' if spec['status'] == 'confirmed' else '待业务确认'}",
            expanded=spec["status"] != "confirmed",
        ):
            st.write(recipe["business_standard"])
            st.write(f"**单题通过分数：** {recipe['pass_threshold']}")
            st.caption(
                f"实际隔离后端：{validation.get('backend', '未记录')} · 验证状态：{validation.get('status', '未验证')}"
            )
            from src.workbench.report_summary import summarize_scoring

            for line in summarize_scoring(spec):
                st.write(line)
            if spec.get("example_results"):
                st.write("**实际正反例分数与理由**")
                st.dataframe(spec["example_results"], hide_index=True, width="stretch")
            if validation.get("status") != "passed":
                st.warning("尚未获得通过的真实隔离验证，不能确认此规则。")
            with st.expander(f"规则 {identity[:12]} 的预期样例、代码与检查记录"):
                st.json(recipe.get("examples", []))
                st.code(recipe.get("source_code", ""), language="python")
                st.json(recipe.get("config", {}))
                st.json(validation)
                st.json(spec.get("trace", []))
            confirmed = spec["status"] == "confirmed"
            acknowledge = st.checkbox(
                "已核对实际正反例分数与理由，确认它们表达当前业务标准。",
                key=f"ack_scoring_{identity}",
                value=confirmed,
                disabled=confirmed,
            )
            if st.button(
                "确认这套业务评分规则",
                key=f"confirm_scoring_{identity}",
                disabled=confirmed or not acknowledge or validation.get("status") != "passed",
            ):
                try:
                    scoring_store().confirm(identity, service.load(session.session_id))
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))


def show_final_acceptance(run: dict) -> None:
    """Keep frozen single-model final evidence separate from development and Agent tools."""
    from src.workbench.acceptance import AcceptanceService
    from src.workbench.business_evaluation import EvaluationModel, EvaluationProtocol
    from src.workbench.report_summary import acceptance_gate_lines

    acceptance = AcceptanceService(
        PROJECT_ROOT / "outputs/workbench/acceptance",
        PROJECT_ROOT / "outputs/workbench/evaluations",
    )
    run_id = run["run_id"]
    with st.expander("按业务标准做最终验收"):
        st.caption(
            "选定这一模型，在固定独立测试题上验收。先写清业务标准、最低通过率及最低测试题数，再冻结条款并执行。执行会暴露题目，换门槛或换题集标识不能把旧题变成新盲测；最终题和坏例不会交给 Agent 优化。"
        )
        if (
            session.dataset
            and run.get("dataset_version") == session.dataset.version
            and session.analysis
            and session.analysis.recipe
        ):
            recipe = session.analysis.recipe
            scorer = (
                "json_fields_exact"
                if recipe.output_format == "json"
                else "classification_exact"
                if len(recipe.targets) == 1 and recipe.targets[0].value_kind == "categorical"
                else "open_review"
            )
            custom_scoring = select_business_scoring(key=f"acceptance_scoring_{run_id}")
            if custom_scoring:
                scorer = "custom_rules"
            st.write(
                "**评分规则：** "
                + {
                    "classification_exact": "类别严格匹配",
                    "json_fields_exact": "全部声明 JSON 字段严格匹配",
                    "open_review": "逐题人工业务判断",
                    "custom_rules": "已确认自定义业务评分规则",
                }[scorer]
            )
            with st.expander("📋 任务规约（冻结验收条款前的口径）"):
                # 与 task-spec-show、训练启动前折叠区同源(summarize_task_spec 单一来源)。
                from src.workbench.task_spec_projection import summarize_task_spec

                spec = collect_current_task_spec(session.session_id)
                for line in summarize_task_spec(spec):
                    st.write(line)
            with st.form(f"acceptance_prepare_{run_id}"):
                standard = st.text_area(
                    "最终业务验收标准", placeholder="说明什么样的回答可交付，以及哪些错误不能接受。"
                )
                minimum_score = st.number_input(
                    "最低通过率（%）",
                    min_value=0.0,
                    max_value=100.0,
                    value=None,
                    key=f"acceptance_min_score_{run_id}",
                )
                minimum_cases = st.number_input(
                    "最低测试题数",
                    min_value=1,
                    value=None,
                    step=1,
                    key=f"acceptance_min_cases_{run_id}",
                )
                generation_limit = st.number_input(
                    "验收每题最大生成 token 数",
                    min_value=1,
                    value=256,
                    step=1,
                    key=f"acceptance_tokens_{run_id}",
                )
                strip_whitespace = st.checkbox(
                    "验收严格匹配时忽略首尾空白", value=True, key=f"acceptance_whitespace_{run_id}"
                )
                # 冻结前算术提示：与验收摘要同源（acceptance_gate_lines 单一来源）；
                # 表单内数值只在提交触发重跑时刷新，冻结后由记录渲染完整算术。
                suite_reference = getattr(session.dataset, "evaluation_suite", None)
                suite_test_count = (
                    (suite_reference.get("case_counts") or {}).get("test")
                    if isinstance(suite_reference, dict)
                    else None
                )
                for gate_line in acceptance_gate_lines(
                    minimum_score / 100 if minimum_score is not None else None,
                    int(minimum_cases) if minimum_cases is not None else None,
                    suite_test_count,
                ):
                    st.caption(gate_line)
                prepare = st.form_submit_button("冻结此模型与业务验收标准")
            if prepare:
                if not standard.strip() or minimum_score is None or minimum_cases is None:
                    st.error("请明确填写业务标准、最低通过率和最低题数，软件不替你设定业务门槛。")
                else:
                    try:
                        acceptance.prepare(
                            service.load(session.session_id),
                            EvaluationModel(
                                "待验收模型", run["model_path"], adapter_path=run["output_dir"]
                            ),
                            EvaluationProtocol(
                                scorer=scorer,
                                **({"custom_scoring": custom_scoring} if custom_scoring else {}),
                                fields=tuple(target.label for target in recipe.targets)
                                if scorer == "json_fields_exact"
                                else (),
                                max_new_tokens=int(generation_limit),
                                strip_whitespace=strip_whitespace,
                            ),
                            {
                                "metric": "pass_rate"
                                if scorer == "custom_rules"
                                else "manual_acceptance_rate"
                                if scorer == "open_review"
                                else "exact_match",
                                "minimum_score": minimum_score / 100,
                                "minimum_cases": int(minimum_cases),
                                "business_standard": standard.strip(),
                            },
                            task_spec=spec,
                        )
                        st.rerun()
                    except (ValueError, RuntimeError, OSError, ImportError) as exc:
                        st.error(str(exc))
        else:
            st.info("当前任务数据已变化；这里只展示该模型已保存的验收记录。")
        try:
            records = acceptance.list_acceptances(session_id=session.session_id)
        except (ValueError, OSError) as exc:
            st.error(f"无法读取最终验收记录：{exc}")
            return
        for record in records:
            if record.get("model", {}).get("adapter_path") != run.get("output_dir"):
                continue
            identity = record["acceptance_id"]
            st.write(f"**最终验收 {identity}**")
            st.write("**已冻结业务标准：** " + record["criteria"]["business_standard"])
            st.write(
                f"最低通过率 {record['criteria']['minimum_score']:.1%} · 最低测试题数 {record['criteria']['minimum_cases']}"
            )
            if record.get("exposure_note"):
                st.warning(record["exposure_note"])
            if record.get("blind_test") is False:
                st.warning("这组题已有暴露记录，本次不能声明为新的盲测。")
            with st.expander(f"验收 {identity} 的冻结模型、题集与评分协议"):
                st.json(
                    {
                        key: record.get(key)
                        for key in ("model", "evaluation_suite", "protocol", "criteria")
                    }
                )
            if record["status"] == "prepared":
                if st.button("按冻结标准执行最终验收", key=f"run_acceptance_{identity}"):
                    try:
                        with st.spinner("正在执行单模型独立测试验收…"):
                            acceptance.run(identity, service.load(session.session_id))
                        st.rerun()
                    except (ValueError, RuntimeError, OSError, ImportError) as exc:
                        st.error(str(exc))
            elif record["status"] == "running":
                st.info("最终验收正在执行。")
            elif record["status"] == "failed":
                st.error(
                    str(
                        record.get("error")
                        or record.get("failure")
                        or "最终验收执行失败，不能认定通过。"
                    )
                )
            result = record.get("result") or {}
            decision = result.get("decision")
            decisions = {
                "passed": "达到已冻结的业务验收标准",
                "failed": "未达到已冻结的业务验收标准",
                "insufficient_evidence": "证据不足，不能确认可交付",
                "pending_review": "等待逐题业务判断",
                "pending_run": "等待按冻结条款执行",
            }
            if decision:
                {
                    "passed": st.success,
                    "failed": st.error,
                    "insufficient_evidence": st.warning,
                    "pending_review": st.info,
                }.get(decision, st.info)(decisions.get(decision, decision))
                if "total_cases" in result:
                    st.write(
                        f"实际验收 {result['total_cases']} 题 · 已通过 {result.get('accepted_cases', '待判断')} 题"
                    )
                if result.get("score") is not None:
                    st.write(f"**实际通过率：** {result['score']:.1%}")
                if "reviewed_cases" in result:
                    st.write(f"已记录 {result['reviewed_cases']} 题判断，剩余题目需核对。")
                if result.get("reason"):
                    st.info(result["reason"])
            from src.workbench.report_summary import summarize_acceptance

            for line in summarize_acceptance(record):
                st.write(line)
            report = record.get("report") or {}
            evaluated_model = (report.get("models") or [{}])[0]
            if report.get("protocol", {}).get("scorer") == "custom_rules":
                scoring_metrics = evaluated_model.get("metrics", {})
                st.dataframe(
                    [
                        {
                            "业务分均值": scoring_metrics.get("business_score"),
                            "业务通过率": scoring_metrics.get("pass_rate"),
                        }
                    ],
                    hide_index=True,
                    width="stretch",
                )
            rows = evaluated_model.get("rows", [])
            if rows:
                row_by_index = {row["index"]: row for row in rows}
                selected = st.selectbox(
                    "查看最终验收题与实际回答",
                    list(row_by_index),
                    format_func=lambda index: f"题目 {index + 1}",
                    key=f"acceptance_row_{identity}",
                )
                row = row_by_index[selected]
                st.write("**输入**")
                st.code(row["prompt"], language=None)
                st.write("**期望或参考答案**")
                st.code(row["expected"], language=None)
                st.write("**实际回答**")
                st.code(row.get("output") or "未生成回答", language=None)
                if row.get("business_score") is not None:
                    st.write(f"**本题业务分：** {row['business_score']}")
                    st.write("**评分理由：** " + row.get("scoring_reason", ""))
                if row.get("error"):
                    st.warning(row["error"])
                if row.get("truncated"):
                    st.warning("回答被截断，不能人工标记通过。")
                saved_judgment = next(
                    (item for item in record.get("decisions", []) if item["index"] == selected),
                    None,
                )
                if saved_judgment:
                    st.info(
                        "已记录"
                        + ("通过" if saved_judgment["decision"] == "accepted" else "不通过")
                        + "："
                        + saved_judgment["reason"]
                    )
                if record["status"] == "needs_business_review" and saved_judgment is None:
                    cannot_accept = (
                        row.get("status") in {"failed", "truncated"}
                        or row.get("truncated")
                        or row.get("output") is None
                    )
                    with st.form(f"acceptance_review_{identity}_{selected}"):
                        review_decision = st.selectbox(
                            "这条回答是否达到冻结业务标准？",
                            ["", "rejected"] if cannot_accept else ["", "accepted", "rejected"],
                            format_func=lambda value: {
                                "": "请选择",
                                "accepted": "通过",
                                "rejected": "不通过",
                            }[value],
                        )
                        reason = st.text_area("这条判断的业务理由")
                        submit_review = st.form_submit_button("保存这条业务判断")
                    if submit_review:
                        if not review_decision or not reason.strip():
                            st.error("请选择通过或不通过，并说明业务理由。")
                        else:
                            try:
                                acceptance.review(
                                    identity,
                                    [
                                        {
                                            "index": selected,
                                            "decision": review_decision,
                                            "reason": reason.strip(),
                                        }
                                    ],
                                )
                                st.rerun()
                            except (ValueError, RuntimeError, OSError) as exc:
                                st.error(str(exc))
            if record.get("decisions"):
                with st.expander(f"验收 {identity} 的人工判断记录"):
                    st.json(record["decisions"])
            st.download_button(
                "下载最终验收记录（含独立测试内容）",
                json.dumps(record, ensure_ascii=False, indent=2),
                file_name=f"acceptance-{identity}.json",
                mime="application/json",
                key=f"download_acceptance_{identity}",
            )


def show_training_recommendations() -> None:
    """Recommend only on request, then prepare a saved plan after business review."""
    from src.workbench.local_models import discover_local_models
    from src.workbench.training_plans import TrainingPlanService

    plan_root = PROJECT_ROOT / "outputs/workbench/training-plans"
    training_root = PROJECT_ROOT / "outputs/workbench/training"
    plans = TrainingPlanService(plan_root, training_root)
    st.write("**让 Agent 推荐训练方案**")
    st.caption(
        "提供一个或多个已准备好的本地模型目录。Agent 将结合业务目标、真实数据统计、模型信息和本机条件，实际检查 tokenizer 后给出建议。不会自动下载模型或启动训练。"
    )
    try:
        discovered = discover_local_models()
    except (ValueError, OSError) as exc:
        discovered = []
        st.warning(f"无法检查本地模型目录：{exc}；可以在高级选项提供已准备好的目录。")
    available = {item["model_path"]: item for item in discovered if item["status"] == "available"}
    selected_models = st.multiselect(
        "本机已准备的候选模型",
        list(available),
        default=[],
        format_func=lambda path: f"{available[path]['name']} · {path}",
        key=f"plan_discovered_{session.session_id}",
    )
    if not available:
        st.info("暂未发现文件完整的本地模型。准备好模型后刷新，或在高级选项填写已有目录。")
    else:
        st.caption(
            "文件完整只表示可以进一步检查；模型是否兼容、训练长度和机器是否适合，仍由方案检查判断。"
        )
    with st.expander("高级：补充本地模型路径与发现详情"):
        candidates = st.text_area(
            "本地候选模型目录（每行一个）",
            placeholder="/path/to/local-model",
            key=f"plan_models_{session.session_id}",
        )
        if discovered:
            st.dataframe(
                [
                    {
                        "模型": item["name"],
                        "目录": item["model_path"],
                        "状态": "文件完整" if item["status"] == "available" else "文件不完整",
                        "问题": "；".join(item.get("issues", [])),
                    }
                    for item in discovered
                ],
                hide_index=True,
                width="stretch",
            )
    authorized = is_local_endpoint(base_url)
    if not authorized:
        authorized = st.checkbox(
            "允许向已选 Agent 服务发送任务、处理方案、数据统计、候选模型配置与本机硬件摘要；不发送训练、开发或测试原文。",
            key=f"plan_consent_{session.session_id}_{base_url}",
        )
    if st.button("让 Agent 推荐训练方案", disabled=not authorized):
        model_paths = list(
            dict.fromkeys(
                [
                    *selected_models,
                    *(line.strip() for line in candidates.splitlines() if line.strip()),
                ]
            )
        )
        if not model_paths:
            st.error("请先从本机候选中选择模型，或在高级选项提供已有模型目录。")
        else:
            try:
                from src.agent.training import recommend_training

                client = CompatibleChatClient(base_url, model, api_key, allow_remote=authorized)
                with st.spinner("Agent 正在核查业务、候选模型和实际 token 预检，形成训练方案…"):
                    recommend_training(
                        service.load(session.session_id),
                        model_paths,
                        client,
                        output_root=plan_root,
                        training_root=training_root,
                    )
                st.rerun()
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                st.error(str(exc))
    try:
        saved_plans = plans.list_plans(session_id=session.session_id)
    except (ValueError, OSError) as exc:
        st.error(f"无法读取已保存训练方案：{exc}")
        return
    labels = {
        "ready": "方案可供确认",
        "needs_data": "需要先完善数据",
        "unsupported": "当前条件不支持",
    }
    for plan in saved_plans:
        proposal = plan["proposal"]
        status = plan["status"]
        plan_id = plan["plan_id"]
        with st.expander(
            f"Agent 训练方案 {plan_id[:12]} · {labels.get(status, status)}", expanded=True
        ):
            {"ready": st.success, "needs_data": st.warning, "unsupported": st.error}.get(
                status, st.info
            )(labels.get(status, status))
            st.write(f"**建议基础模型：** {proposal.get('model_path', '尚未选择')}")
            st.write("**推荐理由**")
            rationale = proposal.get("rationale", [])
            for reason in rationale if isinstance(rationale, list) else [rationale]:
                st.write(reason)
            training_parameters = proposal.get("training_options") or {}
            lora_parameters = proposal.get("lora_options") or {}
            model_parameters = proposal.get("model_options") or {}
            parameters = {
                "每条训练样本最大 token 长度": proposal.get("max_length"),
                "训练轮数": training_parameters.get("num_epochs"),
                "每设备 batch size": training_parameters.get("batch_size"),
                "梯度累积步数": training_parameters.get("gradient_accumulation_steps"),
                "学习率": training_parameters.get("learning_rate"),
                "LoRA rank": lora_parameters.get("r"),
                "模型量化位数": model_parameters.get("quantization_bits") or "不量化",
            }
            st.dataframe(
                [
                    {"建议参数": name, "值": str(value)}
                    for name, value in parameters.items()
                    if value is not None
                ],
                hide_index=True,
                width="stretch",
            )
            for limitation in proposal.get("limitations", []):
                st.info(limitation)
            for question in proposal.get("business_questions", []):
                st.warning(question)
            if plan.get("scope_note"):
                st.caption(plan["scope_note"])
            for issue in (plan.get("probe") or {}).get("issues", []):
                {"blocking": st.error, "warning": st.warning, "info": st.info}.get(
                    issue.get("severity"), st.info
                )(issue.get("message", ""))
            with st.expander(f"方案 {plan_id[:12]} 的候选事实与真实检查记录"):
                st.json(plan.get("probe") or {})
                st.json(proposal)
                st.json(plan.get("context", {}))
                from src.workbench.report_summary import summarize_tool_trace

                for line in summarize_tool_trace(plan.get("trace", []), "方案"):
                    st.write(line)
                st.json(plan.get("trace", []))
            if plan.get("run_id"):
                st.info(f"已准备训练 {plan['run_id']}，请在下方记录核对预检后启动。")
            elif status == "ready":
                st.caption(
                    "确认会按保存的方案重新核对数据和模型，并准备训练；启动仍需在训练记录中点击。数据或模型已改变时需要重新推荐。"
                )
                if st.button("确认推荐方案并准备训练", key=f"prepare_plan_{plan_id}"):
                    try:
                        plans.prepare(plan_id, service.load(session.session_id))
                        st.rerun()
                    except (ValueError, RuntimeError, OSError, ImportError) as exc:
                        st.error(str(exc))


def show_adapter_report(report: dict, adapter: dict | None, *, title: str) -> None:
    """Display saved isolation evidence; execution belongs to Agent/service tools."""
    st.subheader(title)
    validation = report.get("validation", report)
    status = validation.get("status", "unknown")
    messages = {
        "passed": "适配规则的隔离测试已通过，请继续核对实际转换结果是否符合业务含义。",
        "failed": "适配规则的隔离测试未通过，不能据此认定转换可用。",
        "unavailable": "当前隔离后端不可用，未改为直接在宿主运行代码。",
    }
    renderer = (
        st.success
        if status == "passed"
        else st.error
        if status in {"failed", "unavailable"}
        else st.warning
    )
    renderer(messages.get(status, "尚未获得有效的隔离验证结果。"))
    backend = validation.get("backend") or "不可用或未记录"
    st.caption(f"实际隔离后端：{backend}")
    if validation.get("error") or report.get("error"):
        st.error(validation.get("error") or report["error"])
    if adapter:
        st.write("**新增字段：** " + "、".join(adapter.get("new_columns", [])))
    cases = validation.get("cases", [])
    if cases:
        st.dataframe(
            [
                {
                    "测试": case.get("name", ""),
                    "类型": {"business": "业务期望样例", "counterexample": "反例"}.get(
                        case.get("kind"), "未标记"
                    ),
                    "结果": "通过" if case.get("passed") is True else "未通过",
                    "说明": case.get("error") or "",
                }
                for case in cases
            ],
            hide_index=True,
            width="stretch",
        )
    with st.expander(f"{title}：隔离记录与来源"):
        st.json({key: validation.get(key) for key in ("source_digest", "cases_digest", "limits")})
        st.json({"spec_digest": report.get("spec_digest"), "origins": report.get("origins", {})})
    if adapter:
        with st.expander(f"{title}：处理规则与验收样例（可选查看）"):
            st.caption(
                "这里提供规则留档；后续确认针对真实输入和答案的业务含义，不要求人工审查代码。"
            )
            st.code(adapter.get("source_code", ""), language="python")
            st.json({"config": adapter.get("config", {}), "examples": adapter.get("examples", [])})


def show_business_comparison(report, *, key: str) -> None:
    """Show complete stored outputs with the denominator and scoring limits visible."""
    if report.status == "completed":
        st.info("开发集对照已完成；仍需按业务标准判断是否达到目标。")
    elif report.status == "completed_with_failures":
        st.warning("对照已完成，包含失败或截断记录；它们保留在评分分母中。")
    else:
        st.error("评测尚未完整完成，不能作为完整的模型对照结果。")
    if report.protocol["scorer"] == "open_review":
        st.warning("这是开放任务，尚无业务评分规则；以下输出待业务核对，没有认定微调效果通过。")
    for note in report.notes:
        st.caption(note)
    summaries = []
    from src.workbench.evaluation_diagnostics import (
        count_instruction_echo,
        dominant_output_models,
        echo_triage_lines,
        high_truncation_models,
    )

    truncation_models = high_truncation_models(report.models)

    for model_result in report.models:
        metrics = model_result["metrics"]
        rows = model_result["rows"]
        # 指令回声：输出复述提示文本（含改述式长片段共享），诊断口径与服务层一致。
        echo_count = count_instruction_echo(rows)
        summaries.append(
            {
                "模型": model_result["label"],
                "开发集总数": metrics["total"],
                "已评分": metrics["scored"],
                "生成失败": sum(row["status"] == "failed" for row in rows),
                "输出截断": sum(row["status"] == "truncated" for row in rows),
                "复述指令": echo_count,
                "待业务评分": sum(row["status"] == "needs_business_review" for row in rows),
                **(
                    {
                        "业务分均值": metrics.get("business_score"),
                        "业务通过率": metrics.get("pass_rate"),
                    }
                    if report.protocol["scorer"] == "custom_rules"
                    else {"严格准确率": metrics["exact_match"]}
                ),
            }
        )
        if metrics.get("field_accuracy"):
            st.write(f"**{model_result['label']} · 结构化字段准确率**")
            st.json(metrics["field_accuracy"])
        for error in model_result.get("errors", []):
            st.error(f"{model_result['label']}：{error}")
    # 回声分流引导与 CLI 对照摘要同源（echo_triage_lines 单一来源）。
    for model_result in report.models:
        for line in echo_triage_lines(
            model_result["label"], model_result["rows"], protocol=report.protocol
        ):
            st.warning(line)
    if truncation_models:
        limit = report.protocol.get("max_new_tokens")
        st.warning(
            "检测到高比例输出截断（"
            + "、".join(f"{label} {count} 题" for label, count in truncation_models)
            + "）：大量输出触及生成长度上限。可核查：max_new_tokens"
            + (f"（当前 {limit}）" if limit is not None else "")
            + "是否小于最短合法答案；输出是否在重复生成或缺少停止标记。"
            "触及上限不等于只需增加长度，这是观察事实，原因仍需核查。"
        )
    dominant_models = dominant_output_models(report.models)
    if dominant_models:
        dominant_parts = []
        for label, count, generated, top_output in dominant_models:
            clipped = top_output if len(top_output) <= 24 else top_output[:24] + "…"
            dominant_parts.append(f"{label} 有 {count}/{generated} 条输出完全相同（{clipped}）")
        st.warning(
            "观察到输出高度重复（"
            + "、".join(dominant_parts)
            + "）：模型在反复输出同一答案。请对照开发集答案分布——分布本身集中时，"
            "模型可能只是复述多数类，不一定是学坏了；分布不集中时，逐题查看完整输出。"
            "这是观察事实，原因仍需核查。"
        )
    st.dataframe(summaries, hide_index=True, width="stretch")
    from src.workbench.report_summary import summarize_comparison

    with st.expander("用大白话解读这份对照(观察事实,不是达标结论)"):
        for line in summarize_comparison(report):
            st.write(line)
    all_rows = {row["index"]: row for model_result in report.models for row in model_result["rows"]}
    if all_rows:
        selected = st.selectbox(
            "逐样本查看完整输入、期望和输出",
            sorted(all_rows),
            format_func=lambda index: (
                f"样本 {index + 1} · {all_rows[index]['source'].get('source_row_id', '')}"
            ),
            key=f"eval_row_{key}_{report.evaluation_id}",
        )
        reference = all_rows[selected]
        st.write("**完整模型输入**")
        st.code(reference["prompt"], language=None)
        st.write("**期望答案**")
        st.code(reference["expected"], language=None)
        columns = st.columns(max(1, len(report.models)))
        states = {
            "failed": "生成或评分失败",
            "truncated": "输出截断",
            "needs_business_review": "待业务评分",
            "scored": "已评分",
        }
        for column, model_result in zip(columns, report.models):
            row = next((row for row in model_result["rows"] if row["index"] == selected), None)
            with column:
                st.write(f"**{model_result['label']}**")
                if row is None:
                    st.warning("该模型没有本样本结果。")
                    continue
                st.caption(states.get(row["status"], row["status"]))
                st.code(row["output"] if row["output"] is not None else "未生成输出", language=None)
                if row.get("business_score") is not None:
                    st.write(f"**业务分：** {row['business_score']}")
                    st.write("**评分理由：** " + row.get("scoring_reason", ""))
                if row["correct"] is not None:
                    st.write(
                        (
                            "达到单题业务通过标准"
                            if report.protocol["scorer"] == "custom_rules"
                            else "与期望一致"
                        )
                        if row["correct"]
                        else "未达到当前评分规则"
                    )
                if row.get("error"):
                    st.warning(row["error"])
                if row.get("field_scores"):
                    st.json(row["field_scores"])
    with st.expander("评测数据版本与评分协议"):
        st.json(
            {
                "dataset": report.dataset,
                "protocol": report.protocol,
                "comparison_key": report.comparison_key,
            }
        )
    st.download_button(
        "下载完整逐样本对照报告",
        json.dumps(asdict(report), ensure_ascii=False, indent=2),
        file_name=f"comparison-{report.evaluation_id}.json",
        mime="application/json",
        key=f"eval_download_{key}_{report.evaluation_id}",
    )
    from src.agent.evaluation import assess_evaluation, load_assessments

    st.write("**Agent 结果解读与下一步**")
    st.caption(
        "Agent 会结合业务目标、当前方案、训练配置与监督检查、实际评测输出和坏例分析。解读给出事实、待核查原因和建议，不自动修改数据或启动下一轮训练。"
    )
    evaluation_root = PROJECT_ROOT / "outputs/workbench/evaluations"
    current_report = session.dataset is not None and session.dataset.version == report.dataset.get(
        "version"
    )
    report_authorized = True
    if not is_local_endpoint(base_url):
        report_authorized = (
            st.checkbox(
                "允许向已选 Agent 服务发送本次目标、方案、训练配置与检查摘要，以及实际评测输出和坏例，用于结果解读。",
                key=f"eval_agent_send_{key}_{report.evaluation_id}_{base_url}",
            )
            and allow_remote
        )
        if not allow_remote:
            st.caption("请先在上方「分析模型设置」允许向所选服务发送业务资料。")
    if st.button(
        "让 Agent 分析结果与下一步",
        key=f"assess_{key}_{report.evaluation_id}",
        disabled=not report_authorized or not current_report,
    ):
        try:
            client = CompatibleChatClient(
                base_url, model, api_key, allow_remote=report_authorized and allow_remote
            )
            with st.spinner("Agent 正在查看真实坏例、核查证据并形成下一步建议…"):
                assess_evaluation(
                    report, service.load(session.session_id), client, output_root=evaluation_root
                )
            st.rerun()
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    assessments = []
    try:
        assessments = load_assessments(evaluation_root, report.evaluation_id)
        for record in assessments:
            if record.get("session_id") != session.session_id:
                continue
            assessment = record["assessment"]
            with st.expander(f"已保存的 Agent 解读 · {record.get('model', '')}", expanded=True):
                st.write(assessment["summary"])
                if record.get("session_revision") != session.revision:
                    st.caption(
                        "这是此前业务方案下的解读；当前任务已发生变化，需结合新方案重新核查。"
                    )
                st.write("**有证据的观察**")
                for observation in assessment["observations"]:
                    st.write(observation["statement"])
                    if observation.get("evidence_ids"):
                        st.caption("证据：" + "、".join(observation["evidence_ids"]))
                for hypothesis in assessment.get("hypotheses", []):
                    st.write("**待核查原因：** " + hypothesis["statement"])
                    st.write("**验证方式：** " + hypothesis["verification"])
                    if hypothesis.get("evidence_ids"):
                        st.caption("证据：" + "、".join(hypothesis["evidence_ids"]))
                decisions = {
                    "inspect_data": "核查数据",
                    "revise_pipeline": "修订处理方案",
                    "inspect_training": "核查训练行为",
                    "collect_evidence": "补充证据",
                    "business_review": "业务核对",
                }
                st.write(
                    "**建议优先处理：** "
                    + decisions.get(assessment["decision"], assessment["decision"])
                )
                for step in assessment["next_steps"]:
                    st.write(f"- {step}")
                for limitation in assessment["limitations"]:
                    st.info(limitation)
                for question in assessment.get("business_questions", []):
                    st.warning(question)
                with st.expander("本次解读核查记录"):
                    st.json(record.get("tool_trace", []))
                from src.workbench.report_summary import summarize_assessment

                for line in summarize_assessment(record):
                    st.write(line)
    except (ValueError, OSError) as exc:
        st.error(f"无法读取已保存解读：{exc}")
    if current_report:
        with st.expander("将结果转成下一轮改进假设"):
            st.caption(
                "先写清要验证的原因、预期业务结果及实际变更，再确认执行。后续沿用固定开发/测试题目，从同一个基础模型重新微调，父轮模型保留作对照。"
            )
            suggested = assessments[0]["assessment"] if assessments else {}
            hypotheses = suggested.get("hypotheses", [])
            with st.form(f"iteration_proposal_{key}_{report.evaluation_id}"):
                hypothesis = st.text_area(
                    "本轮要验证的改进假设", value=hypotheses[0]["statement"] if hypotheses else ""
                )
                expected_outcome = st.text_area(
                    "希望在固定题集上观察到什么变化？",
                    placeholder="例如：原来错误的类别题减少，同时其他类别不退步。",
                )
                changes = st.text_area(
                    "计划具体修改什么？", value="\n".join(suggested.get("next_steps", []))
                )
                data_change = st.checkbox("本轮会修改业务数据或处理规则")
                change_training = st.checkbox("本轮需要调整训练参数（否则继承父轮）")
                epochs_override = st.number_input(
                    "改进轮次训练轮数", min_value=1, value=1, step=1, disabled=not change_training
                )
                lr_override = st.number_input(
                    "改进轮次学习率",
                    min_value=0.0000001,
                    value=0.0002,
                    format="%.7f",
                    disabled=not change_training,
                )
                max_length_override = st.number_input(
                    "改进轮次最大 token 长度",
                    min_value=1,
                    value=1024,
                    step=1,
                    disabled=not change_training,
                )
                propose_iteration = st.form_submit_button("保存改进提案，进入确认")
            if propose_iteration:
                try:
                    iteration_service.propose(
                        service.load(session.session_id),
                        parent_run_id=key,
                        evaluation_id=report.evaluation_id,
                        hypothesis=hypothesis,
                        expected_outcome=expected_outcome,
                        changes=changes,
                        data_change=data_change,
                        training_options={
                            "num_epochs": int(epochs_override),
                            "learning_rate": float(lr_override),
                        }
                        if change_training
                        else None,
                        max_length=int(max_length_override) if change_training else None,
                    )
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))


def select_task() -> None:
    st.session_state["agent_remote_consent"] = False
    selected = st.session_state.get("intake_select")
    if selected:
        st.session_state["intake_id"] = selected
    else:
        st.session_state.pop("intake_id", None)


def new_task() -> None:
    st.session_state["agent_remote_consent"] = False
    st.session_state.pop("intake_id", None)
    st.session_state["intake_select"] = ""


SHEET_INPUT_LABEL = "Excel 工作表（留空读第一个）"


def excel_sheet_input(upload, *, key: str) -> str | None:
    """Excel 上传时提供可选 sheet 选择；其他情况返回 None，读取行为不变。

    返回值直接透传 read_source 的 sheet 参数：按名称或 1 起始序号指定，留空读
    第一个 sheet。调用方必须把上传控件放在表单外——表单内部件要到提交才提交
    值，放里面就无法在提交前按上传的文件类型显示这个选择。
    """
    if upload is None or not upload.name.lower().endswith((".xlsx", ".xls")):
        return None
    return st.text_input(
        SHEET_INPUT_LABEL, placeholder="按名称如 员工表，或 1 起始序号如 2", key=key
    )


with st.sidebar:
    st.subheader("已有数据任务")
    sessions = service.list_sessions()
    labels = {s.session_id: f"{s.goal[:35]} · {s.session_id[:6]}" for s in sessions}
    st.selectbox(
        "选择任务",
        ["", *labels],
        format_func=lambda key: labels.get(key, "新建任务"),
        key="intake_select",
        on_change=select_task,
    )
    st.button("新建数据任务", on_click=new_task)
    with st.expander("全部任务停点快照"):
        # 北极星「度量体系·过程漏斗」:跨任务的只读停点计数。与 CLI
        # funnel-report 同一份 collect_funnel + summarize_funnel
        # (单一来源,页面与 CLI 不各说各话)。
        from src.workbench.funnel_report import collect_funnel, summarize_funnel

        report = collect_funnel(
            PROJECT_ROOT / "outputs" / "workbench" / "intake",
            PROJECT_ROOT / "outputs" / "workbench" / "training",
            PROJECT_ROOT / "outputs" / "workbench" / "evaluations",
            PROJECT_ROOT / "outputs" / "workbench" / "acceptance",
            PROJECT_ROOT / "outputs" / "workbench" / "iterations",
        )
        for line in summarize_funnel(report):
            st.write(line)

base_url, model, api_key, allow_remote = render_agent_settings(
    PROJECT_ROOT / "outputs" / "workbench" / "agent-settings.json"
)

if "intake_id" not in st.session_state:
    # 上传控件放在表单外：表单内部件要到提交才提交值，放里面就无法在提交前
    # 按上传的文件类型显示 sheet 选择。
    upload = st.file_uploader("提供 CSV、Excel 或 JSONL", type=["csv", "xlsx", "xls", "jsonl"])
    with st.form("new_intake"):
        goal = st.text_area(
            "希望模型完成什么业务工作？",
            placeholder="例如：根据客户首次咨询判断售后问题类型。历史工单里的处理结果可能不是我想要的标签。",
        )
        description = st.text_area(
            "这些数据是什么？有哪些已知情况？",
            placeholder="例如：一行一个工单，描述来自客户，类别由人工审核。有些记录缺类别。",
        )
        scope_label = st.radio("这份文件的用途", ["用于理解结构的样例", "本次任务的全量数据"])
        with st.expander("文件读取设置（通常自动识别即可）"):
            encoding = st.text_input("编码（留空自动识别）", placeholder="utf-8-sig / gb18030")
            delimiter_label = st.selectbox("CSV 分隔符", ["自动", "逗号", "分号", "Tab", "竖线"])
            sheet = excel_sheet_input(upload, key="new_intake_sheet")
        create = st.form_submit_button("读取数据并开始", type="primary")
    if create:
        if upload is None:
            st.error("请提供一份数据样例；不需要先整理成训练格式。")
        else:
            try:
                session = service.create(
                    goal,
                    upload.name,
                    upload.getvalue(),
                    data_description=description,
                    scope="sample" if scope_label.startswith("用于") else "full",
                    encoding=encoding.strip() or None,
                    delimiter={"逗号": ",", "分号": ";", "Tab": "\t", "竖线": "|"}.get(
                        delimiter_label
                    ),
                    sheet=(sheet or "").strip() or None,
                )
                st.session_state["intake_id"] = session.session_id
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    # 内置演示任务:只代替「找文件 + 填表」的冷启动;创建后的每一道关卡
    # 与真实任务完全相同(单一来源 src/workbench/demo_task.py)。
    demo_pair = demo_sample(PROJECT_ROOT)
    if demo_pair is not None:
        with st.expander(DEMO_SAMPLE_ENTRY_LABEL):
            st.caption(
                "演示数据是虚构的售后工单（样例 2 条、配套全量 10 条），"
                "与《用户试用记录》的试点任务同一份文件。创建后的每一步"
                "（分析、预览核对、对比核验、盲标核验）与真实任务完全相同，"
                "没有预设结论。"
            )
            if st.button(DEMO_SAMPLE_BUTTON):
                try:
                    demo_session = service.create(
                        DEMO_GOAL,
                        demo_pair[0],
                        demo_pair[1],
                        data_description=DEMO_DESCRIPTION,
                        scope="sample",
                    )
                    st.session_state["intake_id"] = demo_session.session_id
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
    st.stop()

session = service.load(st.session_state["intake_id"])
iteration_service = IterationService(
    PROJECT_ROOT / "outputs/workbench/iterations",
    PROJECT_ROOT / "outputs/workbench/training",
    PROJECT_ROOT / "outputs/workbench/evaluations",
)
iterations = iteration_service.list_iterations(session_id=session.session_id)
execution_service = None
execution_records = {}
execution_managed_states = {
    "queued",
    "materializing",
    "preparing",
    "awaiting_warning_ack",
    "training",
    "waiting_for_release",
    "evaluating",
}
if iterations:
    from src.workbench.iteration_execution import IterationExecutionService

    execution_service = IterationExecutionService(
        PROJECT_ROOT / "outputs/workbench/iterations/executions",
        service.root,
        PROJECT_ROOT / "outputs/workbench/iterations",
        PROJECT_ROOT / "outputs/workbench/training",
        PROJECT_ROOT / "outputs/workbench/evaluations",
        project_root=PROJECT_ROOT,
    )
    for item in iterations:
        try:
            execution_records[item["iteration_id"]] = execution_service.get(item["iteration_id"])
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"无法读取本轮执行记录：{exc}")
            execution_records[item["iteration_id"]] = {"status": "unavailable", "message": str(exc)}
active_iteration = next((item for item in iterations if item["status"] == "confirmed"), None)
suite_service = EvalSuiteService(PROJECT_ROOT / "outputs/workbench/evaluation-suites")
available_suites = {ref["suite_id"]: ref for ref in suite_service.list_suites()}
if active_iteration:
    available_suites[active_iteration["evaluation_suite"]["suite_id"]] = active_iteration[
        "evaluation_suite"
    ]
st.subheader(session.goal)
st.caption(
    f"{session.source.name} · {len(session.source.rows)} 条记录 · 任务 {session.session_id[:8]}"
)
st.info(session.profile["scope_note"])
show_fact_notes(session.profile)
if iterations:
    st.subheader("改进轮次与当前下一步")
    iteration_states = {
        "proposed": "待确认假设",
        "confirmed": "待准备固定题集数据",
        "preparing": "准备训练中",
        "prepared": "待启动训练",
        "blocked": "准备有阻断问题",
        "running": "等待训练与同题评测",
        "evaluated": "待业务决策",
        "decided": "已记录决策",
    }
    for iteration in iterations:
        identity = iteration["iteration_id"]
        execution = execution_records.get(identity)
        execution_managed = bool(execution and execution["status"] in execution_managed_states)
        with st.expander(
            f"改进 {identity[:11]} · {iteration_states.get(iteration['status'], iteration['status'])}",
            expanded=iteration["status"] != "decided",
        ):
            st.write("**改进假设：** " + iteration["hypothesis"])
            st.write("**预期变化：** " + iteration["expected_outcome"])
            st.write("**确认范围：** " + iteration["changes"])
            st.caption(
                f"父轮 {iteration['parent_run_id']} · 固定题集 {iteration['evaluation_suite']['suite_id'][:12]} · 训练从同一基座重新开始"
            )
            with st.expander("继承配置与确认的覆盖"):
                st.json(iteration.get("options", {}))
            if execution:
                execution_labels = {
                    "queued": "已受理，等待后台执行",
                    "materializing": "正在准备固定题集数据",
                    "preparing": "正在检查训练方案",
                    "awaiting_warning_ack": "预检提示需要您核对",
                    "training": "正在训练",
                    "waiting_for_release": "等待训练释放资源",
                    "evaluating": "正在比较开发集结果",
                    "completed": "开发集对照已完成",
                    "blocked": "存在需要处理的问题",
                    "failed": "执行失败",
                    "stopped": "已停止",
                }
                st.write(
                    "**本轮后台进度：** "
                    + execution_labels.get(execution["status"], execution["status"])
                )
                if execution.get("message"):
                    if execution["status"] in {"failed", "blocked", "unavailable"}:
                        st.error(execution["message"])
                    else:
                        st.info(execution["message"])
                issues = execution.get("issues") or (execution.get("preflight") or {}).get(
                    "issues", []
                )
                for issue in issues:
                    if isinstance(issue, dict):
                        {"blocking": st.error, "warning": st.warning, "info": st.info}.get(
                            issue.get("severity"), st.warning
                        )(issue.get("message", str(issue)))
                    else:
                        st.warning(str(issue))
                if execution.get("run_id"):
                    st.caption(f"对应训练：{execution['run_id']}")
                if execution.get("evaluation_id"):
                    st.caption(
                        f"开发集报告：{execution['evaluation_id']}；请核对结果后决定下一步。"
                    )
                from src.workbench.report_summary import summarize_execution

                for line in summarize_execution(execution):
                    st.write(line)
                if st.button("刷新本轮执行状态", key=f"refresh_execution_{identity}"):
                    st.rerun()
                if execution["status"] == "awaiting_warning_ack":
                    acknowledged = st.checkbox(
                        "已核对上述预检提示，继续按确认方案执行。",
                        key=f"execution_warning_ack_{identity}",
                    )
                    if st.button(
                        "核对后继续到开发集对照",
                        key=f"resume_execution_{identity}",
                        disabled=not acknowledged,
                    ):
                        try:
                            execution_service.start(
                                identity,
                                service.load(session.session_id),
                                acknowledge_warnings=True,
                            )
                            st.rerun()
                        except (ValueError, RuntimeError, OSError, ImportError) as exc:
                            st.error(str(exc))
                if execution_managed and st.button(
                    "停止本轮后台执行", key=f"stop_execution_{identity}"
                ):
                    try:
                        execution_service.stop(identity)
                        st.rerun()
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.error(str(exc))
            if iteration["status"] == "confirmed" and execution is None:
                data_confirmed = bool(
                    session.full_data
                    and session.full_data.status == "confirmed"
                    and session.full_data.confirmed_revision is not None
                )
                if data_confirmed:
                    st.write(
                        "按上方已确认范围，沿用固定题集、训练配置和父轮对照标准，自动完成数据准备、训练及基座/父轮/本轮开发集对照。"
                    )
                    st.caption(
                        "后台独立推进，关闭页面不影响执行；遇到需核对的预检提示会暂停。对照完成后由您判断是否采用，不执行最终验收。"
                    )
                    independent = bool(
                        session.analysis
                        and session.analysis.recipe
                        and session.analysis.recipe.group_columns
                    )
                    if not independent:
                        independent = st.checkbox(
                            "已确认本轮每行代表独立业务对象。",
                            key=f"execution_independent_{identity}",
                        )
                    if st.button(
                        "按确认方案执行到开发集对照",
                        key=f"execute_iteration_{identity}",
                        disabled=not independent,
                        type="primary",
                    ):
                        try:
                            execution_service.start(
                                identity,
                                service.load(session.session_id),
                                independent_rows_confirmed=independent,
                            )
                            st.rerun()
                        except (ValueError, RuntimeError, OSError, ImportError) as exc:
                            st.error(str(exc))
            if iteration["status"] == "proposed" and st.button(
                "确认本轮假设与变更范围", key=f"confirm_iteration_{identity}"
            ):
                try:
                    iteration_service.confirm(identity, service.load(session.session_id))
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))
            if iteration["status"] == "confirmed" and not execution_managed:
                if iteration.get("data_change"):
                    st.info(
                        "下一步：通过下方资料入口修订训练资料，重新分析、确认并验证全量；物化时将自动使用本轮固定题集。原开发/测试题不能改写或进入训练。"
                    )
                    revision_authorized = is_local_endpoint(base_url)
                    if not revision_authorized:
                        revision_authorized = (
                            st.checkbox(
                                "允许向已选 Agent 服务发送当前业务资料、确认方向，以及父轮实际评测输出与坏例，用于修订数据方案。",
                                key=f"revision_consent_{identity}_{base_url}",
                            )
                            and allow_remote
                        )
                    if st.button(
                        "让 Agent 按确认方向修改数据方案",
                        key=f"revise_iteration_{identity}",
                        disabled=not revision_authorized,
                    ):
                        try:
                            from src.agent.revisions import revise_data_for_iteration

                            revision_client = CompatibleChatClient(
                                base_url, model, api_key, allow_remote=revision_authorized
                            )
                            with st.spinner("Agent 正在结合父轮坏例修改方案并执行真实数据预览…"):
                                revise_data_for_iteration(
                                    service,
                                    iteration_service,
                                    identity,
                                    revision_client,
                                    expected_revision=session.revision,
                                )
                            st.rerun()
                        except (ValueError, RuntimeError, OSError, ImportError) as exc:
                            st.error(str(exc))
                else:
                    st.info(
                        "下一步：沿已确认数据重新物化，绑定本轮固定题集；随后使用继承配置准备训练。"
                    )
                with st.expander("高级：分步准备本轮数据与训练"):
                    attached_suite = (
                        getattr(session.dataset, "evaluation_suite", None)
                        if session.dataset
                        else None
                    )
                    if (
                        attached_suite != iteration["evaluation_suite"]
                        and session.full_data
                        and session.full_data.confirmed_revision is not None
                    ):
                        independent_iteration_rows = False
                        if (
                            session.analysis
                            and session.analysis.recipe
                            and not session.analysis.recipe.group_columns
                        ):
                            independent_iteration_rows = st.checkbox(
                                "已确认本轮每行是独立业务对象。",
                                key=f"independent_iteration_{identity}",
                            )
                        if st.button(
                            "用本轮固定题集准备数据版本", key=f"materialize_iteration_{identity}"
                        ):
                            try:
                                service.materialize_dataset(
                                    session.session_id,
                                    session.revision,
                                    evaluation_suite=iteration["evaluation_suite"],
                                    independent_rows_confirmed=independent_iteration_rows,
                                )
                                st.rerun()
                            except (ValueError, RuntimeError, OSError) as exc:
                                st.error(str(exc))
                    if st.button(
                        "按已确认范围准备下一轮训练",
                        key=f"prepare_iteration_{identity}",
                        disabled=attached_suite != iteration["evaluation_suite"],
                    ):
                        try:
                            iteration_service.prepare(identity, service.load(session.session_id))
                            st.rerun()
                        except (ValueError, RuntimeError, OSError, ImportError) as exc:
                            st.error(str(exc))
            if iteration["status"] == "evaluated" and not iteration.get("decision"):
                results = iteration.get("results") or []
                if results:
                    st.write("**同题三模型结果（固定开发集，同一评分协议）：**")
                    st.dataframe(
                        [
                            {
                                "模型": item["label"],
                                **{
                                    key: value
                                    for key, value in (item.get("metrics") or {}).items()
                                    if isinstance(value, (int, float))
                                    and not isinstance(value, bool)
                                },
                            }
                            for item in results
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                    st.caption(
                        "分数未提高也可能是有价值证据；截断与生成失败的行都在分母内，不作美化。"
                    )
                if iteration.get("evaluation_id"):
                    st.caption(f"开发集报告：{iteration['evaluation_id']}")
                st.write(
                    "请核对结果后作出业务决策；决策与理由记录在本轮，作为采用、继续或停止的依据。"
                )
                from src.workbench.report_summary import (
                    ITERATION_DECISION_NAMES,
                    summarize_iteration,
                )

                for line in summarize_iteration(iteration):
                    st.write(line)
                decision_label = st.radio(
                    "本轮决策",
                    list(ITERATION_DECISION_NAMES.values()),
                    key=f"decision_choice_{identity}",
                )
                decision_reason = st.text_area(
                    "业务理由（必填）",
                    key=f"decision_reason_{identity}",
                )
                decision_map = {name: code for code, name in ITERATION_DECISION_NAMES.items()}
                if st.button(
                    "记录本轮决策",
                    key=f"decide_{identity}",
                    type="primary",
                    disabled=not decision_reason.strip(),
                ):
                    try:
                        iteration_service.decide(
                            identity, decision_map[decision_label], decision_reason.strip()
                        )
                        st.rerun()
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.error(str(exc))
            if iteration.get("decision"):
                from src.workbench.report_summary import (
                    ITERATION_DECISION_NAMES,
                    summarize_iteration,
                )

                st.success(
                    f"已记录决策："
                    f"{ITERATION_DECISION_NAMES.get(iteration['decision'], iteration['decision'])}"
                    f" — {iteration.get('decision_reason', '')}"
                )
                for line in summarize_iteration(iteration):
                    st.write(line)
            if iteration.get("data_revision"):
                revision = iteration["data_revision"]
                component_names = {
                    "recipe": "字段与训练样本生成规则",
                    "composition": "多源资料组合",
                    "adapter": "受限适配规则",
                }
                st.write(
                    "**本次方案改动：** "
                    + (
                        "、".join(
                            component_names.get(item, item)
                            for item in revision.get("changed_components", [])
                        )
                        or "方案组件未变化"
                    )
                )
                st.info(
                    "请在下方查看真实转换预览，核对业务含义后确认，再验证全量并沿固定题集准备数据。"
                )
                with st.expander(f"本轮 {identity[:11]} 修改前后方案与工具记录"):
                    st.write("修改前")
                    st.json(revision.get("before", {}))
                    st.write("修改后")
                    st.json(revision.get("after", {}))
                    from src.workbench.report_summary import summarize_tool_trace

                    for line in summarize_tool_trace(revision.get("tool_trace", []), "修订"):
                        st.write(line)
                    st.json(revision.get("tool_trace", []))
            if iteration.get("failure"):
                st.error(str(iteration["failure"]))
            if iteration["status"] in {"prepared", "running"} and not execution_managed:
                st.info(
                    "下一步：在下方对应训练记录中启动或查看进度；成功后运行基座、父轮与本轮的固定题集对照。"
                )
original_sources = session.sources or {"main": session.source}
with st.expander("原始资料与补充文件", expanded=len(original_sources) > 1):
    st.caption(
        "按资料名称保留原文件。说明各表的业务含义和关系，Agent 会检查关联方式，无需自行拼成训练表。"
    )
    st.dataframe(
        [
            {
                "资料名称": alias,
                "文件": source.name,
                "范围": "全量" if source.scope == "full" else "样例",
                "记录数": len(source.rows),
                "字段": "、".join(source.columns),
            }
            for alias, source in original_sources.items()
        ],
        hide_index=True,
        width="stretch",
    )
    # 每份资料如实标注自己的 Excel 事实(读取范围/合并/公式/隐藏);多资料时带名称前缀
    for alias, source in original_sources.items():
        show_fact_notes(source, alias=alias if len(original_sources) > 1 else "")
    # 上传控件放在表单外：表单内部件要到提交才提交值，放里面就无法在提交前
    # 按上传的文件类型显示 sheet 选择。
    source_upload = st.file_uploader("上传补充原始资料", type=["csv", "xlsx", "xls", "jsonl"])
    with st.form(f"source_upload_{session.session_id}"):
        source_alias = st.text_input(
            "补充资料名称", placeholder="例如 labels、orders；main 是首次上传的资料"
        )
        source_description = st.text_area(
            "这份资料的用途和关联关系",
            placeholder="例如：质检人员审核的类别，通过工单编号与 main 对应。",
        )
        source_scope = st.radio("补充资料范围", ["样例", "全量"])
        source_sheet = excel_sheet_input(source_upload, key=f"source_sheet_{session.session_id}")
        replace_source = st.checkbox("若资料名称已存在，替换该份资料并重新分析。")
        add_source = st.form_submit_button("保存补充资料")
    if add_source:
        if source_upload is None:
            st.error("请上传这份原始资料。")
        elif source_alias.strip() in original_sources and not replace_source:
            st.error("该资料名称已存在；请选择新名称，或明确替换。")
        else:
            try:
                updated = service.add_source(
                    session.session_id,
                    session.revision,
                    source_alias.strip(),
                    source_upload.name,
                    source_upload.getvalue(),
                    scope="full" if source_scope == "全量" else "sample",
                    sheet=(source_sheet or "").strip() or None,
                )
                if source_description.strip():
                    service.answer(
                        updated.session_id,
                        f"资料 {source_alias.strip()}：{source_description.strip()}",
                    )
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))

if session.composition_report:
    st.subheader("资料组合与处理结果")
    st.caption("以下为 Agent 方案在真实资料上的执行结果；原始文件仍单独保留。")
    st.dataframe(session.composition_report["steps"], hide_index=True, width="stretch")
    for issue in session.composition_report["issues"]:
        {"blocking": st.error, "review": st.warning, "info": st.info}[issue["severity"]](
            issue["message"]
        )
    with st.expander("组合行的原始来源"):
        st.json(session.composition_report["origins"])

if session.adapter_report:
    show_adapter_report(
        session.adapter_report,
        session.analysis.adapter if session.analysis else None,
        title="受限适配验证结果",
    )

with st.expander("本地数据事实与原始行", expanded=session.analysis is None):
    facts = [
        {
            "字段": name,
            "缺失": info["missing_count"],
            "不同值": info["distinct_count"],
            "最长字符数": info["max_length"],
            "原始类型": ", ".join(info["value_types"]),
        }
        for name, info in session.profile["columns"].items()
    ]
    st.dataframe(facts, hide_index=True, width="stretch")
    st.dataframe(
        [
            {
                "行ID": row.row_id,
                "文件行": row.line,
                **{
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in row.values.items()
                },
            }
            for row in session.source.rows[:30]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "这里展示当前处理阶段的数据；原始资料分别保留在上方资料列表中。统计范围仅为已提供的资料。"
    )

if session.analysis:
    analysis = session.analysis
    st.subheader("业务理解")
    st.write(f"**实际使用输入：** {analysis.task.usage_input}")
    st.write(f"**希望输出：** {analysis.task.desired_output}")
    st.write(f"**每行含义：** {analysis.task.row_meaning}")
    st.write(f"**监督答案来源：** {analysis.task.supervision_source}")
    if analysis.recipe and analysis.recipe.temporal_split:
        show_temporal_policy(analysis.recipe.temporal_split)
    role_names = {
        "input": "模型输入",
        "target": "监督答案",
        "group": "分组/关联",
        "metadata": "检查依据",
        "unused": "暂不使用",
        "unknown": "待澄清",
    }
    st.dataframe(
        [
            {
                "字段": r.column,
                "用途": role_names[r.role],
                "依据": r.reason,
                "预测时可获得": "是"
                if r.available_at_prediction is True
                else "否"
                if r.available_at_prediction is False
                else "待确认",
            }
            for r in analysis.task.field_roles
        ],
        hide_index=True,
        width="stretch",
    )
    st.subheader("数据判断与待确认问题")
    kinds = {
        "observed": "已观察",
        "hypothesis": "待验证推断",
        "needs_business_input": "需要业务解释",
        "needs_full_data": "需要全量验证",
    }
    for finding in analysis.findings:
        evidence = (
            f"（证据：{', '.join(finding.evidence_row_ids)}）" if finding.evidence_row_ids else ""
        )
        st.write(f"**{kinds[finding.kind]}：** {finding.message} {evidence}")
    for question in analysis.questions:
        st.warning(question.question)
        st.caption(
            question.why
            + (" 可选解释：" + " / ".join(question.options) if question.options else "")
        )
    for gap in analysis.capability_gaps:
        st.warning(f"当前能力缺口：{gap}")
    st.write(f"**暂定微调思路：** {analysis.training_approach}")
    st.write("**下一步：**")
    for step in analysis.next_steps:
        st.write(f"- {step}")

feedback = st.text_area(
    "回答问题或修正理解",
    placeholder="请说明真实使用时有哪些信息、哪些答案才是正确的，或指出预览里不符合业务的地方。",
    key=f"intake_feedback_{session.session_id}_{session.revision}",
)
if st.button("保存业务补充，稍后分析", disabled=not feedback.strip()):
    try:
        service.answer(session.session_id, feedback)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))
if st.button(
    "联合分析目标与数据" if not session.analysis else "根据补充说明重新分析", type="primary"
):
    try:
        client = CompatibleChatClient(base_url, model, api_key, allow_remote=allow_remote)
        if feedback.strip():
            session = service.answer(session.session_id, feedback)
        with st.spinner("Agent 正在检查数据、核实字段用途并试运行处理方案…"):
            session = service.analyze(session.session_id, client)
        st.rerun()
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(str(exc))

if session.analysis is None or session.agent_model == "baseline-deterministic":
    with st.expander("没有 Agent 服务？用基础分析开始（产品内置判断，无需任何密钥）"):
        st.caption(
            "基础分析只依据你的字段选择和数据事实生成方案：不判断业务含义，"
            "之后仍需在真实预览里逐行核对；需要多源组合、时间分区或业务问答时再配置 Agent。"
        )
        if session.analysis is not None:
            st.caption(
                "当前已有一份基础分析：调整字段重新生成会替换它，此前的预览确认与对比核验随之失效。"
            )
        baseline_target = st.selectbox(
            "答案列（模型要预测的字段）",
            list(session.source.columns),
            key=f"baseline_target_{session.session_id}",
        )
        baseline_groups = st.multiselect(
            "业务分组字段（同一客户/会话/对象多行时选择，防泄漏；可留空）",
            [column for column in session.source.columns if column != baseline_target],
            key=f"baseline_groups_{session.session_id}",
        )
        baseline_exclude = st.multiselect(
            "排除字段（不作为模型输入；可留空）",
            [
                column
                for column in session.source.columns
                if column != baseline_target and column not in baseline_groups
            ],
            key=f"baseline_exclude_{session.session_id}",
        )
        baseline_temporal = st.checkbox(
            "按时间分区切分（预测未来结果的任务需要，防止把未来信息泄漏进训练）",
            key=f"baseline_temporal_{session.session_id}",
        )
        temporal_policy = None
        if baseline_temporal:
            st.caption(
                "三个时间字段必须互不相同，时间须为带时区的 ISO 格式（如 2026-02-01T00:00:00Z）。"
                "训练/验证的标签必须在下一分区起点前成熟，未成熟的行会明确保留排除，不随机补数。"
            )
            available_options = [
                column for column in session.source.columns if column != baseline_target
            ]
            if len(available_options) < 3:
                st.error(
                    f"当前数据只有 {len(session.source.columns)} 个字段，"
                    "时间分区需要答案列之外还有 3 个互不相同的时间字段；请补充时间列或关闭时间分区。"
                )
            else:
                available_column = st.selectbox(
                    "信息实际可获得时间字段（每行全部输入最晚可得知的时间）",
                    available_options,
                    key=f"baseline_available_at_{session.session_id}",
                )
                prediction_column = st.selectbox(
                    "作出预测时间字段（业务实际下单/决策的时间）",
                    [column for column in available_options if column != available_column],
                    key=f"baseline_prediction_at_{session.session_id}",
                )
                label_end_column = st.selectbox(
                    "标签窗口结束时间字段（仅用于分区，不会作为模型输入）",
                    [
                        column
                        for column in available_options
                        if column not in {available_column, prediction_column}
                    ],
                    key=f"baseline_label_end_at_{session.session_id}",
                )
                validation_start = st.text_input(
                    "验证起点（含时区 ISO 时间）",
                    placeholder="2026-02-01T00:00:00Z",
                    key=f"baseline_validation_start_{session.session_id}",
                )
                test_start = st.text_input(
                    "测试起点（含时区 ISO 时间）",
                    placeholder="2026-03-01T00:00:00Z",
                    key=f"baseline_test_start_{session.session_id}",
                )
                observation_end = st.text_input(
                    "观察截止（含时区 ISO 时间）",
                    placeholder="2026-04-01T00:00:00Z",
                    key=f"baseline_observation_end_{session.session_id}",
                )
                # 边界就地校验：与服务/物化同一套 parse_timestamp 规则，
                # 输入期即可看到哪个边界错，而不是点击后才收到原始报错。
                from src.workbench.temporal_split import parse_timestamp

                boundary_errors = []
                parsed_boundaries = {}
                for name, raw in (
                    ("验证起点", validation_start),
                    ("测试起点", test_start),
                    ("观察截止", observation_end),
                ):
                    value = raw.strip()
                    if not value:
                        continue
                    try:
                        parsed_boundaries[name] = parse_timestamp(value, label=name)
                    except ValueError as exc:
                        boundary_errors.append(str(exc))
                if len(parsed_boundaries) == 3 and not (
                    parsed_boundaries["验证起点"]
                    < parsed_boundaries["测试起点"]
                    < parsed_boundaries["观察截止"]
                ):
                    boundary_errors.append("时间边界需满足：验证起点 < 测试起点 < 观察截止。")
                for message in boundary_errors:
                    st.error(message)
                if label_end_column and not boundary_errors and len(parsed_boundaries) == 3:
                    temporal_policy = {
                        "available_at_column": available_column,
                        "prediction_at_column": prediction_column,
                        "label_end_at_column": label_end_column,
                        "validation_start": validation_start.strip(),
                        "test_start": test_start.strip(),
                        "observation_end": observation_end.strip(),
                    }
        if st.button(
            "生成基础分析并预览",
            key=f"baseline_apply_{session.session_id}",
            disabled=not baseline_target,
        ):
            if baseline_temporal and temporal_policy is None:
                st.error(
                    "已选择按时间分区切分：请选择三个互不相同的时间字段，并填写三个带时区的时间边界。"
                )
            else:
                try:
                    from src.workbench.baseline_analysis import propose_baseline_analysis

                    session = service.apply_analysis(
                        session,
                        propose_baseline_analysis(
                            session,
                            target_column=baseline_target,
                            group_columns=baseline_groups,
                            excluded_columns=baseline_exclude,
                            temporal_policy=temporal_policy,
                        ),
                        model="baseline-deterministic",
                    )
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))

if session.preview:
    st.subheader("真实转换预览")
    st.caption(session.preview.scope_note)

    if session.analysis and session.analysis.recipe and session.analysis.recipe.temporal_split:
        from src.workbench.temporal_split import is_pending_label

        pending_labels = [
            row.row_id
            for row in session.preview.rows
            if is_pending_label(row, session.analysis.recipe.temporal_split)
        ]
        if pending_labels:
            st.info(
                f"其中 {len(pending_labels)} 条记录截至观察截止时标签尚未成熟，将明确保留排除。仍需至少一条已成熟、真实有标签的样例才能确认。"
            )
    st.write(f"**任务指令：** {session.preview.instruction}")
    status_names = {
        "ready": "已生成预览",
        "needs_label": "缺少答案",
        "invalid": "需修正处理",
        "conflict": "答案有冲突",
    }
    summary = st.columns(4)
    for column, (state, label) in zip(summary, status_names.items()):
        column.metric(label, session.preview.counts[state])
    for entry in session.tool_trace:
        if entry.get("tool") == "validate_business_examples" and not entry.get("ok"):
            st.warning(
                "新方案改变了此前认可的样例，请重新核对业务含义：" + "；".join(entry["errors"])
            )
    sample_rows_to_show = select_preview_rows(
        session.preview.rows,
        key=f"sample_preview_{session.session_id}_{session.revision}",
        label="样例",
    )
    show_missing_label_next_step(session.preview.rows)
    for row in sample_rows_to_show:
        with st.expander(
            f"{row.row_id} · {status_names[row.status]}", expanded=len(session.preview.rows) <= 3
        ):
            before, after = st.columns(2)
            with before:
                st.write("原始记录")
                st.json(row.original)
            with after:
                st.write("模型实际输入")
                st.code(row.input, language=None)
                st.write("监督答案")
                st.code(
                    preview_answer(row),
                    language=None,
                )
            for issue in row.issues:
                st.warning(issue)
    status = next_action(session)
    if status == "review_preview":
        st.subheader("对比核验（确认前先配对一次，防止盲点头）")
        contrast = service.contrast_check_status(session.session_id)

        def render_contrast_round() -> None:
            with st.form("contrast_check_form"):
                st.write("开始配对对比：抽取两条答案不同的输入，把答案配到正确的输入上。")
                start_contrast = st.form_submit_button("开始配对对比", type="primary")
            if start_contrast:
                try:
                    pending = service.start_contrast_check(session.session_id, session.revision)
                    st.session_state["pending_contrast_check"] = pending
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))
            pending_contrast = st.session_state.get("pending_contrast_check")
            if pending_contrast:
                st.write(f"**配对题（verification {pending_contrast['check_id'][:8]}）**")
                with st.form(f"contrast_answer_{pending_contrast['check_id']}"):
                    mapping = {}
                    for item in pending_contrast["items"]:
                        st.code(item["input"], language=None)
                        # 默认不预选任何答案:漏选提交不记录核验轮次,瞎点不算配对证据。
                        # 答案加引号展示:仅差空白的两个答案肉眼可辨。
                        mapping[item["row_id"]] = st.selectbox(
                            f"这条输入的正确答案（{item['row_id']}）",
                            ["", *pending_contrast["options"]],
                            format_func=lambda value: "请选择" if value == "" else f"「{value}」",
                            key=f"cc_{pending_contrast['check_id']}_{item['row_id']}",
                        )
                    submitted_contrast = st.form_submit_button("提交配对", type="primary")
                if submitted_contrast:
                    if any(value == "" for value in mapping.values()):
                        st.error("请为每条输入选择答案；未选择不算作答，也不会记录核验轮次。")
                    else:
                        try:
                            service.submit_contrast_check(
                                session.session_id, pending_contrast["check_id"], mapping
                            )
                            st.session_state["pending_contrast_check"] = None
                            st.rerun()
                        except (ValueError, RuntimeError, OSError) as exc:
                            st.error(str(exc))

        def render_contrast_history() -> None:
            """轮次历史逐轮可查:第几轮、题目、你的选择、正确答案、对错。"""
            history = (contrast or {}).get("history") or []
            if not history:
                return
            st.caption("核验轮次历史：每一轮的题目、你的选择与对错都可回查，二连对有据可依。")
            st.dataframe(
                [
                    {
                        "轮次": f"第 {entry['round']} 轮",
                        "题目": item["input"],
                        "你的选择": item["chosen"],
                        "正确答案": item["correct_answer"],
                        "对错": "对" if item["match"] else "错",
                    }
                    for entry in history
                    for item in entry["items"]
                ],
                hide_index=True,
                width="stretch",
            )

        verified_streak = (
            contrast.get("streak", 0) if contrast and contrast.get("verdict") == "verified" else 0
        )
        # 连胜三档词汇与 CLI 同源(contrast_streak_banner):还差一轮/二连对/N 轮连胜,
        # 页面横幅与 CLI stderr 不各说各话。
        from src.workbench.intake_service import contrast_streak_banner

        banner = contrast_streak_banner(contrast)
        if banner:
            if verified_streak >= 2:
                st.success(banner)
            elif contrast and contrast.get("verdict") == "mismatch":
                st.error(banner)
            else:
                st.info(banner)
        render_contrast_history()
        if verified_streak < 2:
            render_contrast_round()
        if verified_streak >= 2:
            # 二连对后核验已达标;第三轮起只是自愿加练,不强制——文案必须与行为一致。
            with st.expander(f"可选：继续第 {verified_streak + 1} 轮对比核验提高置信度（不强制）"):
                st.caption(
                    "核验已达标，不需要再核验；多配一轮只是自愿加练——碰巧连续蒙对的概率会越来越低。"
                )
                render_contrast_round()
        accepted = st.checkbox(
            "已核对预览：输入是模型实际可获得的信息，答案与我希望模型学会的目标一致。",
            key=f"sample_review_{session.session_id}_{session.revision}_{','.join(row.row_id for row in sample_rows_to_show)}",
        )
        if st.button("确认当前转换含义", disabled=not accepted or not sample_rows_to_show):
            try:
                service.confirm(
                    session.session_id,
                    session.revision,
                    [r.row_id for r in sample_rows_to_show],
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    elif status == "awaiting_full_data":
        st.success(
            "样例转换含义已确认，提供全量文件并运行 full-validate 验证覆盖、冲突与独立分组"
            "（多资料任务用 full-sources）；尚未认定可以正式训练。"
        )
    elif status == "awaiting_full_validation":
        st.info(
            "转换含义已确认，运行 full-validate 完成全量业务质量、分区与训练消费检查"
            "（已声明全量可省略 --input；多资料任务用 full-sources）。"
        )
    elif status == "needs_data_revision":
        st.warning(
            "转换存在异常或同输入答案冲突：查看问题行后重新分析——配置了 Agent 运行 analyze；"
            "此前的基础分析可调整字段重跑 baseline-analyze（零密钥）。"
        )
    with st.expander("处理规则与工具记录"):
        st.json(
            session.analysis.recipe.model_dump()
            if session.analysis and session.analysis.recipe
            else {}
        )
        from src.workbench.report_summary import summarize_tool_trace

        for line in summarize_tool_trace(session.tool_trace, "分析"):
            st.write(line)
        st.json(session.tool_trace)

if session.confirmed_revision is not None or session.full_data is not None:
    st.subheader("全量数据验证")
    st.caption("沿已确认方案处理真实全量数据。原样例、业务证据与确认记录单独保留。")
    if session.confirmed_revision is not None:
        if session.analysis and session.analysis.composition:
            composition = session.analysis.composition
            from src.workbench.composition import required_sources as composition_sources

            required_sources = sorted(composition_sources(composition))
            st.write("需要这些原始资料的全量文件：" + "、".join(required_sources))
            reusable = (
                session.full_data.sources
                if session.full_data and session.full_data.sources
                else original_sources
            )
            can_reuse = all(
                alias in reusable and reusable[alias].scope == "full" for alias in required_sources
            )
            if can_reuse and st.button("验证已提供的全部全量资料"):
                try:
                    service.validate_full_sources(session.session_id, session.revision)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
            # 上传控件放在表单外：表单内部件要到提交才提交值，放里面就无法在提交前
            # 按上传的文件类型显示 sheet 选择。
            full_files = {
                alias: st.file_uploader(
                    f"全量原始资料：{alias}",
                    type=["csv", "xlsx", "xls", "jsonl"],
                    key=f"full_source_{session.session_id}_{alias}",
                )
                for alias in required_sources
            }
            with st.form(f"full_sources_{session.session_id}"):
                full_sheets = {
                    alias: excel_sheet_input(upload, key=f"full_sheet_{session.session_id}_{alias}")
                    for alias, upload in full_files.items()
                }
                validate_sources = st.form_submit_button("按组合方案验证全部全量资料")
            if validate_sources:
                missing_sources = [alias for alias, upload in full_files.items() if upload is None]
                if missing_sources:
                    st.error(
                        "请提供以下全量资料："
                        + "、".join(missing_sources)
                        + "。不会将样例自动升级为全量。"
                    )
                else:
                    try:
                        service.validate_full_sources(
                            session.session_id,
                            session.revision,
                            {
                                alias: (upload.name, upload.getvalue())
                                for alias, upload in full_files.items()
                            },
                            # 只有用户填写了 sheet 的资料才进入指定;留空仍读第一个 sheet。
                            sheets={
                                alias: (value or "").strip() or None
                                for alias, value in full_sheets.items()
                                if value
                            }
                            or None,
                        )
                        st.rerun()
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
        else:
            if session.source.scope == "full" and st.button("验证首次上传的全量文件"):
                try:
                    service.validate_full_data(session.session_id, session.revision)
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
            # 配套演示全量:只对「来源就是演示样例」的任务开放(内容摘要比对,
            # 同名不同内容不算)——演示数据不会混进任何真实任务。
            demo_full_pair = (
                demo_full(PROJECT_ROOT)
                if is_demo_session(session.source.digest, PROJECT_ROOT)
                else None
            )
            if demo_full_pair is not None and st.button(DEMO_FULL_BUTTON):
                try:
                    service.validate_full_data(
                        session.session_id,
                        session.revision,
                        demo_full_pair[0],
                        demo_full_pair[1],
                    )
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
            # 上传控件放在表单外：表单内部件要到提交才提交值，放里面就无法在提交前
            # 按上传的文件类型显示 sheet 选择。
            full_upload = st.file_uploader(
                "提供本次任务的全量文件", type=["csv", "xlsx", "xls", "jsonl"]
            )
            with st.form(f"full_upload_{session.session_id}"):
                with st.expander("全量文件读取设置"):
                    full_encoding = st.text_input("全量文件编码（留空自动识别）")
                    full_delimiter = st.selectbox(
                        "全量 CSV 分隔符", ["自动", "逗号", "分号", "Tab", "竖线"]
                    )
                    full_sheet = excel_sheet_input(
                        full_upload, key=f"full_sheet_{session.session_id}"
                    )
                validate_full = st.form_submit_button("按已确认方案验证全量数据")
            if validate_full:
                if full_upload is None:
                    st.error("请提供全量文件；不会自动把样例当作全量。")
                else:
                    try:
                        service.validate_full_data(
                            session.session_id,
                            session.revision,
                            full_upload.name,
                            full_upload.getvalue(),
                            encoding=full_encoding.strip() or None,
                            delimiter={"逗号": ",", "分号": ";", "Tab": "\t", "竖线": "|"}.get(
                                full_delimiter
                            ),
                            sheet=(full_sheet or "").strip() or None,
                        )
                        st.rerun()
                    except (ValueError, OSError) as exc:
                        st.error(str(exc))
    report = session.full_data
    if report is not None:
        st.caption(
            f"全量来源：{report.source.name} · {len(report.source.rows)} 条 · {report.source.digest[:12]}。以下行 ID 仅属于这份全量文件。"
        )
        show_fact_notes(report.profile)
        if report.status == "stale":
            st.warning(
                "业务理解或方案已变化，以下全量报告已失效。请重新分析、确认样例方案，再验证全量数据。"
            )
        if (
            report.status != "stale"
            and report.preview
            and session.analysis
            and session.analysis.recipe
            and session.analysis.recipe.temporal_split
        ):
            from src.workbench.temporal_split import temporal_assignment

            try:
                temporal_preview = temporal_assignment(
                    report.preview.rows,
                    session.analysis.recipe.temporal_split,
                    session.analysis.recipe.group_columns,
                )
                time_counts = {
                    name: len(indices) for name, indices in temporal_preview["assignments"].items()
                }
                st.info(
                    f"按当前时间方案预览：训练 {time_counts['train']} 条、验证 {time_counts['validation']} 条、测试 {time_counts['test']} 条；排除并保留 {len(temporal_preview['excluded_rows'])} 条。请在确认前核对窗口与原行。"
                )
                show_temporal_exclusions(
                    temporal_preview["excluded_rows"], title="全量时间方案的排除预览"
                )
            except ValueError as exc:
                st.error(f"时间分区仍需修正：{exc}")
        if report.composition_report:
            with st.expander("全量资料组合过程与原始来源"):
                st.dataframe(report.composition_report["steps"], hide_index=True, width="stretch")
                st.json(report.composition_report["origins"])
        if report.adapter_report:
            show_adapter_report(
                report.adapter_report,
                session.analysis.adapter if session.analysis else None,
                title="全量受限适配验证结果",
            )
        for issue in report.issues:
            show_issue = {"blocking": st.error, "review": st.warning, "info": st.info}[
                issue.severity
            ]
            show_issue(issue.message)
            if issue.row_ids:
                st.caption(
                    "全量证据行："
                    + "、".join(issue.row_ids[:20])
                    + (f"（共 {len(issue.row_ids)} 条）" if len(issue.row_ids) > 20 else "")
                )
        if report.new_target_values:
            st.write("**样例未覆盖的类别答案**")
            st.json(report.new_target_values)
        with st.expander("全量原始记录与结构变化"):
            st.json(report.schema_drift)
            st.json(report.profile)
            for source_row in report.source.rows[:20]:
                st.write(f"全量 {source_row.row_id} · 文件行 {source_row.line}")
                st.json(source_row.values)
        if report.preview:
            st.write("**全量真实转换预览**")
            full_status_names = {
                "ready": "已生成预览",
                "needs_label": "缺少答案",
                "invalid": "需修正处理",
                "conflict": "答案有冲突",
            }
            st.caption(
                " · ".join(
                    f"{full_status_names[state]} {count} 条"
                    for state, count in report.preview.counts.items()
                )
            )
            issue_ids = {
                row_id
                for issue in report.issues
                if issue.severity != "info"
                for row_id in issue.row_ids
            }
            rows_to_show = select_preview_rows(
                report.preview.rows,
                key=f"full_preview_{session.session_id}_{session.revision}",
                label="全量",
                issue_ids=issue_ids,
            )
            show_missing_label_next_step(report.preview.rows, full=True)
            for row in rows_to_show:
                with st.expander(f"全量 {row.row_id} · {full_status_names[row.status]}"):
                    st.json(row.original)
                    st.write("模型实际输入")
                    st.code(row.input, language=None)
                    st.write("监督答案")
                    st.code(preview_answer(row), language=None)
                    for issue in row.issues:
                        st.warning(issue)
            if next_action(session) == "review_full_data":
                full_accepted = st.checkbox(
                    "已核对全量报告和展示的记录，新增类别与字段符合业务理解。",
                    key=f"full_review_{session.session_id}_{session.revision}_{','.join(row.row_id for row in rows_to_show)}",
                )
                if st.button("确认全量数据含义", disabled=not full_accepted or not rows_to_show):
                    try:
                        service.confirm_full_data(
                            session.session_id, session.revision, [r.row_id for r in rows_to_show]
                        )
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        if next_action(session) in {"awaiting_dataset_split", "ready_for_training_preflight"}:
            st.subheader("盲标核验（训练前的语义安全关卡）")
            st.caption(
                "系统随机抽取几条已标注行并隐藏答案，请仅根据输入给出你的答案；"
                "与数据标签全部一致才允许准备训练。这验证的是监督信号的业务含义，"
                "数据或方案修订后需重新核验。重新核验会换一组题——"
                "未通过时公布的正确答案照抄进下一轮是无效的，防止背题。"
            )
            verification = session.label_verification
            if verification and verification.get("stale"):
                st.warning(
                    "数据或处理方案修订后，此前完成的盲标核验已失效——监督信号的业务含义可能已经改变，"
                    "请重新完成一轮核验后再准备训练。"
                )

            def result_evidence_note(record: dict) -> str:
                """核验结论附统计说明:旧记录没有存 evidence_note 时现算,口径与服务层一致。"""
                note = record.get("evidence_note")
                if note:
                    return note
                matched_count, size = record.get("matched"), record.get("sample_size")
                if isinstance(matched_count, int) and isinstance(size, int) and size > 0:
                    return agreement_evidence_note(matched_count, size)
                return ""

            if verification and verification.get("verdict") == "verified":
                st.success(
                    f"盲标核验已通过（{verification['matched']}/{verification['sample_size']} 一致）。"
                    + result_evidence_note(verification)
                )
            else:
                if verification and verification.get("verdict") == "insufficient_agreement":
                    st.error(
                        f"盲标核验未通过（{verification['matched']}/{verification['sample_size']} 一致）；"
                        "训练不会开始。请逐条核对不一致原因："
                    )
                    for item in verification["items"]:
                        if not item["match"]:
                            with st.expander(
                                f"{item['row_id']}：你的答案「{item['submitted_answer']}」 vs 数据标签「{item['data_label']}」"
                            ):
                                st.code(item["input"], language=None)
                    # 三因分辨行与服务层同源(记录的 mismatch_triage 键):页面渲染记录内行,
                    # 不另起一份词汇;早期存档没有该键时如实没有这些行。
                    for triage_line in verification.get("mismatch_triage", []):
                        st.caption(triage_line)
                    st.caption("标签错误、业务歧义或任务定义不清都会造成不一致；修正后重新核验。")
                    note = result_evidence_note(verification)
                    if note:
                        st.caption(note)
                with st.form("label_verification_form"):
                    st.write("开始一轮新的盲标核验：题目在提交表单后展示（答案不随题目显示）。")
                    sample_size = st.number_input(
                        "核验样本量（抽取多少条已标注行）",
                        min_value=1,
                        max_value=50,
                        value=5,
                        step=1,
                        key=f"lv_sample_size_{session.session_id}",
                    )
                    st.caption(
                        "默认 5 条：快速关卡，适合首轮快速发现问题或低风险业务。"
                        "高风险业务需要更强证据，建议 20 条以上——"
                        f"5 条全部一致的 95% 置信下界约 {wilson_lower_bound(5, 5):.0%}，"
                        f"20 条约 {wilson_lower_bound(20, 20):.0%}，"
                        f"30 条约 {wilson_lower_bound(30, 30):.0%}。"
                        "核验强度由你按业务风险决定，系统不替你设定。"
                    )
                    started = st.form_submit_button("抽取盲标核验题目", type="primary")
                if started:
                    try:
                        pending = service.start_label_verification(
                            session.session_id, session.revision, sample_size=int(sample_size)
                        )
                        st.session_state["pending_label_verification"] = pending
                        st.rerun()
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.error(str(exc))
            pending_items = st.session_state.get("pending_label_verification")
            concluded = bool(
                verification
                and verification.get("verdict") in {"verified", "insufficient_agreement"}
                and verification.get("verification_id")
                == (pending_items or {}).get("verification_id")
            )
            if pending_items and not concluded:
                st.write(
                    f"**本轮核验（{pending_items['sample_size']} 条，verification {pending_items['verification_id'][:8]}）**"
                )
                start_note = pending_items.get("evidence_note")
                if start_note:
                    st.caption(start_note)
                shortfall = pending_items.get("shortfall_note")
                if shortfall:
                    st.caption(shortfall)
                with st.form(f"label_verification_answer_{pending_items['verification_id']}"):
                    answers = {}
                    for item in pending_items["items"]:
                        st.code(item["input"], language=None)
                        answers[item["row_id"]] = st.text_input(
                            f"你的答案（{item['row_id']}）",
                            key=f"lv_{pending_items['verification_id']}_{item['row_id']}",
                        )
                    submitted = st.form_submit_button("提交盲标核验答案", type="primary")
                if submitted:
                    try:
                        service.submit_label_verification(
                            session.session_id,
                            pending_items["verification_id"],
                            answers,
                        )
                        st.session_state["pending_label_verification"] = None
                        st.rerun()
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.error(str(exc))
        if next_action(session) == "awaiting_dataset_split":
            st.success("全量报告已确认。下一步准备独立训练与评测分区；尚未认定可以正式训练。")
            st.subheader("生成独立训练与评测分区")
            groups = session.analysis.recipe.group_columns
            temporal_policy = session.analysis.recipe.temporal_split
            independent_rows = False
            if groups:
                st.write(
                    "按已确认的业务对象分组：" + "、".join(groups) + "。同组记录保持在同一分区。"
                )
            else:
                independent_rows = st.checkbox(
                    "已确认每行是独立业务对象，不存在需要保持同组的客户、会话或文档。"
                )
            with st.expander("分区设置"):
                selected_suite = st.selectbox(
                    "后续轮次的固定开发/测试题集",
                    [active_iteration["evaluation_suite"]["suite_id"]]
                    if active_iteration
                    else ["", *available_suites],
                    disabled=active_iteration is not None,
                    format_func=lambda identity: (
                        "首次建立独立分区"
                        if not identity
                        else f"固定题集 {identity[:12]} · 开发 {available_suites[identity]['case_counts']['validation']} / 测试 {available_suites[identity]['case_counts']['test']}"
                    ),
                )
                if selected_suite:
                    st.info(
                        "保持原开发和测试题目不变；新增资料按已确认时间归属，新增开发/测试记录不扩充原评分题。"
                        if temporal_policy
                        else "保持原开发和测试题目不变；新增独立资料进入训练集。同一业务对象不能跨入训练集。"
                    )
                    from src.workbench.report_summary import summarize_suite

                    for line in summarize_suite(available_suites[selected_suite]):
                        st.write(line)
                dataset_name = st.text_input("数据集名称（留空自动生成）")
                validation_fraction, test_fraction, split_seed = 0.1, 0.1, 42
                if temporal_policy:
                    st.info("本方案按已确认时间边界划分训练、验证与测试；随机比例与种子不生效。")
                elif not selected_suite:
                    from src.workbench.training_guidance import split_settings_guidance_lines

                    for line in split_settings_guidance_lines():
                        st.caption(line)
                    validation_fraction = st.number_input(
                        "验证集比例", min_value=0.01, max_value=0.49, value=0.1, step=0.01
                    )
                    test_fraction = st.number_input(
                        "独立测试集比例", min_value=0.01, max_value=0.49, value=0.1, step=0.01
                    )
                    split_seed = st.number_input("可复现分区种子", min_value=0, value=42, step=1)
            if st.button("生成数据集版本", disabled=not groups and not independent_rows):
                try:
                    service.materialize_dataset(
                        session.session_id,
                        session.revision,
                        name=dataset_name.strip() or None,
                        validation_fraction=validation_fraction,
                        test_fraction=test_fraction,
                        seed=int(split_seed),
                        independent_rows_confirmed=independent_rows,
                        **(
                            {"evaluation_suite": available_suites[selected_suite]}
                            if selected_suite
                            else {}
                        ),
                    )
                    st.rerun()
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(str(exc))

dataset = getattr(session, "dataset", None)
if dataset is not None:
    st.subheader("数据集版本与分区产物")
    st.write(f"**{dataset.name}** · 版本 `{dataset.version}`")
    counts = dataset.statistics["row_counts"]
    st.write(
        f"训练 {counts['train']} 条 · 验证 {counts['validation']} 条 · 独立测试 {counts['test']} 条"
    )
    # 分区统计大白话摘要：非专家读句子核对「怎么分的、排除了什么」，不读 JSON。
    # 独立测试集过小的提醒由 summarize_dataset 内部统一带出——页面与 CLI
    # 同走这一条渲染路径，不另设直渲染位（避免同一行出现两次）。
    from src.workbench.report_summary import summarize_dataset

    for line in summarize_dataset(dataset.statistics):
        st.write(line)
    if dataset.statistics.get("split_method", "").startswith("temporal"):
        st.info(
            f"按时间规则纳入 {dataset.statistics.get('included_rows', sum(counts.values()))} 条，排除并保留 {dataset.statistics.get('excluded_rows', 0)} 条；没有随机回退或补标签。"
        )
        for reason, count in dataset.statistics.get("exclusion_counts", {}).items():
            st.write(f"{TEMPORAL_EXCLUSION_NAMES.get(reason, reason)}：{count} 条")
        try:
            manifest = json.loads(Path(dataset.paths["manifest"]).read_text(encoding="utf-8"))
            metadata = manifest.get("metadata", {})
            policy_payload = metadata.get("temporal_policy")
            if policy_payload:
                # 版本卡回显本版本实际使用的时间字段与边界：数据集是产物，
                # 核对「当时是按什么分界线切的」不应要求用户去解析 manifest JSON。
                from src.workbench.intake_models import TemporalSplitPolicy

                show_temporal_policy(
                    TemporalSplitPolicy.model_validate(policy_payload),
                    title="**本数据集版本实际使用的时间分区方案（核对边界与字段）**",
                )
            excluded = metadata.get("excluded_rows", [])
            show_temporal_exclusions(excluded, title="本版本的时间排除明细")
            st.download_button(
                "下载时间排除记录",
                json.dumps(excluded, ensure_ascii=False, indent=2),
                file_name=f"temporal-exclusions-{dataset.version}.json",
                mime="application/json",
                key=f"download_temporal_{dataset.version}",
            )
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"无法读取本版本排除明细：{exc}")
    with st.expander("分区统计与来源"):
        st.json(dataset.statistics)
        st.json(dataset.paths)
    fixed_suite = getattr(dataset, "evaluation_suite", None)
    if fixed_suite:
        st.info(
            f"本轮沿用固定题集 {fixed_suite['suite_id'][:12]}：开发 {fixed_suite['case_counts']['validation']} 题，独立测试 {fixed_suite['case_counts']['test']} 题。"
        )
    elif next_action(session) == "ready_for_training_preflight":
        st.caption("需要进行多轮改进时，先固定当前开发与测试题目，使后续提升可在同一题集上验证。")
        if st.button("固定当前开发与测试题集"):
            try:
                reference = suite_service.freeze(session)
                st.session_state[f"frozen_suite_{session.session_id}"] = reference["suite_id"]
                st.rerun()
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(str(exc))
        frozen_id = st.session_state.get(f"frozen_suite_{session.session_id}")
        if frozen_id:
            st.success(
                f"已固定题集 {frozen_id[:12]}，后续轮次使用这组题目；当前训练版本保持原记录。"
            )
    if next_action(session) == "ready_for_training_preflight":
        st.success("独立数据分区已生成，可继续训练前检查；分区就绪不代表模型效果已验收。")
        with st.expander("📋 任务规约投影（训练启动前对齐「我们在教模型什么」）"):
            # ADR-1 只读投影:由既有确认记录汇编,不新增状态;与 CLI task-spec-show 同源同词汇。
            from src.workbench.task_spec_projection import summarize_task_spec

            spec = collect_current_task_spec(session.session_id)
            for line in summarize_task_spec(spec):
                st.write(line)
            with st.expander("查看规约原始 JSON"):
                st.json(spec)
        st.subheader("训练前检查")
        st.caption(
            "只读取本地目录或已有缓存中的 tokenizer，不自动下载、不加载模型权重、不启动训练。"
        )
        with st.expander("可学性探针（可选）：这份数据学得出这个任务吗？"):
            st.caption(
                "用基座模型对开发集抽样做零样本探测，与「瞎猜多数类」基线对比。"
                "这是最便宜的证据，不是判决：样本量小、零样本差异不能预测微调效果；"
                "显著低于基线通常意味着提示格式或任务定义需要先核查。会实际加载本地模型。"
            )
            probe_model = st.text_input(
                "本地基础模型目录（已准备好权重）",
                key=f"probe_model_{session.session_id}",
            )
            if st.button(
                "运行可学性探针",
                key=f"probe_run_{session.session_id}",
                disabled=not probe_model.strip(),
            ):
                try:
                    from src.workbench.learnability_probe import probe_learnability, save_probe

                    with st.spinner("基座零样本探测开发集抽样…"):
                        result = probe_learnability(
                            service.load(session.session_id), probe_model.strip()
                        )
                        save_probe(PROJECT_ROOT / "outputs" / "workbench" / "probes", result)
                    render_probe_result(
                        result, source_hints=probe_source_hints(service.load(session.session_id))
                    )
                except (ValueError, RuntimeError, OSError, ImportError) as exc:
                    st.error(str(exc))
            else:
                # 探针要真实加载本地模型,重跑成本高;结果已存盘,任何交互后回读最近一次,
                # 不让用户为了再看一眼而重新加载模型。按数据版本过滤,旧版本不冒充新证据。
                from src.workbench.learnability_probe import load_latest_probe

                saved = load_latest_probe(
                    PROJECT_ROOT / "outputs" / "workbench" / "probes", dataset.version
                )
                if saved:
                    st.caption(
                        f"显示最近一次已保存的探针结果（基座 `{saved.get('model_path', '未知')}`，"
                        "抽样见原始记录）；需要重新探测请再运行一次。"
                    )
                    render_probe_result(saved, source_hints=probe_source_hints(session))
        with st.form(f"training_preflight_{session.session_id}"):
            tokenizer_path = st.text_input("本地 tokenizer 目录或已缓存标识")
            max_length = st.number_input("训练最大 token 长度", min_value=1, value=2048, step=1)
            run_preflight = st.form_submit_button("检查实际截断与答案保留")
        if run_preflight:
            if not tokenizer_path.strip():
                st.error("请填写已准备好的本地 tokenizer 目录或缓存标识。")
            else:
                try:
                    from src.workbench.training_preflight import load_local_tokenizer

                    with st.spinner("正在按训练模板检查真实 token 与答案保留情况…"):
                        tokenizer = load_local_tokenizer(
                            tokenizer_path.strip(), local_files_only=True
                        )
                        service.preflight_training(
                            session.session_id, session.revision, tokenizer, int(max_length)
                        )
                    st.rerun()
                except (ValueError, RuntimeError, OSError, ImportError) as exc:
                    st.error(str(exc))
    else:
        st.warning("以下为先前生成的数据版本；业务理解或资料更新后，需要重新完成确认与分区。")
    with st.expander("训练数据配置"):
        st.json(dataset.data_config)
    st.download_button(
        "下载训练数据配置",
        json.dumps(dataset.data_config, ensure_ascii=False, indent=2),
        file_name=f"{dataset.name}-data-config.json",
        mime="application/json",
    )
    for split, label in (("train", "训练集"), ("validation", "验证集"), ("test", "独立测试集")):
        artifact_path = Path(dataset.paths[split])
        if artifact_path.is_file():
            st.download_button(
                f"下载{label}",
                artifact_path.read_bytes(),
                file_name=f"{dataset.name}-{split}.jsonl",
                mime="application/x-ndjson",
            )
        else:
            st.warning(f"{label}文件当前不可读取：{artifact_path}")
    st.caption(
        "训练时同时使用配置中的训练与验证文件，独立测试集留待最终评测。数据配置应放入完整训练配置的 data 部分。"
    )

if session.training_preflight is not None:
    preflight = session.training_preflight
    st.subheader("训练前检查报告")
    {"blocked": st.error, "warnings": st.warning, "passed": st.success}[preflight["status"]](
        {
            "blocked": "存在阻断问题，请先修正数据或训练长度。",
            "warnings": "检查完成，有需要核对的风险。",
            "passed": "当前数据与 token 消费检查通过。",
        }[preflight["status"]]
    )
    st.caption(preflight["scope_note"])
    for issue in preflight["issues"]:
        {"blocking": st.error, "warning": st.warning, "info": st.info}[issue["severity"]](
            issue["message"]
        )
        if issue.get("row_ids"):
            st.caption(f"分区 {issue.get('split') or '全部'} · 行：" + "、".join(issue["row_ids"]))
    with st.expander("各分区的截断与答案丢失统计", expanded=True):
        st.json(preflight["splits"])
    with st.expander("实际 token 消费与问题行"):
        st.dataframe(preflight["rows"], hide_index=True, width="stretch")
        st.json(preflight["tokenizer"])

if next_action(session) == "ready_for_training_preflight":
    show_business_scoring()

training_service = TrainingRunService(
    PROJECT_ROOT / "outputs/workbench/training", project_root=PROJECT_ROOT
)
training_runs = training_service.list_runs(session_id=session.session_id)
if next_action(session) == "ready_for_training_preflight" or training_runs:
    st.subheader("用当前数据微调模型")
    st.caption(
        "使用已确认的数据版本和独立分区。选择本地基础模型后重新检查对应 tokenizer，再启动训练。"
    )
    if next_action(session) == "ready_for_training_preflight":
        show_training_recommendations()
        with st.expander("高级：手工配置训练参数"):
            from src.workbench.training_guidance import manual_training_parameter_lines

            for line in manual_training_parameter_lines():
                st.caption(line)
            row_counts = (
                (session.dataset.statistics or {}).get("row_counts") or {}
                if session.dataset
                else {}
            )
            total_rows = sum(int(count) for count in row_counts.values())
            suggested_lr, lr_reason = learning_rate_suggestion(total_rows)
            lr_key = f"training_lr_{session.session_id}"
            if lr_key not in st.session_state:
                st.session_state[lr_key] = 0.0002  # 表单默认值：与「推荐起步值」2e-4 一致
            # Streamlit 表单内不能放普通按钮：采用按钮放在表单上方，点击后把建议值填入表单内学习率输入。
            if st.button(
                "采用建议学习率",
                key=f"adopt_lr_{session.session_id}",
                help=f"按全量 {total_rows} 条填入建议起步值 {suggested_lr:.7f}；填入后仍可手改。",
            ):
                st.session_state[lr_key] = suggested_lr
            with st.form(f"prepare_training_{session.session_id}"):
                training_model_path = st.text_input(
                    "本地基础模型目录", placeholder="包含基础模型权重、配置和 tokenizer 的目录"
                )
                training_length = st.number_input(
                    "本轮训练最大 token 长度", min_value=1, value=1024, step=1
                )
                training_epochs = st.number_input("训练轮数", min_value=1, value=1, step=1)
                training_batch = st.number_input("每设备 batch size", min_value=1, value=1, step=1)
                with st.expander("基础训练参数"):
                    st.caption(f"学习率分档建议：{lr_reason}{LR_TIER_DISCLAIMER}")
                    training_lr = st.number_input(
                        "学习率", min_value=0.0000001, format="%.7f", key=lr_key
                    )
                    training_accumulation = st.number_input(
                        "梯度累积步数", min_value=1, value=4, step=1
                    )
                    training_rank = st.number_input("LoRA rank", min_value=1, value=8, step=1)
                    training_quantized = st.checkbox(
                        "使用 4-bit 量化（仅适用兼容的 NVIDIA CUDA 环境）"
                    )
                prepare_training = st.form_submit_button("准备本轮训练方案")
            if prepare_training:
                if not training_model_path.strip():
                    st.error("请选择已准备好的本地基础模型目录。")
                else:
                    try:
                        with st.spinner("正在核对本地模型与已确认数据，并检查实际 token 消费…"):
                            training_service.prepare(
                                session,
                                training_model_path.strip(),
                                max_length=int(training_length),
                                training_options={
                                    "num_epochs": int(training_epochs),
                                    "batch_size": int(training_batch),
                                    "learning_rate": float(training_lr),
                                    "gradient_accumulation_steps": int(training_accumulation),
                                },
                                lora_options={
                                    "r": int(training_rank),
                                    "lora_alpha": int(training_rank) * 2,
                                },
                                model_options={
                                    "quantization_bits": 4 if training_quantized else None
                                },
                            )
                        st.rerun()
                    except (ValueError, RuntimeError, OSError, ImportError) as exc:
                        st.error(str(exc))
    if training_runs:
        if st.button("刷新训练状态和日志"):
            st.rerun()
        run_status_names = {
            "prepared": "方案已准备",
            "blocked": "需要修正",
            "running": "正在训练",
            "succeeded": "训练完成",
            "failed": "训练失败",
            "stopped": "已停止",
            "stopping": "正在停止",
            "unknown": "状态待核实",
        }
        for saved_run in reversed(training_runs):
            try:
                run = training_service.get_status(saved_run["run_id"])
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(str(exc))
                continue
            run_id = run["run_id"]
            run_iteration = next(
                (item for item in iterations if item.get("new_run_id") == run_id), None
            )
            if run_iteration is None and run.get("recovery_parent_run_id"):
                candidate_iteration = next(
                    (
                        item
                        for item in iterations
                        if item.get("new_run_id") == run["recovery_parent_run_id"]
                    ),
                    None,
                )
                if candidate_iteration:
                    try:
                        original_run = training_service.get_status(run["recovery_parent_run_id"])
                        if (
                            original_run.get("recover_technical_failures")
                            and (original_run.get("recovery") or {}).get("child_run_id") == run_id
                        ):
                            run_iteration = candidate_iteration
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.warning(f"无法核对本轮技术恢复关联：{exc}")
            run_execution = (
                execution_records.get(run_iteration["iteration_id"]) if run_iteration else None
            )
            run_execution_managed = bool(
                run_execution and run_execution["status"] in execution_managed_states
            )
            with st.expander(
                f"训练 {run_id} · {run_status_names.get(run['status'], run['status'])}",
                expanded=run["status"] in {"prepared", "blocked", "running", "stopping", "failed"},
            ):
                st.write(f"**基础模型：** {run.get('model_path', '')}")
                st.write(f"**数据版本：** `{run.get('dataset_version', '')}`")
                for issue in run.get("issues", []):
                    if isinstance(issue, dict):
                        severity, message = issue.get("severity"), issue.get("message", str(issue))
                    else:
                        severity, message = "warning", str(issue)
                    {"blocking": st.error, "warning": st.warning, "info": st.info}.get(
                        severity, st.warning
                    )(message)
                preflight = run.get("preflight") or {}
                for issue in preflight.get("issues", []):
                    if issue in run.get("issues", []):
                        continue
                    {"blocking": st.error, "warning": st.warning, "info": st.info}.get(
                        issue.get("severity"), st.info
                    )(issue.get("message", ""))
                from src.workbench.report_summary import summarize_preflight, summarize_training_run

                with st.expander("用大白话看这轮训练与检查"):
                    for line in summarize_training_run(run):
                        st.write(line)
                    st.write("")
                    for line in summarize_preflight(preflight):
                        st.write(line)
                with st.expander("本轮配置与预检记录（原始数据）"):
                    st.json(run.get("config", {}))
                    st.json(preflight)
                recovery = run.get("recovery") or {}
                if run.get("recovery_parent_run_id"):
                    st.info(
                        f"本次是训练 {run['recovery_parent_run_id']} 的一次技术恢复；原失败记录继续保留。"
                    )
                if recovery:
                    recovery_states = {
                        "dispatch_waiting": "等待原训练退出并释放资源",
                        "preparing_retry": "正在准备一次技术重试",
                        "retry_started": "已启动技术重试",
                        "retry_blocked": "重试被阻断，需要处理",
                        "dispatch_failed": "自动恢复启动失败",
                        "cancelled": "已取消自动恢复",
                    }
                    st.write(
                        "**技术恢复状态：** "
                        + recovery_states.get(
                            recovery.get("status"), str(recovery.get("status", "未记录"))
                        )
                    )
                    if recovery.get("reason"):
                        st.write(recovery["reason"])
                    if recovery.get("child_run_id"):
                        st.info(
                            f"恢复训练：{recovery['child_run_id']}，请查看对应记录的进度与产物。"
                        )
                    with st.expander(f"训练 {run_id} 的恢复依据与参数变化"):
                        st.json(
                            {
                                "changes": recovery.get("changes", {}),
                                "evidence": recovery.get("evidence", {}),
                            }
                        )
                    child_status = next(
                        (
                            item.get("status")
                            for item in training_runs
                            if item["run_id"] == recovery.get("child_run_id")
                        ),
                        None,
                    )
                    recovery_can_stop = recovery.get("status") in {
                        "dispatch_waiting",
                        "preparing_retry",
                        "retry_started",
                    } and child_status not in {"succeeded", "failed", "stopped"}
                    if (
                        recovery_can_stop
                        and not run_execution_managed
                        and st.button("停止此训练及其自动恢复", key=f"stop_recovery_{run_id}")
                    ):
                        try:
                            training_service.stop(run_id)
                            st.rerun()
                        except (ValueError, RuntimeError, OSError) as exc:
                            st.error(str(exc))
                if run["status"] == "prepared" and run.get("recovery_parent_run_id"):
                    st.info(
                        "此子训练由获准恢复流程管理；请刷新查看进展。恢复被阻断时，先查看原训练的原因与参数核查。"
                    )
                if run_execution_managed:
                    st.info("这轮训练由上方后台执行统一推进；请在那里查看、核对提示或停止。")
                dataset_current = bool(
                    session.dataset and run.get("dataset_version") == session.dataset.version
                )
                if run["status"] == "prepared" and not dataset_current:
                    st.caption(
                        "该方案基于旧数据版本"
                        f"（{(run.get('dataset_version') or '未知')[:8]}，当前 "
                        f"{(session.dataset.version if session.dataset else '无')[:8]}）；"
                        "启动会被拒绝，请用当前已确认数据重新准备训练。"
                    )
                if (
                    run["status"] == "prepared"
                    and not run.get("recovery_parent_run_id")
                    and not run_execution_managed
                    and dataset_current
                ):
                    with st.expander("📋 任务规约（启动本轮训练前的口径）"):
                        # 与训练前检查位的规约卡同源同词汇(ADR-1 只读投影):
                        # 能启动时先对齐「我们在教模型什么」,再决定按当前方案启动。
                        from src.workbench.task_spec_projection import summarize_task_spec

                        spec = collect_current_task_spec(session.session_id)
                        for line in summarize_task_spec(spec):
                            st.write(line)
                    recover_technical = False
                    if not run.get("recovery_parent_run_id"):
                        recover_technical = st.checkbox(
                            "允许显存不足后自动技术重试一次（保持业务目标、数据和有效 batch）。",
                            key=f"recover_{run_id}",
                        )
                        st.caption(
                            "仅尝试缩小微批次并增加梯度累积，或启用梯度检查点。其他失败与重试失败会保留错误，停止操作会取消后续恢复。"
                        )
                    recovery_options = (
                        {"recover_technical_failures": True} if recover_technical else {}
                    )
                    acknowledge = False
                    if preflight.get("status") == "warnings":
                        acknowledge = st.checkbox(
                            "已核对任务规约与预检提示，按当前方案开始训练。",
                            key=f"train_warnings_{run_id}",
                        )
                    if st.button(
                        "启动这轮训练",
                        key=f"start_{run_id}",
                        disabled=preflight.get("status") == "warnings" and not acknowledge,
                    ):
                        try:
                            if run_iteration:
                                iteration_service.start(
                                    run_iteration["iteration_id"],
                                    service.load(session.session_id),
                                    acknowledge_warnings=acknowledge,
                                    **recovery_options,
                                )
                            else:
                                training_service.start(
                                    run_id,
                                    service.load(session.session_id),
                                    acknowledge_warnings=acknowledge,
                                    **recovery_options,
                                )
                            st.rerun()
                        except (ValueError, RuntimeError, OSError, ImportError) as exc:
                            st.error(str(exc))
                if (
                    run["status"] in {"running", "unknown"}
                    and not run_execution_managed
                    and st.button("停止这轮训练", key=f"stop_{run_id}")
                ):
                    try:
                        training_service.stop(run_id)
                        st.rerun()
                    except (ValueError, RuntimeError, OSError) as exc:
                        st.error(str(exc))
                if run["status"] == "stopping":
                    st.info("已提交停止请求，刷新后查看进程最终状态。")
                try:
                    logs = training_service.read_logs(run_id, tail=100)
                except (ValueError, RuntimeError, OSError) as exc:
                    logs = ""
                    st.error(f"无法读取日志：{exc}")
                if logs:
                    st.code(logs, language=None)
                if run.get("error"):
                    st.error(run["error"])
                if run.get("failure"):
                    failure = run["failure"]
                    st.error(f"{failure.get('stage', '训练')}：{failure.get('message', '')}")
                if run.get("output_dir") and run["status"] != "succeeded":
                    # 环节⑤「实时曲线」:训练中/中断态也画曲线(数据来自
                    # LiveLossWriter 的增量写入)。训练中只画曲线不给三态判定
                    # (半程数据不足以支持整场结论);中断态给趋势但注明只代表
                    # 已训练部分。成功态走下方 metrics 块,两块互斥。
                    from src.workbench.training_progress import (
                        load_loss_history,
                        loss_trend_lines,
                    )

                    partial = load_loss_history(run["output_dir"])
                    if len(partial) >= 2:
                        if run["status"] in {"running", "stopping"}:
                            st.write("**训练中的 loss 曲线**")
                            st.caption(
                                "训练进行中，曲线只画到最近一次日志点；刷新页面查看最新进度。"
                            )
                        else:
                            st.write("**训练未完成时已记录的 loss 曲线**")
                            st.caption("训练中断，以下内容只代表已训练的部分。")
                            for line in loss_trend_lines(partial):
                                st.caption(line)
                        from ui.components.charts import make_metric_timeseries

                        st.plotly_chart(
                            make_metric_timeseries(
                                {"loss": [(int(p["step"]), p["loss"]) for p in partial]}
                            ),
                            width="stretch",
                        )
                if run.get("metrics"):
                    st.write("**本轮训练指标**")
                    # 环节⑤可视化：逐条 loss 序列 → 趋势人话 + 曲线（单一来源
                    # loss_trend_lines，与 CLI train-status 同源同词汇）；
                    # 原始 flat metrics 收进折叠区保透明，不再裸倾倒。
                    from src.workbench.training_progress import (
                        load_loss_history,
                        loss_trend_lines,
                    )

                    history = load_loss_history(run.get("output_dir"))
                    for line in loss_trend_lines(history):
                        st.caption(line)
                    if len(history) >= 2:
                        from ui.components.charts import make_metric_timeseries

                        st.plotly_chart(
                            make_metric_timeseries(
                                {"loss": [(int(p["step"]), p["loss"]) for p in history]}
                            ),
                            width="stretch",
                        )
                    with st.expander("查看原始指标 JSON"):
                        st.json(run["metrics"])
                if run.get("output_dir"):
                    st.write(f"**产物目录：** `{run['output_dir']}`")
                if run.get("artifacts"):
                    st.json(run["artifacts"])
                if run["status"] == "succeeded":
                    st.success("本轮训练已完成。先比较开发集表现，独立测试集留待最终业务验收。")
                    try:
                        from src.workbench.registry_link import run_registration_status

                        registration = run_registration_status(
                            run_id, f"sqlite:///{Path(training_service.root) / 'mlflow.db'}"
                        )
                        # 注册状态人话与 CLI train-lineage 同一份摘要
                        # (summarize_registration 单一来源),页面与 CLI 同源同词汇;
                        # 未注册时命令原文收进折叠区,已注册/失败态直接平铺。
                        from src.workbench.report_summary import summarize_registration

                        lines = summarize_registration(registration)
                        if registration["status"] == "not_registered":
                            with st.expander("把这次训练的模型注册进模型库（可选）"):
                                for line in lines:
                                    st.caption(line)
                        else:
                            for line in lines:
                                st.caption(line)
                    except Exception as exc:  # 血缘查询失败不阻塞训练信息展示
                        st.caption(f"模型库查询失败：{exc}")
                    # 环节⑨交接出口:北极星旅程的最后一环是「模型 + 证据」离场。
                    # 页面只做只读盘点(不在页面跑重合并),命令与状态人话由
                    # summarize_export 单一来源输出,与 CLI train-export 同源同词汇。
                    with st.expander("📦 合并导出：把这次训练的模型带出工作台"):
                        from src.workbench.model_export import (
                            default_export_dir,
                            plan_model_export,
                        )

                        export_plan = plan_model_export(
                            run,
                            default_export_dir(Path(training_service.root), run_id),
                        )
                        from src.workbench.report_summary import summarize_export

                        for line in summarize_export(export_plan):
                            st.caption(line)
                    with st.expander("这次训练花了多少成本（如实估算）"):
                        from src.utils.platform_utils import get_platform
                        from src.workbench.cost_summary import cost_lines, summarize_run_cost

                        api_price = st.number_input(
                            "对比用 API 单价（元/百万 token，留 0 不对比）",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"api_price_{run_id}",
                        )
                        monthly = st.number_input(
                            "预计月调用量（次，0 不对比）",
                            min_value=0,
                            value=0,
                            step=10000,
                            key=f"api_queries_{run_id}",
                        )
                        account = summarize_run_cost(
                            run,
                            device=get_platform().device,
                            api_price_per_million_tokens=api_price or None,
                            expected_monthly_queries=monthly or None,
                        )
                        for line in cost_lines(account):
                            st.write(line)
                        with st.expander("原始成本数据"):
                            st.json(account)
                    from src.workbench.business_evaluation import (
                        BusinessEvaluationService,
                        EvaluationModel,
                        EvaluationProtocol,
                    )

                    evaluation_service = BusinessEvaluationService(
                        PROJECT_ROOT / "outputs/workbench/evaluations"
                    )
                    st.write("**基座与本轮微调：同开发集对照**")
                    st.caption(
                        "顺序加载本地模型版本，使用同一固定开发集、输入模板与生成规则。独立测试集保留给最终验收。"
                    )
                    if (
                        session.dataset is not None
                        and run.get("dataset_version") == session.dataset.version
                        and session.analysis
                        and session.analysis.recipe
                        and not run_execution_managed
                    ):
                        recipe = session.analysis.recipe
                        scorer = (
                            "json_fields_exact"
                            if recipe.output_format == "json"
                            else "classification_exact"
                            if len(recipe.targets) == 1
                            and recipe.targets[0].value_kind == "categorical"
                            else "open_review"
                        )
                        custom_scoring = select_business_scoring(key=f"eval_scoring_{run_id}")
                        if custom_scoring:
                            scorer = "custom_rules"
                        scorer_labels = {
                            "classification_exact": "类别严格匹配",
                            "json_fields_exact": "已声明 JSON 字段逐项严格匹配",
                            "open_review": "开放任务：保留输出，等待业务评分",
                            "custom_rules": "已确认业务规则：业务分均值与单题通过率",
                        }
                        st.write("**评分方式：** " + scorer_labels[scorer])
                        parent_runs = {}
                        if getattr(session.dataset, "evaluation_suite", None):
                            parent_runs = {
                                item["run_id"]: item
                                for item in training_service.list_runs()
                                if item["status"] == "succeeded" and item["run_id"] != run_id
                            }
                        with st.form(f"evaluate_run_{run_id}"):
                            parent_choice = ""
                            if run_iteration:
                                parent_choice = run_iteration["parent_run_id"]
                                st.write(
                                    f"本轮已锁定父轮模型：{parent_choice}。将比较基座、父轮与本轮三个版本。"
                                )
                            elif parent_runs:
                                parent_choice = st.selectbox(
                                    "同固定题集加入上一轮模型",
                                    ["", *parent_runs],
                                    format_func=lambda identity, choices=parent_runs: (
                                        "仅比较基座与本轮"
                                        if not identity
                                        else f"父轮 {identity} · {choices[identity].get('dataset_version', '')[:12]}"
                                    ),
                                    key=f"compare_parent_{run_id}",
                                )
                            max_new_tokens = st.number_input(
                                "每条回答最多生成 token 数",
                                min_value=1,
                                value=256,
                                step=1,
                                key=f"eval_tokens_{run_id}",
                            )
                            strip_whitespace = st.checkbox(
                                "精确比较时忽略首尾空白",
                                value=True,
                                key=f"eval_whitespace_{run_id}",
                            )
                            compare_models = st.form_submit_button("比较基座与本轮微调效果")
                        if compare_models:
                            try:
                                with st.spinner("正在使用相同开发集顺序生成并比较实际输出…"):
                                    comparison_models = [EvaluationModel("基座", run["model_path"])]
                                    if parent_choice:
                                        parent_run = training_service.get_status(parent_choice)
                                        if parent_run["status"] != "succeeded":
                                            raise ValueError(
                                                "父轮模型尚未成功完成训练，无法进行三模型对照。"
                                            )
                                        comparison_models.append(
                                            EvaluationModel(
                                                "父轮模型",
                                                parent_run["model_path"],
                                                adapter_path=parent_run["output_dir"],
                                            )
                                        )
                                    comparison_models.append(
                                        EvaluationModel(
                                            "本轮微调",
                                            run["model_path"],
                                            adapter_path=run["output_dir"],
                                        )
                                    )
                                    comparison_report = evaluation_service.compare(
                                        service.load(session.session_id),
                                        comparison_models,
                                        EvaluationProtocol(
                                            scorer=scorer,
                                            **(
                                                {"custom_scoring": custom_scoring}
                                                if custom_scoring
                                                else {}
                                            ),
                                            fields=tuple(target.label for target in recipe.targets)
                                            if scorer == "json_fields_exact"
                                            else (),
                                            max_new_tokens=int(max_new_tokens),
                                            strip_whitespace=strip_whitespace,
                                        ),
                                    )
                                    if run_iteration and run_iteration["status"] in {
                                        "prepared",
                                        "running",
                                    }:
                                        iteration_service.bind_evaluation(
                                            run_iteration["iteration_id"],
                                            service.load(session.session_id),
                                            comparison_report.evaluation_id,
                                        )
                                st.rerun()
                            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                                st.error(str(exc))
                    elif run_execution_managed:
                        st.info("后台正在沿用已确认的对照协议推进，完成后在此展示结果。")
                    else:
                        st.info("当前任务数据已变化；以下只展示该训练版本已保存的对照报告。")
                    try:
                        reports = [
                            report
                            for report in evaluation_service.list_reports(
                                dataset_version=run.get("dataset_version")
                            )
                            if any(
                                model_result["requested_model"].get("adapter_path")
                                == run.get("output_dir")
                                for model_result in report.models
                            )
                        ]
                        for comparison in reports:
                            with st.expander(
                                f"开发集对照 · {comparison.evaluation_id[:8]} · {comparison.created_at}",
                                expanded=True,
                            ):
                                show_business_comparison(comparison, key=run_id)
                    except (ValueError, OSError) as exc:
                        st.error(f"无法读取已保存评测：{exc}")
                    if not run_execution_managed:
                        show_final_acceptance(run)

st.download_button(
    "下载当前分析与预览记录",
    session.model_dump_json(indent=2),
    file_name=f"intake-{session.session_id[:8]}.json",
    mime="application/json",
)
