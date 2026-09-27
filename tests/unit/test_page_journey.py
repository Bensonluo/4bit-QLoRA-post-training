"""M2 页面旅程走查:九段旅程中,页面每一段都要有明确的下一步指引,没有断头路。

状态由真实服务推进(与用户操作等效),页面只负责呈现;每段断言:
① 无异常 ② 本段内容可见 ③ 下一步指引可见。走查失败即旅程断点。
"""

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui
from tests.unit.test_full_data import FULL

data_page = intake_ui.data_page
PAGE = intake_ui.PAGE


@pytest.fixture()
def walker(data_page):
    service, session, page = data_page
    return service, session, page


def _open(page, session):
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception, [e.message for e in page.exception]


def _reload(service, session):
    return service.load(session.session_id)


def test_page_journey_every_stage_shows_next_step(walker):
    from src.workbench.baseline_analysis import propose_baseline_analysis

    service, session, page = walker
    context = {"service": service, "session": session}

    def stage(title, cues, texts=()):
        _open(page, context["session"])
        joined = "\n".join(
            [b.value for b in page.markdown]
            + [w.value for w in page.warning]
            + [s.value for s in page.success]
            + [i.value for i in page.info]
            + [c.value for c in page.caption]
            + [h.value for h in page.subheader]
            + [b.label for b in page.button]
            + [a.label for a in page.text_area]
        )
        for cue in cues:
            assert cue in joined, f"[{title}] 缺少下一步指引/内容: {cue}"
        for text in texts:
            assert text in joined, f"[{title}] 缺少本段内容: {text}"

    # ① 建任务后(样例已读入): 无 Agent 也有入口
    stage("①已建任务", ["基础分析"], ["联合分析目标与数据"])

    # ② 基础分析后: 预览+对比核验+确认按钮
    context["session"] = service.apply_analysis(
        context["session"],
        propose_baseline_analysis(context["session"], target_column="类别", group_columns=["编号"]),
        model="baseline-deterministic",
    )
    stage("②已分析", ["对比核验", "确认当前转换含义"], ["真实转换预览"])

    # ③ 样例确认后: 全量数据入口
    context["session"] = service.confirm(context["session"].session_id, context["session"].revision)
    stage("③已确认样例", ["全量数据验证", "验证"], ["全量数据验证"])

    # ④ 全量验证后: 确认全量按钮
    context["session"] = service.validate_full_data(
        context["session"].session_id, context["session"].revision, "full.csv", FULL
    )
    stage("④已验证全量", ["确认全量数据含义"])

    # ⑤ 全量确认后: 盲标核验 + 分区入口
    context["session"] = service.confirm_full_data(context["session"].session_id, context["session"].revision)
    stage("⑤已确认全量", ["盲标核验", "生成独立训练与评测分区"])

    # ⑥ 盲标核验通过 + 物化后: 训练前检查
    pending = service.start_label_verification(context["session"].session_id, context["session"].revision)
    targets = {row.row_id: row.target for row in context["session"].full_data.preview.rows}
    service.submit_label_verification(
        context["session"].session_id,
        pending["verification_id"],
        {item["row_id"]: targets[item["row_id"]] for item in pending["items"]},
    )
    context["session"] = service.materialize_dataset(context["session"].session_id, context["session"].revision)
    stage(
        "⑥已物化",
        ["训练前检查", "可学性探针", "用当前数据微调模型"],
        ["盲标核验已通过"],
    )


def test_page_journey_agent_path_keeps_entry(walker):
    """Agent 路径与零密钥路径并列存在,不互为断头路。"""
    service, session, page = walker
    _open(page, session)
    labels = [b.label for b in page.button]
    assert any("联合分析目标与数据" in label for label in labels)
    assert any("基础分析" in e.label or "没有 Agent" in e.label for e in page.expander)
