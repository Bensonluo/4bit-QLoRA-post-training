"""盲标核验页面流:抽取题目隐藏答案 → 逐条作答 → 判定与不一致展示。"""

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui
from tests.unit.test_full_data import FULL, approved

data_page = intake_ui.data_page


@pytest.fixture()
def verify_page(data_page):
    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    return service, session, page


def test_blind_verification_flow_via_page(verify_page):
    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("盲标核验" in header.value for header in page.subheader)
    # 尚未核验:先抽取题目
    assert any(button.label == "抽取盲标核验题目" for button in page.button)
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    assert not page.exception

    # 题目展示且不含答案;用业务正确答案作答(来自全量预览的目标值)
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    inputs = [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]
    assert inputs
    for field in inputs:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert not page.exception
    assert any("盲标核验已通过" in message.value for message in page.success)
    current = service.load(session.session_id)
    assert current.label_verification["verdict"] == "verified"
    # 通过后不再显示抽取入口
    assert not any(b.label == "抽取盲标核验题目" for b in page.button)


def test_verified_result_shows_statistical_lower_bound(verify_page):
    """核验结论附统计局限说明:抽题前预告下界,通过后展示 95% 下界而非冒充 100%。"""
    from src.workbench.intake_service import wilson_lower_bound

    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    assert not page.exception
    # 抽题后先如实预告:本轮即使全部一致,95% 置信下真实一致率下界也有限
    assert any("即使全部一致" in c.value for c in page.caption)
    assert any("下界才是你能依赖的数" in c.value for c in page.caption)

    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    for field in [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert not page.exception
    success = next(m.value for m in page.success if "盲标核验已通过" in m.value)
    assert "下界才是你能依赖的数" in success
    size = service.load(session.session_id).label_verification["sample_size"]
    assert f"{wilson_lower_bound(size, size):.0%}" in success


def test_mismatch_page_shows_lower_bound_caption(verify_page):
    """未通过时页面同样给出统计说明:4/5 这类观测一致率不粉饰证据强度。"""
    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    fields = [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]
    fields[0].input("故意答错").run()
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    for field in fields[1:]:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert not page.exception
    assert any("盲标核验未通过" in message.value for message in page.error)
    assert any("下界才是你能依赖的数" in c.value for c in page.caption)


def test_sample_size_selection_guidance_and_honest_shortfall(verify_page):
    """抽取表单引导样本量选择:默认 5 是快速关卡,高风险建议 20+;选超了如实说明不足。"""
    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    size_input = next(n for n in page.number_input if "核验样本量" in n.label)
    assert size_input.value == 5, "默认 5 条:快速关卡"
    guidance = next(c.value for c in page.caption if "核验强度由你按业务风险决定" in c.value)
    assert "默认 5 条：快速关卡" in guidance
    assert "高风险业务需要更强证据" in guidance
    assert "20 条以上" in guidance

    # 选择 20 条但已标注行只有 3 条:抽题后如实说明不足,不静默按更小量缩水
    size_input.set_value(20).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    assert not page.exception
    pending = page.session_state["pending_label_verification"]
    assert pending["requested_sample_size"] == 20
    assert pending["sample_size"] < 20
    assert any("不足你选择的 20 条" in c.value for c in page.caption)
    assert any("即使全部一致" in c.value for c in page.caption)

    # 统计说明按实际抽样条数计算:下界来自截断后的条数,不按请求的 20 条夸大证据
    from src.workbench.intake_service import wilson_lower_bound

    actual = pending["sample_size"]
    note = next(c.value for c in page.caption if "即使全部一致" in c.value)
    assert f"本轮 {actual} 条" in note
    assert "本轮 20 条" not in note
    assert f"{wilson_lower_bound(actual, actual):.0%}" in note
    # 本轮核验标题同样用实际条数,与统计说明一致
    assert any(f"本轮核验（{actual} 条" in m.value for m in page.markdown)


def test_old_archive_without_statistical_fields_recomputes_note(verify_page):
    """旧存档记录没有 evidence_note/agreement_lower_bound 时,回读页面按同口径现算统计说明。

    模拟 2374e43 之前的存档:result JSON 只含 matched/sample_size 等旧字段;
    页面回读不得静默丢掉统计说明,通过与未通过两种回读都现算。
    """
    import json
    import sqlite3

    from src.workbench.intake_service import agreement_evidence_note

    def strip_statistical_fields(service, *, expect_fields: bool):
        """把已完成核验的存档结果抹成旧版格式(去掉两个新字段)。

        expect_fields 只在第一次抹除时为 True:断言当前实现确实存了这两个字段。
        """
        with sqlite3.connect(service.database) as connection:
            stored = connection.execute(
                "SELECT result FROM label_verifications WHERE status='completed'"
            ).fetchall()
        assert stored and all(row[0] for row in stored)
        with sqlite3.connect(service.database) as connection:
            for (result_json,) in stored:
                legacy = json.loads(result_json)
                if expect_fields:
                    assert "evidence_note" in legacy and "agreement_lower_bound" in legacy
                legacy.pop("evidence_note", None)
                legacy.pop("agreement_lower_bound", None)
                connection.execute(
                    "UPDATE label_verifications SET result=? WHERE status='completed'",
                    (json.dumps(legacy, ensure_ascii=False),),
                )

    def answer_round_via_page(page, session, *, wrong_first=False):
        page.run()
        page.selectbox(key="intake_select").select(session.session_id).run()
        next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
        assert not page.exception
        targets = {row.row_id: row.target for row in session.full_data.preview.rows}
        fields = [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]
        for index, field in enumerate(fields):
            row_id = str(field.key).rsplit("_", 1)[-1]
            field.input("故意答错" if wrong_first and index == 0 else targets[row_id]).run()
        next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
        assert not page.exception

    def fresh_page(session):
        fresh = intake_ui.AppTest.from_file(str(intake_ui.PAGE), default_timeout=20)
        fresh.run()
        fresh.selectbox(key="intake_select").select(session.session_id).run()
        assert not fresh.exception
        return fresh

    service, session, page = verify_page
    # 通过态旧记录回读:success 按旧字段现算统计说明
    answer_round_via_page(page, session)
    assert any("盲标核验已通过" in m.value for m in page.success)
    strip_statistical_fields(service, expect_fields=True)
    record = service.load(session.session_id).label_verification
    assert "evidence_note" not in record and "agreement_lower_bound" not in record
    fresh = fresh_page(session)
    success = next(m.value for m in fresh.success if "盲标核验已通过" in m.value)
    assert f"{record['matched']}/{record['sample_size']} 一致" in success
    assert agreement_evidence_note(record["matched"], record["sample_size"]) in success
    assert "下界才是你能依赖的数" in success

    # 未通过态旧记录回读:error 后的 caption 同样现算,不粉饰不一致的证据强度
    pending = service.start_label_verification(session.session_id, session.revision)
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    answers = {item["row_id"]: targets[item["row_id"]] for item in pending["items"]}
    answers[pending["items"][0]["row_id"]] = "旧记录里也是答错的"
    service.submit_label_verification(session.session_id, pending["verification_id"], answers)
    strip_statistical_fields(service, expect_fields=False)
    record = service.load(session.session_id).label_verification
    assert record["verdict"] == "insufficient_agreement"
    assert "evidence_note" not in record and "agreement_lower_bound" not in record
    fresh = fresh_page(session)
    assert any("盲标核验未通过" in m.value for m in fresh.error)
    assert agreement_evidence_note(record["matched"], record["sample_size"]) in [
        c.value for c in fresh.caption
    ]
    assert any("下界才是你能依赖的数" in c.value for c in fresh.caption)


def test_mismatch_shows_per_row_differences_and_blocks_training(verify_page):
    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    fields = [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]
    fields[0].input("故意答错").run()
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    for field in fields[1:]:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert not page.exception
    assert any("盲标核验未通过" in message.value for message in page.error)
    assert any("故意答错" in block.label for block in page.expander)
    current = service.load(session.session_id)
    assert current.label_verification["verdict"] == "insufficient_agreement"


def test_contrast_check_via_page(data_page, monkeypatch):
    """预览确认前出现配对对比;配对正确后展示通过状态。"""
    import src.agent.intake
    from tests.unit.test_data_intake import analysis, model_for

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "联合分析目标与数据").click().run()
    assert not page.exception
    assert any("对比核验" in header.value for header in page.subheader)
    next(b for b in page.button if b.label == "开始配对对比").click().run()
    assert not page.exception
    # 用数据中的真实目标完成配对
    targets = {row.row_id: row.target for row in service.load(session.session_id).preview.rows}
    boxes = [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]
    assert len(boxes) == 2
    for box in boxes:
        row_id = str(box.key).rsplit("_", 1)[-1]
        box.select(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("第一轮配对正确" in message.value for message in page.info)

    # 第二轮(换题):二连对后才显示通过
    next(b for b in page.button if b.label == "开始配对对比").click().run()
    boxes = [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]
    for box in boxes:
        row_id = str(box.key).rsplit("_", 1)[-1]
        box.select(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验二连对" in message.value for message in page.success)

    # 二连对后核验已达标;强制的表单收进可选 expander,第三轮不强制
    expander = next(e for e in page.expander if "可选" in e.label and "对比核验" in e.label)
    optional_buttons = [b for b in expander.button if b.label == "开始配对对比"]
    assert optional_buttons
    next(b for b in page.button if b.label == "开始配对对比").click().run()
    assert not page.exception
    boxes = [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]
    for box in boxes:
        row_id = str(box.key).rsplit("_", 1)[-1]
        box.select(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验3轮连胜" in message.value for message in page.success)


def test_contrast_check_third_round_is_optional_entry(data_page, monkeypatch):
    """二连对后不再强制配对:核验入口收进可选 expander,直接确认不受阻。"""
    import src.agent.intake
    from tests.unit.test_data_intake import analysis, model_for

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "联合分析目标与数据").click().run()
    targets = {row.row_id: row.target for row in service.load(session.session_id).preview.rows}
    for _ in range(2):
        next(b for b in page.button if b.label == "开始配对对比").click().run()
        for box in [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]:
            row_id = str(box.key).rsplit("_", 1)[-1]
            box.select(targets[row_id]).run()
        next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验二连对" in message.value for message in page.success)
    # 可选入口在,但默认不展开、不强制;确认按钮可用
    assert any("可选" in e.label and "对比核验" in e.label for e in page.expander)
    next(c for c in page.checkbox if c.label.startswith("已核对预览")).check().run()
    assert not next(b for b in page.button if b.label == "确认当前转换含义").disabled


def test_contrast_meeting_copy_matches_optional_behavior(data_page, monkeypatch):
    """二连对达标后文案与行为一致:入口说「可选提高置信度」,不说「需要」。"""
    import src.agent.intake
    from tests.unit.test_data_intake import analysis, model_for

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "联合分析目标与数据").click().run()
    targets = {row.row_id: row.target for row in service.load(session.session_id).preview.rows}
    for _ in range(2):
        next(b for b in page.button if b.label == "开始配对对比").click().run()
        for box in [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]:
            row_id = str(box.key).rsplit("_", 1)[-1]
            box.select(targets[row_id]).run()
        next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("对比核验二连对" in message.value for message in page.success)

    expander = next(e for e in page.expander if "可选" in e.label and "对比核验" in e.label)
    assert "提高置信度" in expander.label
    assert "不强制" in expander.label
    assert "需要" not in expander.label
    captions = [c.value for c in expander.caption]
    assert any("核验已达标" in value for value in captions)
    assert any("不需要再核验" in value for value in captions)
    # 未达标时的「再配一组」提示不再出现——达标后页面上没有强制性文案
    assert not any("再配一组不同的题" in info.value for info in page.info)


def test_contrast_unanswered_selection_is_not_recorded(data_page, monkeypatch):
    """配对题默认不预选答案:漏选提交不记录核验轮次,补选后照常完成判定。"""
    import src.agent.intake
    from tests.unit.test_data_intake import analysis, model_for

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "联合分析目标与数据").click().run()
    next(b for b in page.button if b.label == "开始配对对比").click().run()
    assert not page.exception

    targets = {row.row_id: row.target for row in service.load(session.session_id).preview.rows}
    boxes = [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]
    assert len(boxes) == 2
    assert all(box.value == "" for box in boxes), "默认不预选任何答案,瞎点不算配对证据"
    keys = [box.key for box in boxes]
    row_ids = [str(key).rsplit("_", 1)[-1] for key in keys]

    # 只答一题就提交:漏选不算作答,也不记录核验轮次(连胜不被误伤)
    boxes[0].select(targets[row_ids[0]]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("请为每条输入选择答案" in message.value for message in page.error)
    assert service.contrast_check_status(session.session_id) is None

    # 补选另一题后照常提交,这一轮正常判定为第一轮配对正确
    page.selectbox(key=keys[1]).select(targets[row_ids[1]]).run()
    next(b for b in page.button if b.label == "提交配对").click().run()
    assert not page.exception
    assert any("第一轮配对正确" in message.value for message in page.info)
    status = service.contrast_check_status(session.session_id)
    assert status and status["verdict"] == "verified" and status["streak"] == 1


def test_contrast_history_rounds_visible_after_two_rounds(data_page, monkeypatch):
    """二连对后页面展示轮次历史:第几轮/题目/选择/对错逐轮可查,不是一句口号。"""
    import src.agent.intake
    from tests.unit.test_data_intake import analysis, model_for

    service, session, page = data_page
    monkeypatch.setattr(
        src.agent.intake, "CompatibleChatClient", lambda *a, **kw: model_for(analysis())
    )
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "联合分析目标与数据").click().run()
    assert not page.exception
    targets = {row.row_id: row.target for row in service.load(session.session_id).preview.rows}
    inputs = {row.row_id: row.input for row in service.load(session.session_id).preview.rows}

    def history_table():
        tables = [
            frame.value for frame in page.dataframe if "轮次" in getattr(frame.value, "columns", ())
        ]
        assert tables, "对比核验区应渲染轮次历史表"
        return tables[-1]

    # 没做过核验时不渲染空历史表
    assert not any("轮次" in getattr(f.value, "columns", ()) for f in page.dataframe)

    for expected_rounds in ([1], [1, 2]):
        next(b for b in page.button if b.label == "开始配对对比").click().run()
        for box in [s for s in page.selectbox if s.key and str(s.key).startswith("cc_")]:
            row_id = str(box.key).rsplit("_", 1)[-1]
            box.select(targets[row_id]).run()
        next(b for b in page.button if b.label == "提交配对").click().run()
        assert not page.exception
        table = history_table()
        # 每轮两道题:历史表一行一题,轮次按题重复
        assert list(table["轮次"]) == [
            f"第 {number} 轮" for number in expected_rounds for _ in range(2)
        ]
        assert list(table["对错"]) == ["对"] * (2 * len(expected_rounds))
        assert all(value in set(targets.values()) for value in table["正确答案"])
        # 题目列展示的是预览行的真实输入原文
        assert all(value in set(inputs.values()) for value in table["题目"])

    # 二连对达标后历史依然可见(有据可查),两轮四题都在表里
    assert any("对比核验二连对" in message.value for message in page.success)
    final_table = history_table()
    assert list(dict.fromkeys(final_table["轮次"])) == ["第 1 轮", "第 2 轮"]
    assert len(final_table) == 4


def test_stale_warning_renders_after_revision(verify_page):
    """数据修订后,页面明确显示「核验已失效请重验」,而不是静默回到初始状态。"""
    from copy import deepcopy

    from src.workbench.intake_models import Transform

    service, session, page = verify_page
    # 先完成一轮核验(通过)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    for field in [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input(targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert any("盲标核验已通过" in m.value for m in page.success)

    # 修订配方(新增 replace 转换)后重新确认全量 → 核验失效
    analysis = deepcopy(session.analysis)
    analysis.recipe.inputs[0].transforms.append(Transform(operation="replace", old="x", new="y"))
    service.apply_analysis(session, analysis)
    # 修订后重新确认全量,使核验进入 stale 态(与旅程一致)
    revised = service.load(session.session_id)
    revised = service.confirm(revised.session_id, revised.revision)
    revised = service.validate_full_data(revised.session_id, revised.revision, "full.csv", FULL)
    revised = service.confirm_full_data(revised.session_id, revised.revision)
    # 新会话读取:同会话内控件序列随新增控件变化会触发 AppTest 的状态清理怪癖
    fresh = intake_ui.AppTest.from_file(str(intake_ui.PAGE), default_timeout=20)
    fresh.run()
    fresh.selectbox(key="intake_select").select(session.session_id).run()
    assert not fresh.exception
    assert any("已失效" in w.value for w in fresh.warning)


def test_insufficient_verification_page_renders_stored_triage_lines(verify_page):
    """未通过页渲染记录内三因分辨行(词汇方向+对号修正),与页面通用指引并存。"""
    service, session, page = verify_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    next(b for b in page.button if b.label == "抽取盲标核验题目").click().run()
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    fields = [t for t in page.text_input if t.key and str(t.key).startswith("lv_")]
    first_row = str(fields[0].key).rsplit("_", 1)[-1]
    for field in fields:
        row_id = str(field.key).rsplit("_", 1)[-1]
        field.input("明显不同的答案" if row_id == first_row else targets[row_id]).run()
    next(b for b in page.button if b.label == "提交盲标核验答案").click().run()
    assert not page.exception
    assert any("盲标核验未通过" in m.value for m in page.error)

    # 记录里存了三因分辨,页面以 caption 渲染同一份行
    record = service.load(session.session_id).label_verification
    assert record["mismatch_triage"]
    rendered = [c.value for c in page.caption]
    assert any("全部标签里没有出现过" in text for text in rendered)
    assert any(text.startswith("对号修正") for text in rendered)
    # 通用指引保留:三因词汇仍出现在页面,分辨行是补充而不是替换
    assert any("标签错误、业务歧义或任务定义不清" in text for text in rendered)
