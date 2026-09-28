"""探针标签问题候选的页面呈现:有候选出表,无候选出提示。"""

import pytest

pytest.importorskip("streamlit")

from tests.unit import test_data_intake_ui as intake_ui
from tests.unit.test_full_data import FULL, approved

data_page = intake_ui.data_page


@pytest.fixture()
def probe_page(data_page):
    """推进到 ready_for_training_preflight(探针区只在物化后渲染)。"""
    service, _, page = data_page
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)
    return service, session, page


def _probe_result(candidates, version="test-version"):
    return {
        "kind": "learnability_probe",
        "dataset_version": version,
        "model_path": "/tmp/local-base",
        "zero_shot_accuracy": 0.25,
        "majority_baseline": 0.5,
        "difference": -0.25,
        "note": "证据说明。",
        "candidates_note": "候选不等于错误——模型可能错。",
        "label_error_candidates": candidates,
    }


def test_probe_candidates_table_with_strong_evidence_first(probe_page, monkeypatch):
    service, session, page = probe_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    textbox = next(t for t in page.text_input if "本地基础模型目录" in t.label)
    textbox.input("/tmp/local-base").run()

    import ui.pages  # noqa: F401  确保模块路径已初始化

    captured = {}

    def fake_probe(current, model_path, **kwargs):
        captured["called"] = True
        return _probe_result(
            [
                {
                    "row_id": "r000002",
                    "data_label": "物流",
                    "base_zero_shot": "基座输出A",
                    "user_blind_answer": "用户答案B",
                    "evidence": "强证据,优先人工核对",
                },
                {
                    "row_id": "r000001",
                    "data_label": "质量",
                    "base_zero_shot": "基座输出B",
                    "user_blind_answer": None,
                    "evidence": "弱信号供参考",
                },
            ]
        )

    import src.workbench.learnability_probe as probe_module

    monkeypatch.setattr(probe_module, "probe_learnability", fake_probe)
    button = next(b for b in page.button if "运行可学性探针" in b.label)
    button.click().run()
    assert not page.exception
    assert captured.get("called")
    assert any("标签问题候选" in h.value for h in page.subheader)
    # 候选表已渲染(st.dataframe 存在),空提示不出现即证明走了候选分支
    assert len(page.dataframe) >= 1
    assert not any("没有发现值得优先核对的行" in i.value for i in page.info)


def test_probe_no_candidates_shows_clean_info(probe_page, monkeypatch):
    service, session, page = probe_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    textbox = next(t for t in page.text_input if "本地基础模型目录" in t.label)
    textbox.input("/tmp/local-base").run()

    import src.workbench.learnability_probe as probe_module

    monkeypatch.setattr(
        probe_module, "probe_learnability", lambda current, model_path, **kw: _probe_result([])
    )
    next(b for b in page.button if "运行可学性探针" in b.label).click().run()
    assert not page.exception
    assert any("没有发现值得优先核对的行" in i.value for i in page.info)


def _candidate(index, *, blind=None):
    return {
        "row_id": f"r{index:06d}",
        "data_label": f"标签{index}",
        "base_zero_shot": f"基座输出{index}",
        "user_blind_answer": blind,
        "evidence": "强证据,优先人工核对" if blind else "弱信号供参考",
    }


def test_probe_candidates_over_twenty_paginate_with_honest_counts(probe_page, monkeypatch):
    """候选 >20 分页展示:每页 20,强证据在先不被分页打乱,总数说明始终如实。"""
    service, session, page = probe_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    textbox = next(t for t in page.text_input if "本地基础模型目录" in t.label)
    textbox.input("/tmp/local-base").run()

    candidates = [_candidate(1, blind="用户答案A"), _candidate(2), _candidate(3, blind="用户答案C")]
    candidates += [_candidate(index) for index in range(4, 26)]  # 共 25 条 → 两页

    import src.workbench.learnability_probe as probe_module

    monkeypatch.setattr(
        probe_module,
        "probe_learnability",
        lambda current, model_path, **kw: _probe_result(
            candidates, version=session.dataset.version
        ),
    )
    next(b for b in page.button if "运行可学性探针" in b.label).click().run()
    assert not page.exception
    pager = next(n for n in page.number_input if n.label == "候选预览页码")
    assert pager.value == 1
    assert any("第 1/2 页" in item.value for item in page.caption)
    assert any("共 25 条候选" in item.value for item in page.caption)

    def visible_ids():
        table = next(frame.value for frame in page.dataframe if "证据" in frame.value.columns)
        return list(table["行ID"]), list(table["你的盲标答案"])

    ids, blinds = visible_ids()
    assert ids == [f"r{index:06d}" for index in (1, 3, 2)] + [
        f"r{index:06d}" for index in range(4, 21)
    ]
    assert blinds[:3] == ["用户答案A", "用户答案C", "—"], "强证据在先不因分页打乱"

    pager.set_value(2).run()
    assert not page.exception
    assert any("第 2/2 页" in item.value for item in page.caption)
    assert any("显示第 21–25 条，共 25 条候选" in item.value for item in page.caption)
    ids, _ = visible_ids()
    assert ids == [f"r{index:06d}" for index in range(21, 26)]

    # 探针区还应保留 CSV 导出入口,说明文本如实(导出含全部候选)
    assert any("导出候选为 CSV" in b.label for b in page.get("download_button"))
    assert any("包含全部候选" in item.value for item in page.caption)


def test_probe_candidates_twenty_exact_needs_no_pager(probe_page, monkeypatch):
    """恰好 20 条不出现分页控件,一次展示全部,计数如实。"""
    service, session, page = probe_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    textbox = next(t for t in page.text_input if "本地基础模型目录" in t.label)
    textbox.input("/tmp/local-base").run()

    import src.workbench.learnability_probe as probe_module

    monkeypatch.setattr(
        probe_module,
        "probe_learnability",
        lambda current, model_path, **kw: _probe_result(
            [_candidate(index) for index in range(1, 21)], version=session.dataset.version
        ),
    )
    next(b for b in page.button if "运行可学性探针" in b.label).click().run()
    assert not page.exception
    assert not any(n.label == "候选预览页码" for n in page.number_input)
    assert any("显示第 1–20 条，共 20 条候选" in item.value for item in page.caption)
    table = next(frame.value for frame in page.dataframe if "证据" in frame.value.columns)
    assert list(table["行ID"]) == [f"r{index:06d}" for index in range(1, 21)]


def test_probe_result_survives_page_rerun_from_saved_record(probe_page, monkeypatch):
    """探针结果已存盘:任何交互后回读最近一次结果,不需要重新加载模型重跑。"""
    service, session, page = probe_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    textbox = next(t for t in page.text_input if "本地基础模型目录" in t.label)
    textbox.input("/tmp/local-base").run()

    import src.workbench.learnability_probe as probe_module

    calls = []

    def fake_probe(current, model_path, **kwargs):
        calls.append(model_path)
        return _probe_result([], version=session.dataset.version)

    monkeypatch.setattr(probe_module, "probe_learnability", fake_probe)
    next(b for b in page.button if "运行可学性探针" in b.label).click().run()
    assert not page.exception
    assert len(calls) == 1

    # 再次整页渲染(模拟用户做了别的操作),探针区应回读已存盘结果,而不是空白
    page.run()
    assert not page.exception
    assert len(calls) == 1, "回看结果不应重新运行探针(模型加载成本高)"
    assert any("显示最近一次已保存的探针结果" in c.value for c in page.caption)
    assert any("没有发现值得优先核对的行" in i.value for i in page.info)


def test_probe_candidates_table_shows_source_hints(probe_page, monkeypatch):
    """候选表带「来源提示」列:行号取自全量资料,定位不到的行如实说明。"""
    service, session, page = probe_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    textbox = next(t for t in page.text_input if "本地基础模型目录" in t.label)
    textbox.input("/tmp/local-base").run()

    # FULL 文件第 1 行是表头:r000001 对应文件第 2 行
    line_of = {row.row_id: row.line for row in session.full_data.source.rows}
    assert line_of["r000001"] == 2

    import src.workbench.learnability_probe as probe_module

    monkeypatch.setattr(
        probe_module,
        "probe_learnability",
        lambda current, model_path, **kw: _probe_result(
            [
                {
                    "row_id": "r000002",
                    "data_label": "物流",
                    "base_zero_shot": "基座输出A",
                    "user_blind_answer": "用户答案B",
                    "evidence": "强证据,优先人工核对,建议对照原始来源行",
                },
                {
                    "row_id": "r099999",
                    "data_label": "质量",
                    "base_zero_shot": "基座输出B",
                    "user_blind_answer": None,
                    "evidence": "弱信号供参考",
                },
            ],
            version=session.dataset.version,
        ),
    )
    next(b for b in page.button if "运行可学性探针" in b.label).click().run()
    assert not page.exception
    table = next(frame.value for frame in page.dataframe if "证据" in frame.value.columns)
    assert "来源提示" in table.columns, "候选表应有来源提示列供人工溯源"
    hints = dict(zip(table["行ID"], table["来源提示"]))
    assert hints["r000002"] == f"全量资料第 {line_of['r000002']} 行"
    assert hints["r099999"] == "未在全量资料中定位到该行", "定位不到不编造行号"
    # 强证据行的证据列带溯源建议
    assert "建议对照原始来源行" in table.loc[table["行ID"] == "r000002", "证据"].iloc[0]
    # 计数如实:两条候选都展示
    assert any("共 2 条候选" in c.value for c in page.caption)


def test_probe_below_baseline_renders_triage_captions(probe_page, monkeypatch):
    """低于基线时页面逐行渲染方向分辨(与 CLI stderr 同一来源),不低于基线不渲染。"""
    service, session, page = probe_page
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    textbox = next(t for t in page.text_input if "本地基础模型目录" in t.label)
    textbox.input("/tmp/local-base").run()

    import src.workbench.learnability_probe as probe_module

    def fake_probe(current, model_path, **kwargs):
        result = _probe_result([])
        # 低于基线 + 词表外输出:命中「提示模板方向」分辨行
        result["label_vocabulary"] = ["yes", "no"]
        result["observations"] = [
            {
                "row_id": "r000003",
                "expected": "yes",
                "generated": "词表外的输出",
                "match": False,
                "truncated": False,
            }
        ]
        return result

    monkeypatch.setattr(probe_module, "probe_learnability", fake_probe)
    next(b for b in page.button if "运行可学性探针" in b.label).click().run()
    assert any("不在这份开发集的标签里出现过" in c.value for c in page.caption)
    assert any(c.value.startswith("对号处理") for c in page.caption)
