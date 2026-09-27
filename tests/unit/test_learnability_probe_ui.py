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
