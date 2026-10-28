"""Recorded pair differences and table navigation retain the original cases."""

from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.components.case_comparison import pair_facts, pair_indices
from ui.showcase import load_showcase

ROOT = Path(__file__).resolve().parents[2]


def render(data):
    from ui.components.case_comparison import render_case_comparison

    render_case_comparison(data, key_prefix="case")


def boot():
    return AppTest.from_function(render, args=(load_showcase(ROOT),)).run(timeout=30)


def test_overview_keeps_all_cases_groups_and_model_results():
    page = boot()
    assert not page.exception
    table = page.dataframe[0].value
    assert len(table) == 8
    assert table["业务分组"].nunique() == 2
    assert table["基础模型"].eq("… 截断").all()
    assert table["GGUF Q4_K_M"].eq("✓ 匹配").all()
    original = load_showcase(ROOT)["report"]["models"][0]["rows"][-1]
    page.selectbox(key="case-question").set_value(original["index"]).run()
    assert not page.exception
    assert original["prompt"].strip() in [item.value.strip() for item in page.code]
    assert original["output"].strip() in [item.value.strip() for item in page.code]


def test_equal_export_pair_has_no_differences_and_can_return_to_all_cases():
    page = boot()
    page.segmented_control(key="case-view").set_value("两模型对照").run()
    page.selectbox(key="case-left").set_value(2).run()
    page.selectbox(key="case-right").set_value(3).run()
    assert not page.exception
    assert any("8/8 题的原始文本与生成 token 均一致" in item.value for item in page.caption)
    assert len(page.dataframe[0].value) == 8
    assert list(page.dataframe[0].value.columns) == [
        "题目",
        "业务分组",
        "期望答案",
        "合并 HF",
        "GGUF Q4_K_M",
    ]
    page.selectbox(key="case-filter").set_value("输出或 token 不同").run()
    assert not page.exception
    assert any("当前显示 0/8" in item.value for item in page.caption)
    assert any("没有符合筛选条件" in item.value for item in page.info)
    assert not page.code
    page.button(key="case-reset-filter").click().run()
    assert not page.exception
    assert len(page.dataframe[0].value) == 8
    assert page.code
    assert "**合并 HF**" in [item.value for item in page.markdown]
    assert "**GGUF Q4_K_M**" in [item.value for item in page.markdown]
    assert "**基础模型**" not in [item.value for item in page.markdown]


def test_overview_follows_pair_selection_and_restores_all_models():
    page = boot()
    page.segmented_control(key="case-view").set_value("两模型对照").run()
    page.selectbox(key="case-left").set_value(0).run()
    page.selectbox(key="case-right").set_value(3).run()
    assert not page.exception
    table = page.dataframe[0].value
    assert list(table.columns) == ["题目", "业务分组", "期望答案", "基础模型", "GGUF Q4_K_M"]
    assert table["基础模型"].eq("… 截断").all()
    assert table["GGUF Q4_K_M"].eq("✓ 匹配").all()
    page.selectbox(key="case-filter").set_value("存在截断").run()
    assert not page.exception
    assert len(page.dataframe[0].value) == 8
    assert list(page.dataframe[0].value.columns) == list(table.columns)
    page.segmented_control(key="case-view").set_value("全部模型").run()
    assert not page.exception
    assert list(page.dataframe[0].value.columns) == [
        "题目",
        "业务分组",
        "期望答案",
        "基础模型",
        "训练适配器",
        "合并 HF",
        "GGUF Q4_K_M",
    ]


@pytest.mark.parametrize("criterion", ["输出或 token 不同", "至少一侧未通过", "存在截断"])
def test_filters_apply_to_current_pair_including_failed_base(criterion):
    models = load_showcase(ROOT)["report"]["models"]
    assert pair_indices(models, 0, 1, criterion) == list(range(8))
    assert pair_indices(models, 2, 3, criterion) == []


def test_missing_token_proof_is_unknown_and_same_text_does_not_hide_different_tokens():
    models = load_showcase(ROOT)["report"]["models"]
    left, right = deepcopy(models[2]["rows"][0]), deepcopy(models[3]["rows"][0])
    right.pop("token_proof")
    assert pair_facts(left, right)["same_tokens"] is None
    right["token_proof"] = deepcopy(left["token_proof"])
    right["token_proof"]["generated_token_ids"][0] += 1
    facts = pair_facts(left, right)
    assert facts["same_output"] is True
    assert facts["same_tokens"] is False


def test_table_selection_maps_filtered_position_and_does_not_override_later_dropdown(monkeypatch):
    import streamlit as st

    from ui.components import case_comparison

    # Dataframe selections are not yet exposed by AppTest; exercise its real
    # callback at the widget boundary, using non-contiguous source case IDs.
    monkeypatch.setattr(case_comparison, "pair_indices", lambda *args: [1, 5])
    once = []

    def select_row(data, **kwargs):
        if not once:
            once.append(True)
            st.session_state[kwargs["key"]] = {"selection": {"rows": [1]}}
            kwargs["on_select"]()

    monkeypatch.setattr(st, "dataframe", select_row)
    page = AppTest.from_function(render, args=(load_showcase(ROOT),))
    page.session_state["case-view"] = "两模型对照"
    page.run(timeout=30)
    assert not page.exception
    assert page.selectbox(key="case-question").value == 5
    page.selectbox(key="case-question").set_value(1).run()
    assert not page.exception
    assert page.selectbox(key="case-question").value == 1
    assert page.session_state["case-table-epoch"] == 1


def test_changing_left_model_keeps_two_distinct_models_selected():
    page = boot()
    page.segmented_control(key="case-view").set_value("两模型对照").run()
    page.selectbox(key="case-left").set_value(1).run()
    assert not page.exception
    assert page.selectbox(key="case-left").value != page.selectbox(key="case-right").value
