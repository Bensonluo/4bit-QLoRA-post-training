"""All public handoff views must honor the same verified group evidence."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.showcase import R149_SOURCE, load_showcase

ROOT = Path(__file__).resolve().parents[2]


def render_summary(root):
    from ui.components.showcase import render_history_summary

    render_history_summary(root)


def render_history(root):
    from ui.components.showcase import render_history

    render_history(root)


def render_delivery(root):
    from ui.components.delivery_archive import render_delivery_archive

    render_delivery_archive(root)


@pytest.mark.parametrize(
    "damage",
    ["missing_preparation", "wrong_preparation", "missing_group", "string_group", "string_source"],
)
def test_unknown_group_evidence_is_consistent_and_detail_remains_browsable(tmp_path, damage):
    target = tmp_path / "docs/validation" / R149_SOURCE
    target.parent.mkdir(parents=True)
    data = json.loads((ROOT / "docs/validation" / R149_SOURCE).read_bytes())
    if damage == "missing_preparation":
        del data["preparation"]["development_groups"]
    elif damage == "wrong_preparation":
        data["preparation"]["development_groups"] = ["not-the-recorded-groups"]
    else:
        for model in data["actual_report"]["models"]:
            for row in model["rows"]:
                if damage == "missing_group":
                    row["source"].pop("group")
                elif damage == "string_group":
                    row["source"]["group"] = "unverified"
                else:
                    row["source"] = "unverified"
    target.write_text(json.dumps(data, ensure_ascii=False))
    before = target.read_bytes()
    assert load_showcase(tmp_path)["counts"]["development_groups"] is None
    for view in (render_summary, render_history, render_delivery):
        page = AppTest.from_function(view, args=(tmp_path,), default_timeout=30).run()
        assert not page.exception, [item.message for item in page.exception]
        captions = "\n".join(item.value for item in page.caption)
        assert "8 道开发题 · 分组信息不可核验" in captions
        assert "None 个分组" not in captions
        assert not any("个分组" in item.value for item in page.caption)
        assert page.metric
        if view is render_history:
            assert page.dataframe[0].value["业务分组"].eq("未记录或不可核验").all()
            assert page.code
    assert target.read_bytes() == before


def test_verified_group_count_and_case_labels_remain_available():
    data = load_showcase(ROOT)
    assert data["counts"]["development_groups"] == 2
    for view in (render_summary, render_history, render_delivery):
        page = AppTest.from_function(view, args=(ROOT,), default_timeout=30).run()
        assert not page.exception
        assert any("8 道开发题 · 2 个分组" in item.value for item in page.caption)
        if view is render_history:
            assert page.dataframe[0].value["业务分组"].nunique() == 2


@pytest.mark.parametrize("group", [{}, {"name": " "}, {"name": 5}, {"a": "x", "b": "y"}])
def test_case_table_does_not_render_blank_or_malformed_group(group):
    from ui.components.case_comparison import case_overview

    models = load_showcase(ROOT)["report"]["models"]
    models[0]["rows"][0]["source"]["group"] = group
    table = case_overview(models)
    assert table.iloc[0]["业务分组"] == "未记录或不可核验"
    assert table.iloc[1]["业务分组"] != "未记录或不可核验"
