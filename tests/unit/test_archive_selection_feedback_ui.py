"""Archive card selection remains visible and leads to the selected detail."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def page():
    result = AppTest.from_file(str(ROOT / "ui/views/08_Showcase.py"), default_timeout=45).run()
    assert not result.exception, [item.message for item in result.exception]
    return result


def test_card_and_dropdown_selection_show_same_record_and_detail_link():
    result = page()
    first = result.button(key="archive-select-r149")
    assert first.label == "当前查看"
    assert first.proto.type == "primary"
    assert result.selectbox(key="archive-selection").value == "r149"
    assert any(
        "前往已选详情" in item.value and "#archive-details" in item.value
        for item in result.markdown
    )

    result.button(key="archive-select-r148-product-smoke").click().run()
    assert not result.exception, [item.message for item in result.exception]
    assert result.selectbox(key="archive-selection").value == "r148-product-smoke"
    assert result.button(key="archive-select-r148-product-smoke").label == "当前查看"
    assert result.button(key="archive-select-r149").label == "查看记录"
    assert any(
        item.value == "已选记录" and item.proto.anchor == "archive-details"
        for item in result.subheader
    )

    result.selectbox(key="archive-selection").set_value("r148-controlled-v2").run()
    assert not result.exception, [item.message for item in result.exception]
    assert result.button(key="archive-select-r148-controlled-v2").label == "当前查看"
    assert result.button(key="archive-select-r148-product-smoke").label == "查看记录"
