"""Archive cards should not hide failed generations in their denominator."""

from streamlit.testing.v1 import AppTest

from tests.unit.test_case_comparison_evidence_ui import failed_handoff
from ui.showcase import available_history_catalog


def render(catalog):
    from ui.components.archive_cards import render_archive_cards

    render_archive_cards(catalog)


def test_failed_results_are_disclosed_in_the_archive_card(tmp_path):
    target, data = failed_handoff(tmp_path)
    before = target.read_bytes()
    catalog, _ = available_history_catalog(tmp_path)
    assert [entry["id"] for entry in catalog] == ["r149"]
    assert [result["failed"] for result in catalog[0]["results"]] == [0, 0, 1, 1]
    page = AppTest.from_function(render, args=(catalog,), default_timeout=30).run()
    assert not page.exception, [item.message for item in page.exception]
    progress = [item.proto.text for item in page.get("progress")]
    for index in (2, 3):
        label = data["report"]["models"][index]["label"]
        assert progress[index] == f"{label} · 答对 7 / 8 · 截断 0 题 · 处理失败 1 题"
    assert "处理失败" not in progress[0]
    assert target.read_bytes() == before
