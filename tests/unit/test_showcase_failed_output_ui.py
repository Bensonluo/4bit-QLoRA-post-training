"""A retained generation failure remains visible without losing other evidence."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from ui.showcase import R149_SOURCE, load_showcase

ROOT = Path(__file__).resolve().parents[2]


def summary(root):
    from ui.components.showcase import render_history_summary

    render_history_summary(root)


def detail(root):
    from ui.components.showcase import render_history

    render_history(root)


def test_missing_generation_proof_keeps_failure_and_fixed_denominator(tmp_path):
    """Exercise a valid failure using a copy of the recorded report, never the source."""
    original = ROOT / "docs/validation" / R149_SOURCE
    data = json.loads(original.read_text())
    model = data["actual_report"]["models"][0]
    row = model["rows"][0]
    row.update(
        status="failed",
        output=None,
        token_proof=None,
        truncated=False,
        generated_tokens=0,
        error="模拟生成失败",
        correct=False,
    )
    model["metrics"].update(generated=7, generation_coverage=7 / 8)
    directory = tmp_path / "docs/validation"
    directory.mkdir(parents=True)
    (directory / R149_SOURCE).write_text(json.dumps(data, ensure_ascii=False))

    # The public loader accepts this as a failed record, not a missing archive.
    loaded = load_showcase(tmp_path)
    assert loaded["report"]["models"][0]["rows"][0]["status"] == "failed"
    assert loaded["report"]["models"][0]["metrics"]["total"] == 8

    page = AppTest.from_function(summary, args=(tmp_path,), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert page.metric[0].value == "0 / 8"
    assert any("完整结束 0 题 · 截断 7 题 · 处理失败 1 题" in c.value for c in page.caption)

    page = AppTest.from_function(detail, args=(tmp_path,), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert page.metric[0].value == "0 / 8"
    assert any("未生成" in c.value and "0 token" in c.value for c in page.caption)
    assert any("模拟生成失败" in c.value for c in page.caption)


def test_failed_record_with_output_is_not_labeled_as_a_normal_mismatch(tmp_path):
    original = ROOT / "docs/validation" / R149_SOURCE
    data = json.loads(original.read_text())
    model = data["actual_report"]["models"][1]
    row = model["rows"][0]
    raw_output = row["output"]
    row.update(status="failed", correct=False, error="模拟评分失败")
    model["metrics"].update(scored=7, scorable_coverage=7 / 8, exact_match=7 / 8)
    directory = tmp_path / "docs/validation"
    directory.mkdir(parents=True)
    (directory / R149_SOURCE).write_text(json.dumps(data, ensure_ascii=False))

    loaded = load_showcase(tmp_path)
    assert loaded["report"]["models"][1]["rows"][0]["output"] == raw_output
    page = AppTest.from_function(detail, args=(tmp_path,), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert page.metric[1].value == "7 / 8"
    assert any("处理失败" in c.value and "完整结束" in c.value for c in page.caption)
    assert any(c.value == "模拟评分失败" for c in page.caption)
    assert any(code.value == raw_output for code in page.code)
