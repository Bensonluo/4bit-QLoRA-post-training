"""Delivery cards keep valid failed evidence visible without inventing results."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.showcase import R149_SOURCE, load_showcase

ROOT = Path(__file__).resolve().parents[2]


def delivery(root):
    from ui.components.delivery_archive import render_delivery_archive

    render_delivery_archive(root)


@pytest.mark.parametrize("label", ["合并 HF", "GGUF Q4_K_M"])
@pytest.mark.parametrize("generated", [False, True])
def test_delivery_preserves_failed_case_and_fixed_denominator(tmp_path, label, generated):
    original = ROOT / "docs/validation" / R149_SOURCE
    data = json.loads(original.read_text())
    model = next(item for item in data["actual_report"]["models"] if item["label"] == label)
    row = model["rows"][0]
    row.update(status="failed", correct=False, error="测试处理失败")
    if not generated:
        row.update(output=None, token_proof=None, truncated=False, generated_tokens=0)
    model["metrics"].update(scored=7, scorable_coverage=7 / 8, exact_match=7 / 8)
    if not generated:
        model["metrics"].update(generated=7, generation_coverage=7 / 8)
    target = tmp_path / "docs/validation" / R149_SOURCE
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(data, ensure_ascii=False))
    before = target.read_bytes()

    verified = load_showcase(tmp_path)
    found = next(item for item in verified["report"]["models"] if item["label"] == label)
    assert found["rows"][0]["status"] == "failed"
    assert found["rows"][0]["output"] == row["output"]
    assert found["metrics"]["total"] == 8

    page = AppTest.from_function(delivery, args=(tmp_path,), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert [item.value for item in page.metric if item.label == "开发集严格匹配"] == (
        ["7 / 8", "8 / 8"] if label == "合并 HF" else ["8 / 8", "7 / 8"]
    )
    finish_count = 8 if generated else 7
    assert any(f"完整结束 {finish_count} 题 · 处理失败 1 题" in item.value for item in page.caption)
    assert target.read_bytes() == before
