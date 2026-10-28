"""Unknown output evidence is not counted as a confirmed pair difference."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.components.case_comparison import pair_facts, pair_indices
from ui.showcase import R149_SOURCE, load_showcase

ROOT = Path(__file__).resolve().parents[2]


def render(data):
    from ui.components.case_comparison import render_case_comparison

    render_case_comparison(data, key_prefix="evidence")


def failed_handoff(tmp_path):
    data = json.loads((ROOT / "docs/validation" / R149_SOURCE).read_text())
    for model in data["actual_report"]["models"][2:4]:
        model["rows"][0].update(
            status="failed",
            output=None,
            token_proof=None,
            truncated=False,
            generated_tokens=0,
            error="模拟生成失败",
            correct=False,
        )
        model["metrics"].update(
            scored=7,
            generated=7,
            exact_match=7 / 8,
            scorable_coverage=7 / 8,
            generation_coverage=7 / 8,
        )
    target = tmp_path / "docs/validation" / R149_SOURCE
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(data, ensure_ascii=False))
    return target, load_showcase(tmp_path)


def test_failed_outputs_are_unknown_and_can_be_found_in_comparison(tmp_path):
    target, data = failed_handoff(tmp_path)
    before = target.read_bytes()
    left, right = data["report"]["models"][2:4]
    assert pair_facts(left["rows"][0], right["rows"][0])["same_output"] is None
    assert pair_facts(left["rows"][0], right["rows"][0])["comparison"] == "unknown"
    assert pair_indices(data["report"]["models"], 2, 3, "输出或 token 不同") == []
    assert pair_indices(data["report"]["models"], 2, 3, "输出或 token 无法比较") == [0]

    page = AppTest.from_function(render, args=(data,), default_timeout=45).run()
    page.segmented_control(key="evidence-view").set_value("两模型对照").run()
    page.selectbox(key="evidence-left").set_value(2).run()
    page.selectbox(key="evidence-right").set_value(3).run()
    assert not page.exception, [item.message for item in page.exception]
    assert any(
        "7/8 题的原始文本与生成 token 均一致" in c.value and "1 题缺少输出或 token 证据" in c.value
        for c in page.caption
    )
    page.selectbox(key="evidence-filter").set_value("输出或 token 无法比较").run()
    assert not page.exception
    assert len(page.dataframe[0].value) == 1
    assert page.selectbox(key="evidence-question").value == 0
    assert sum("模拟生成失败" in item.value for item in page.caption) == 2
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ({"output": None}, {"output": None}, "unknown"),
        ({"output": None}, {"output": "answer"}, "unknown"),
        ({"output": "a"}, {"output": "b"}, "different"),
        ({"output": "a"}, {"output": "a"}, "unknown"),
        (
            {"output": "a", "token_proof": {"generated_token_ids": [1]}},
            {"output": "a", "token_proof": {"generated_token_ids": [2]}},
            "different",
        ),
        (
            {"output": "a", "token_proof": {"generated_token_ids": [1]}},
            {"output": "a", "token_proof": {"generated_token_ids": [1]}},
            "same",
        ),
    ],
)
def test_pair_classification_tracks_evidence(left, right, expected):
    assert (
        pair_facts({**left, "correct": False}, {**right, "correct": False})["comparison"]
        == expected
    )
