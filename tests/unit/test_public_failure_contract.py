"""Public reports reject incomplete failure rows before any UI consumes them."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.showcase import R149_SOURCE, load_showcase

ROOT = Path(__file__).resolve().parents[2]


def failed_report(tmp_path):
    source = ROOT / "docs/validation" / R149_SOURCE
    report = json.loads(source.read_text())
    model = report["actual_report"]["models"][0]
    row = model["rows"][0]
    row.update(
        status="failed",
        output=None,
        token_proof=None,
        truncated=False,
        generated_tokens=None,
        correct=False,
    )
    model["metrics"].update(generated=7, generation_coverage=7 / 8)
    directory = tmp_path / "docs/validation"
    directory.mkdir(parents=True)
    return directory / R149_SOURCE, report, row


def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False))


def detail(root):
    from ui.components.showcase import render_history

    render_history(root)


def test_failed_report_keeps_unknown_token_count_and_fixed_denominator(tmp_path):
    path, report, row = failed_report(tmp_path)
    write_report(path, report)
    before = path.read_bytes()
    loaded = load_showcase(tmp_path)
    actual = loaded["report"]["models"][0]
    assert actual["rows"][0]["generated_tokens"] is None
    assert actual["rows"][0]["truncated"] is False
    assert actual["metrics"]["total"] == 8
    assert path.read_bytes() == before


@pytest.mark.parametrize(("token_count", "expected"), [(None, "token 数未记录"), (0, "0 token")])
def test_failed_report_displays_unknown_or_zero_token_count(tmp_path, token_count, expected):
    path, report, row = failed_report(tmp_path)
    row["generated_tokens"] = token_count
    write_report(path, report)
    page = AppTest.from_function(detail, args=(tmp_path,), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("未生成" in item.value and expected in item.value for item in page.caption)


@pytest.mark.parametrize(
    "mutation",
    ["missing_count", "missing_truncation", "false_truncation", "invalid_count", "boolean_count"],
)
def test_invalid_failure_fields_are_reported_before_rendering(tmp_path, mutation):
    path, report, row = failed_report(tmp_path)
    if mutation == "missing_count":
        del row["generated_tokens"]
    elif mutation == "missing_truncation":
        del row["truncated"]
    elif mutation == "false_truncation":
        row["truncated"] = True
    elif mutation == "invalid_count":
        row["generated_tokens"] = -1
    else:
        row["generated_tokens"] = True
    write_report(path, report)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="失败记录"):
        load_showcase(tmp_path)
    assert path.read_bytes() == before
