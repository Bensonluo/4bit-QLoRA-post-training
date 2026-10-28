"""Evaluation sample labels reflect recorded status and preserve untrusted text."""

import pytest

from tests.unit.test_eval_result_resilience_ui import page_for, write_results


@pytest.mark.parametrize("filename", ["02_Evaluation.py", "03_Model_Comparison.py"])
def test_unknown_statuses_are_not_errors_or_silently_hidden(monkeypatch, tmp_path, filename):
    rows = [
        {
            "model": "甲",
            "per_sample": [
                {"query": "真实错误", "correct": False, "entity_type": "drug"},
                {"query": "缺失状态"},
                {"query": "空状态", "correct": None},
                {"query": "字符串状态", "correct": "false"},
                {"query": "数字状态", "correct": 0},
                {"query": "已通过", "correct": True},
            ],
        },
        {"model": "乙", "per_sample": [{"query": "乙错误", "correct": False}]},
    ]
    root = write_results(tmp_path, latest=rows)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    page = page_for(monkeypatch, tmp_path, filename)
    labels = [item.label for item in page.expander]
    assert any("1 个错误样本" in label and "甲" in label for label in labels)
    assert any("4 个状态未知样本" in label and "甲" in label for label in labels)
    text = "\n".join(item.value for item in page.text)
    assert all(
        name in text for name in ("真实错误", "缺失状态", "空状态", "字符串状态", "数字状态")
    )
    assert "已通过" not in text
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("filename", ["02_Evaluation.py", "03_Model_Comparison.py"])
def test_model_output_and_query_are_plain_text_not_active_html(monkeypatch, tmp_path, filename):
    hostile = '<img src=x onerror="alert(1)"> `</span><a href="https://example.com">click</a>'
    row = {"query": hostile, "predicted_name": hostile, "ground_truth": hostile, "correct": False}
    write_results(tmp_path, latest=[{"model": "甲", "per_sample": [row]}, {"model": "乙"}])
    page = page_for(monkeypatch, tmp_path, filename)
    assert any(hostile in item.value for item in page.text)
    assert not any(hostile in item.value and item.proto.allow_html for item in page.markdown)


def test_error_view_discloses_when_not_all_samples_are_shown(monkeypatch, tmp_path):
    errors = [{"query": str(i), "correct": False} for i in range(21)]
    write_results(tmp_path, latest=[{"model": "甲", "per_sample": errors}])
    page = page_for(monkeypatch, tmp_path, "02_Evaluation.py")
    assert any("仅展示前 20" in item.value for item in page.caption)
