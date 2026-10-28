"""Missing optional handoff metadata must not hide verified delivery evidence."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui.showcase import R149_SOURCE, load_showcase

ROOT = Path(__file__).resolve().parents[2]


def render(root):
    from ui.components.delivery_archive import render_delivery_archive

    render_delivery_archive(root)


@pytest.mark.parametrize("damage", ["missing", "invalid", "malformed_date", "date_only"])
def test_optional_metadata_is_unknown_without_hiding_verified_results(tmp_path, damage):
    original = ROOT / "docs/validation" / R149_SOURCE
    data = json.loads(original.read_text())
    dataset = data["actual_report"]["dataset"]
    training = data["training"]
    if damage == "missing":
        dataset.pop("name")
        dataset.pop("version")
        training.pop("finished_at")
    else:
        dataset["name"] = []
        dataset["version"] = 123
        training["finished_at"] = {
            "invalid": {"unknown": True},
            "malformed_date": "not-a-date",
            "date_only": "2026-10-22",
        }[damage]
    target = tmp_path / "docs/validation" / R149_SOURCE
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(data, ensure_ascii=False))
    before = target.read_bytes()

    # Core report, training and weight evidence remains independently verifiable.
    verified = load_showcase(tmp_path)
    page = AppTest.from_function(render, args=(tmp_path,), default_timeout=30).run()
    assert not page.exception, [item.message for item in page.exception]
    assert [metric.value for metric in page.metric if metric.label == "开发集严格匹配"] == [
        "8 / 8",
        "8 / 8",
    ]
    captions = [item.value for item in page.caption]
    assert "数据版本未记录或不可核验。" in captions
    assert "数据集名称未记录或不可核验。" in captions
    assert "训练结束时间未记录或不可核验。" in captions
    assert verified["training"]["run_id"] in [item.value for item in page.code]
    lineage = json.loads(page.json[0].value)
    assert lineage["dataset"]["name"] is None
    assert lineage["dataset"]["version"] is None
    assert lineage["training"]["finished_at"] is None
    assert target.read_bytes() == before


def test_recorded_delivery_version_is_shown_without_unknown_notice():
    expected = load_showcase(ROOT)
    page = AppTest.from_function(render, args=(ROOT,), default_timeout=30).run()
    assert not page.exception
    assert expected["report"]["dataset"]["version"] in [item.value for item in page.code]
    assert not any("未记录或不可核验" in item.value for item in page.caption)


@pytest.mark.parametrize(
    "value",
    ["2026-10-22T15:40:49.330214+00:00", "2026-10-22T15:40:49Z", "2026-10-22 15:40:49"],
)
def test_valid_timestamp_retains_original_precision_and_zone(value):
    from ui.components.delivery_archive import _recorded_timestamp

    assert _recorded_timestamp(value) == value
