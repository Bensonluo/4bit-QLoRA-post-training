"""Archive cards retain actual per-row scores, order and read-only navigation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ROOT / "docs/validation/complete-output-handoff-live-2026-10-22.json",
    ROOT / "docs/validation/sft-answer-supervision-live-2026-10-22.json",
)
IDS = ("r149", "r148-controlled-v2", "r148-product-smoke")
LABELS = {
    "alpaca_full_prompt_v1": "全提示监督",
    "alpaca_answer_only_v1": "仅答案监督 · Alpaca",
    "native_chat_answer_only_v1": "仅答案监督 · 原生对话（实验）",
    "base": "基础模型",
    "trained": "产品微调模型",
}


def source_results():
    """Count the raw recorded rows, independently of the catalog summaries."""
    handoff, studies = [json.loads(path.read_text()) for path in SOURCES]

    def counts(label, rows, accepted):
        return (
            LABELS.get(label, label),
            sum(row[accepted] is True for row in rows),
            len(rows),
            sum(row["truncated"] is True for row in rows),
        )

    return {
        "r149": [
            counts(model["label"], model["rows"], "correct")
            for model in handoff["actual_report"]["models"]
        ],
        "r148-controlled-v2": [
            counts(label, arm["evaluation"]["rows"], "accepted")
            for label, arm in studies["controlled_v2"]["results"]["arms"].items()
        ],
        "r148-product-smoke": [
            counts(label, evaluation["rows"], "accepted")
            for label, evaluation in studies["product_smoke"]["results"]["evaluations"].items()
        ],
    }


def source_group_counts():
    handoff, studies = [json.loads(path.read_text()) for path in SOURCES]
    handoff_groups = {
        json.dumps(row["metadata"]["group"], sort_keys=True)
        for row in handoff["development_data"]["records"]
    }
    shared_groups = {row["group_id"] for row in studies["controlled_v2"]["data"]["development"]}
    return {
        "r149": len(handoff_groups),
        "r148-controlled-v2": len(shared_groups),
        "r148-product-smoke": len(shared_groups),
    }


def boot(ids=IDS):
    return AppTest.from_string(
        "from pathlib import Path\n"
        "from ui.showcase import available_history_catalog\n"
        "from ui.components.archive_cards import render_archive_cards\n"
        f"wanted = {ids!r}\n"
        f"catalog, errors = available_history_catalog(Path({str(ROOT)!r}))\n"
        "assert not errors\n"
        "render_archive_cards([item for item in catalog if item['id'] in wanted])\n"
    ).run(timeout=30)


@pytest.mark.parametrize("ids", [IDS, *((identity,) for identity in IDS)])
def test_cards_show_actual_results_and_retained_truncation_in_original_order(ids):
    page = boot(ids)
    assert not page.exception
    expected = source_results()
    rows = [row for identity in IDS if identity in ids for row in expected[identity]]
    progress = page.get("progress")
    assert [item.proto.text for item in progress] == [
        f"{label} · 答对 {correct} / {total} · 截断 {truncated} 题"
        for label, correct, total, truncated in rows
    ]
    assert [item.proto.value for item in progress] == [
        int(100 * correct / total) for _, correct, total, _ in rows
    ]
    assert [item.key for item in page.button] == [
        f"archive-select-{identity}" for identity in IDS if identity in ids
    ]
    assert any("合成开发题" in item.value for item in page.caption)
    groups = source_group_counts()
    assert [item.value for item in page.caption if "道合成开发题" in item.value] == [
        f"{expected[identity][0][2]} 道合成开发题 · {groups[identity]} 个业务分组 · 真实执行"
        for identity in IDS
        if identity in ids
    ]
    assert not page.success


@pytest.mark.parametrize("missing", [False, True])
def test_unverified_group_count_stays_unknown_even_with_other_counts(monkeypatch, missing):
    from ui import showcase

    item = deepcopy(showcase.history_catalog(ROOT)[1])
    if missing:
        item["counts"].pop("development_groups", None)
    else:
        item["counts"]["development_groups"] = None
    monkeypatch.setattr(showcase, "available_history_catalog", lambda root: ([item], []))
    page = boot()
    assert not page.exception
    assert any(
        value.value
        == f"{item['counts']['development_rows']} 道合成开发题 · 分组信息不可核验 · 真实执行"
        for value in page.caption
    )
    assert not any("个业务分组" in value.value for value in page.caption)
    assert len(page.get("progress")) == len(item["results"])
    page.button(key=f"archive-select-{item['id']}").click().run()
    assert not page.exception
    assert page.session_state["archive-selection"] == item["id"]


@pytest.mark.parametrize("identity", IDS)
def test_card_button_selects_its_record_without_registering_or_running_anything(identity):
    page = boot()
    page.button(key=f"archive-select-{identity}").click().run()
    assert not page.exception
    assert page.session_state["archive-selection"] == identity
    assert len(page.button) == len(IDS)
    assert not page.get("file_uploader")


def test_browsing_and_selecting_does_not_modify_catalog_evidence_or_local_storage(
    monkeypatch, tmp_path
):
    from ui import config, showcase

    catalog = showcase.history_catalog(ROOT)
    original = deepcopy(catalog)
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES}
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(showcase, "available_history_catalog", lambda root: (catalog, []))
    page = boot()
    for identity in reversed(IDS):
        page.button(key=f"archive-select-{identity}").click().run()
        assert not page.exception
    assert catalog == original
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in SOURCES} == hashes
    assert list(tmp_path.iterdir()) == []


def test_empty_catalog_has_no_executable_record_button():
    page = AppTest.from_string(
        "from ui.components.archive_cards import render_archive_cards\nrender_archive_cards([])\n"
    ).run(timeout=30)
    assert not page.exception
    assert not page.button
    assert not page.get("progress")
    assert any("没有可浏览的公开历史记录" in item.value for item in page.caption)
