"""Built-in examples must faithfully display usable CSV input and answers."""

import hashlib

import pytest

from src.workbench.intake_service import IntakeService
from tests.unit.test_example_resilience_ui import example_path, example_root
from tests.unit.test_showcase_workspace_ui import assert_clean, boot, tree_snapshot
from ui.showcase import example_tasks, example_tasks_tolerant


@pytest.mark.parametrize(
    "invalid",
    [
        "客户描述,类别\n杯子破损,\n",
        "客户描述,类别\n  ,质量\n",
        "客户描述,类别\n杯子破损,质量\n物流延误, \n",
        "客户描述,类别,类别\n杯子破损,质量,物流\n",
        "客户描述,类别, 类别 \n杯子破损,质量,物流\n",
        "客户描述,类别, \n杯子破损,质量,x\n",
        "客户描述,类别\n杯子破损\n",
        "客户描述,类别\n杯子破损,质量,意外多列\n",
        '客户描述,类别\n"杯子破损,质量\n',
    ],
    ids=[
        "no-answer",
        "no-input",
        "later-answer-empty",
        "duplicate",
        "duplicate-spaces",
        "blank-header",
        "short-row",
        "extra-row",
        "unclosed-quote",
    ],
)
def test_bad_example_csv_isolated_without_changing_other_sources(tmp_path, invalid):
    root = example_root(tmp_path)
    broken = example_path(root, None)
    raw = invalid.encode("utf-8")
    broken.write_bytes(raw)
    tasks, unavailable = example_tasks_tolerant(root)
    assert [item["id"] for item in tasks] == ["extraction", "generation"]
    assert unavailable == [{"id": "classification", "title": "售后工单分类"}]
    assert broken.read_bytes() == raw
    with pytest.raises(ValueError):
        example_tasks(root)


def test_good_quoted_utf8_csv_retains_original_bytes_and_exact_values(tmp_path):
    root = example_root(tmp_path)
    raw = b"\xef\xbb\xbf" + '客户描述,类别\r\n"杯子,破损",质量\r\n'.encode()
    example_path(root, None).write_bytes(raw)
    tasks, unavailable = example_tasks_tolerant(root)
    assert not unavailable
    sample = next(item for item in tasks if item["id"] == "classification")
    assert sample["bytes"] == raw
    assert sample["rows"] == [{"客户描述": "杯子,破损", "类别": "质量"}]


@pytest.mark.parametrize("display", [False, True])
@pytest.mark.parametrize(
    "invalid",
    ["客户描述,类别\n杯子破损,\n", "客户描述,类别,类别\n杯子破损,质量,物流\n"],
    ids=["missing-answer", "duplicate-answer-column"],
)
def test_ui_hides_unusable_example_and_keeps_healthy_preview(
    monkeypatch, tmp_path, display, invalid
):
    from ui.components import workspace

    root = example_root(tmp_path)
    example_path(root, None).write_text(invalid, encoding="utf-8")
    page = boot(monkeypatch, tmp_path / "workbench", display=display)
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", root)
    before = tree_snapshot(tmp_path)
    page.run()
    assert_clean(page)
    assert any(
        "售后工单分类" in item.value and "其余示例仍可浏览" in item.value for item in page.warning
    )
    assert not any(item.key == "showcase-preview-classification" for item in page.button)
    assert not any(item.key == "showcase-create-classification" for item in page.button)
    page.button(key="showcase-preview-extraction").click().run()
    assert_clean(page)
    assert page.selectbox(key="showcase-example").value == "extraction"
    assert tree_snapshot(tmp_path) == before


def test_healthy_example_can_still_create_exact_task_when_peer_is_invalid(monkeypatch, tmp_path):
    from ui.components import workspace

    root = example_root(tmp_path)
    example_path(root, None).write_text("客户描述,类别\n杯子破损,\n", encoding="utf-8")
    raw = example_path(root, "extraction").read_bytes()
    workbench = tmp_path / "workbench"
    page = boot(monkeypatch, workbench)
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", root)
    page.run()
    assert_clean(page)
    page.button(key="showcase-create-extraction").click().run()
    assert_clean(page)
    session = IntakeService(workbench / "outputs/workbench/intake").load(
        page.session_state["intake_id"]
    )
    assert session.source.digest == hashlib.sha256(raw).hexdigest()
    assert session.analysis is None
