"""A missing example should not hide the other real CSVs on the homepage."""

import hashlib
import shutil

import pytest

pytest.importorskip("streamlit")

from src.workbench.demo_task import demo_create_payload
from tests.unit.test_showcase_workspace_ui import REAL_ROOT, assert_clean, boot, tree_snapshot


def example_root(tmp_path):
    root = tmp_path / "public-assets"
    target = root / "data/custom/examples"
    target.mkdir(parents=True)
    for variant in (None, "extraction", "generation"):
        _, _, name, _ = demo_create_payload(REAL_ROOT, variant)
        shutil.copyfile(REAL_ROOT / "data/custom/examples" / name, target / name)
    return root


def example_path(root, variant):
    _, _, name, _ = demo_create_payload(REAL_ROOT, variant)
    return root / "data/custom/examples" / name


@pytest.mark.parametrize("display", [False, True])
def test_one_bad_example_keeps_other_previews_and_falls_back_from_lost_selection(
    monkeypatch, tmp_path, display
):
    from ui.components import workspace

    root = example_root(tmp_path)
    page = boot(monkeypatch, tmp_path / "workbench", display=display)
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", root)
    page.run()
    assert_clean(page)
    page.button(key="showcase-preview-generation").click().run()
    assert_clean(page)
    assert page.selectbox(key="showcase-example").value == "generation"

    example_path(root, "generation").write_bytes(b"not,the,expected,columns\nx,y,z,w\n")
    before = tree_snapshot(tmp_path)
    page.run()
    assert_clean(page)
    warning = "\n".join(item.value for item in page.warning)
    assert "客服答复生成" in warning and "其余示例仍可浏览" in warning
    assert str(tmp_path) not in warning
    assert page.selectbox(key="showcase-example").value == "classification"
    assert page.button(key="showcase-preview-classification")
    assert page.button(key="showcase-preview-extraction")
    assert not any(item.key == "showcase-preview-generation" for item in page.button)
    page.button(key="showcase-preview-extraction").click().run()
    assert_clean(page)
    assert page.selectbox(key="showcase-example").value == "extraction"
    assert tree_snapshot(tmp_path) == before


def test_all_missing_examples_are_honestly_reported_without_blocking_workspace(
    monkeypatch, tmp_path
):
    from ui.components import workspace

    root = example_root(tmp_path)
    for variant in (None, "extraction", "generation"):
        example_path(root, variant).unlink()
    page = boot(monkeypatch, tmp_path / "workbench")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", root)
    before = tree_snapshot(tmp_path)
    page.run()
    assert_clean(page)
    warning = "\n".join(item.value for item in page.warning)
    assert all(name in warning for name in ("售后工单分类", "工单编号抽取", "客服答复生成"))
    assert any("当前没有可用的示例资料" in item.value for item in page.caption)
    assert any(item.label == "创建业务任务" and not item.disabled for item in page.button)
    assert tree_snapshot(tmp_path) == before


def test_healthy_example_still_creates_its_exact_task_when_another_is_missing(
    monkeypatch, tmp_path
):
    from src.workbench.intake_service import IntakeService
    from ui.components import workspace

    root = example_root(tmp_path)
    example_path(root, "generation").unlink()
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
    goal, raw, name, _ = demo_create_payload(REAL_ROOT, "extraction")
    assert session.goal == goal
    assert session.source.name == name
    assert session.source.digest == hashlib.sha256(raw).hexdigest()
    assert session.analysis is None
