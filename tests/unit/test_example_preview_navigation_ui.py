"""Example cards and the data selector describe the same preview."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
IDS = ("classification", "extraction", "generation")


@pytest.mark.parametrize("display", [False, True])
def test_preview_selection_and_link_follow_cards_selector_and_navigation(
    monkeypatch, tmp_path, display
):
    from src.workbench.intake_service import IntakeService
    from ui import config, queries, runtime
    from ui.components import workspace

    def no_create(*args, **kwargs):
        pytest.fail("Previewing an example must not create a task")

    monkeypatch.setattr(IntakeService, "create", no_create)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "local")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'local/mlruns'}")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", ROOT)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])

    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()

    def assert_selected(identity):
        assert not page.exception, [item.message for item in page.exception]
        assert page.selectbox(key="showcase-example").value == identity
        assert next(x for x in page.expander if x.label == "原始示例数据与下载").proto.expanded
        for item in IDS:
            label = "当前预览" if item == identity else "预览数据"
            assert page.button(key=f"showcase-preview-{item}").label == label
        assert sum("[前往已选数据 ↓](#example-data)" in x.value for x in page.markdown) == 1
        assert any(
            x.value == "示例数据预览" and x.proto.anchor == "example-data" for x in page.subheader
        )

    page.button(key="showcase-preview-extraction").click().run()
    assert_selected("extraction")
    page.button(key="showcase-preview-classification").click().run()
    assert_selected("classification")
    page.selectbox(key="showcase-example").select("generation").run()
    assert_selected("generation")
    page.switch_page("views/08_Showcase.py").run()
    page.button(key="archive-open-examples").click().run()
    assert_selected("generation")
    assert not (tmp_path / "local").exists()
