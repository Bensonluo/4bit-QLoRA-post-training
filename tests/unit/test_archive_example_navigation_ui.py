"""The archive's examples link leads to the existing, non-creating preview."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("display", [False, True])
@pytest.mark.parametrize("archives_unavailable", [False, True])
def test_archive_opens_example_previews_without_creating_a_task(
    monkeypatch, tmp_path, display, archives_unavailable
):
    from src.workbench.intake_service import IntakeService
    from ui import config, queries, runtime, showcase
    from ui.components import workspace

    def no_create(*args, **kwargs):
        pytest.fail("Browsing examples must not create a task")

    monkeypatch.setattr(IntakeService, "create", no_create)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "local")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'local/mlruns'}")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", ROOT)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])
    if archives_unavailable:
        monkeypatch.setattr(showcase, "available_history_catalog", lambda root: ([], []))

    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    assert not page.exception
    page.switch_page("views/08_Showcase.py").run()
    assert not page.exception
    page.button(key="archive-open-examples").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert any(item.value == "业务模型工作台" for item in page.title)
    assert any(item.value == "从一份具体的数据开始" for item in page.subheader)
    assert page.button(key="showcase-preview-classification")
    assert not (tmp_path / "local").exists()
