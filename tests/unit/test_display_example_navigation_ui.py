"""Display-only empty states lead to the examples they describe."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["05_Data_Wizard.py", "06_Chat.py"])
def test_display_empty_state_opens_real_examples_without_creating_task(
    monkeypatch, tmp_path, filename
):
    from ui import config, queries, runtime
    from ui.components import workspace

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "workbench")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", (tmp_path / "workbench/mlruns").as_uri())
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", ROOT)
    monkeypatch.setattr(queries, "fetch_runs", lambda *args, **kwargs: pd.DataFrame())
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    page.switch_page(f"views/{filename}").run()
    assert not page.exception, [item.message for item in page.exception]
    assert any(button.label == "查看目标与数据" for button in page.button)
    page.button(key="display-browse-examples").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert any(item.value == "从一份具体的数据开始" for item in page.subheader)
    assert page.button(key="showcase-preview-classification")
    assert not (tmp_path / "workbench").exists()
