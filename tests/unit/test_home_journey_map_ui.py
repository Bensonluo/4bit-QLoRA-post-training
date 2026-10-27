"""Task-first navigation retains every existing workbench page."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "target",
    [
        "00_Training_Lab.py",
        "01_Experiments.py",
        "02_Evaluation.py",
        "03_Model_Comparison.py",
        "04_Registry.py",
        "05_Data_Wizard.py",
        "06_Chat.py",
        "07_Data_Intake.py",
    ],
)
def test_existing_pages_remain_reachable_from_the_shared_entry(monkeypatch, tmp_path, target):
    from ui import config, queries

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    assert not page.exception
    page.switch_page("views/" + target).run()
    assert not page.exception, [e.message for e in page.exception]
    assert page.title
