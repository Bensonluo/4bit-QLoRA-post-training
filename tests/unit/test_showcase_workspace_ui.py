"""Real example previews and public history stay separate from local task writes."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from src.workbench.demo_task import demo_create_payload
from src.workbench.intake_service import IntakeService

REAL_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (("classification", None), ("extraction", "extraction"), ("generation", "generation"))


def tree_snapshot(root):
    """Include directories too: browsing must not initialize empty local stores."""
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def asset_snapshot():
    paths = list((REAL_ROOT / "data/custom/examples").glob("*_sample.csv"))
    paths += list((REAL_ROOT / "docs/validation").glob("complete-output-handoff-*2026-10-22.*"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def boot(monkeypatch, tmp_path, *, display=False):
    from ui import config, queries, runtime
    from ui.components import workspace

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", REAL_ROOT)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    return AppTest.from_file(str(REAL_ROOT / "ui/app.py"), default_timeout=45).run()


def assert_clean(page):
    assert not page.exception, [item.message for item in page.exception]


def expected_example(variant):
    payload = demo_create_payload(REAL_ROOT, variant)
    assert payload is not None
    goal, raw, name, _ = payload
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    return goal, raw, name, rows


@pytest.mark.parametrize("example_id,variant", EXAMPLES)
def test_example_preview_shows_real_csv_and_selection_changes(
    monkeypatch, tmp_path, example_id, variant
):
    before = tree_snapshot(tmp_path)
    page = boot(monkeypatch, tmp_path)
    assert_clean(page)
    for identity, _ in EXAMPLES:
        assert page.button(key=f"showcase-preview-{identity}").label == "预览数据"
        assert page.button(key=f"showcase-create-{identity}").label == "用这份数据开始"
    page.button(key=f"showcase-preview-{example_id}").click().run()
    assert_clean(page)
    selector = next(item for item in page.selectbox if item.label == "示例数据")
    assert selector.value == example_id
    _, _, _, rows = expected_example(variant)
    first = rows[0]
    matches = [
        element.value for element in page.dataframe if set(first).issubset(element.value.columns)
    ]
    assert matches, "The selected example must display its actual CSV columns."
    assert any(
        all(str(frame.iloc[0][column]) == value for column, value in first.items())
        for frame in matches
    )
    assert tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("example_id,variant", EXAMPLES)
def test_starting_an_example_creates_only_its_original_unconfirmed_task(
    monkeypatch, tmp_path, example_id, variant
):
    page = boot(monkeypatch, tmp_path)
    assert_clean(page)
    page.button(key=f"showcase-create-{example_id}").click().run()
    assert_clean(page)
    identity = page.session_state["intake_id"]
    service = IntakeService(tmp_path / "outputs/workbench/intake")
    sessions = service.list_sessions()
    assert [session.session_id for session in sessions] == [identity]
    session = service.load(identity)
    goal, raw, name, rows = expected_example(variant)
    assert session.goal == goal
    assert session.source.name == name
    assert session.source.digest == hashlib.sha256(raw).hexdigest()
    assert [row.values for row in session.source.rows] == rows
    assert session.analysis is None
    assert session.preview is None
    assert session.confirmed_revision is None
    assert session.full_data is None
    assert session.dataset is None
    assert session.training_preflight is None
    assert session.label_verification is None


@pytest.mark.parametrize("display", [False, True])
def test_public_history_is_reachable_without_writing_local_or_original_evidence(
    monkeypatch, tmp_path, display
):
    before = tree_snapshot(tmp_path)
    assets = asset_snapshot()
    page = boot(monkeypatch, tmp_path, display=display)
    assert_clean(page)
    page.button(key="showcase-history").click().run()
    assert_clean(page)
    assert page.title
    assert any(item.label == "历史记录" for item in page.selectbox)
    assert {"同题对照", "训练过程", "交付与成本", "来源与下载"}.issubset(
        {item.label for item in page.tabs}
    )
    # AppTest retains its original page hash after a button-driven switch.
    # The assertions above prove that button navigation; anchor subsequent reruns.
    page.switch_page("pages/08_Showcase.py").run()
    assert_clean(page)
    report = json.loads(
        (REAL_ROOT / "docs/validation/complete-output-handoff-live-2026-10-22.json").read_text()
    )["actual_report"]
    reference = report["models"][0]["rows"][-1]
    page.selectbox(key="archive-question").select(reference["index"]).run()
    assert_clean(page)
    # Streamlit code blocks normalize surrounding whitespace for display.
    rendered = [item.value.strip() for item in page.code]
    assert reference["prompt"].strip() in rendered
    assert reference["output"].strip() in rendered
    assert reference["expected"] in rendered
    assert tree_snapshot(tmp_path) == before
    assert asset_snapshot() == assets


def test_display_mode_can_preview_every_example_but_cannot_create_a_task(monkeypatch, tmp_path):
    before = tree_snapshot(tmp_path)
    page = boot(monkeypatch, tmp_path, display=True)
    assert_clean(page)
    for example_id, _ in EXAMPLES:
        page.button(key=f"showcase-preview-{example_id}").click().run()
        assert_clean(page)
        assert next(x for x in page.selectbox if x.label == "示例数据").value == example_id
        create = page.button(key=f"showcase-create-{example_id}")
        assert create.disabled
        # Exercise the Python boundary too; AppTest can signal disabled widgets.
        create.click().run()
        assert_clean(page)
    assert tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("record_id", ["r148-controlled-v2", "r148-product-smoke"])
def test_other_historical_records_can_be_selected(monkeypatch, tmp_path, record_id):
    from ui.showcase import history_catalog

    expected = next(item for item in history_catalog(REAL_ROOT) if item["id"] == record_id)
    page = boot(monkeypatch, tmp_path)
    page.switch_page("pages/08_Showcase.py").run()
    page.selectbox(key="archive-selection").set_value(record_id).run()
    assert_clean(page)
    table = next(item.value for item in page.dataframe if "答对 / 固定题数" in item.value.columns)
    assert table["答对 / 固定题数"].tolist() == [
        f"{item['correct']} / {item['total']}" for item in expected["results"]
    ]
    assert tree_snapshot(tmp_path) == {}


def test_missing_public_assets_keep_local_task_entry_available(monkeypatch, tmp_path):
    from ui.components import workspace

    page = boot(monkeypatch, tmp_path)
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", tmp_path)
    page.run()
    assert_clean(page)
    assert any("示例数据暂不可读" in item.value for item in page.warning)
    assert any("公开历史记录暂不可读" in item.value for item in page.warning)
    assert any(item.label == "创建业务任务" and not item.disabled for item in page.button)
    assert tree_snapshot(tmp_path) == {}
