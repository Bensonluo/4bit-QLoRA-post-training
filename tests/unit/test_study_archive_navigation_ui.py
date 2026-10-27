"""The real R148 studies are browsable without manufacturing curves or prompts."""

import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
from streamlit.testing.v1 import AppTest

REAL_ROOT = Path(__file__).resolve().parents[2]
LIVE = REAL_ROOT / "docs/validation/sft-answer-supervision-live-2026-10-22.json"
ARCHIVE = REAL_ROOT / "docs/validation/sft-answer-supervision-evidence-2026-10-22.tar.gz"
CONTROLLED = "r148-controlled-v2"
PRODUCT = "r148-product-smoke"


def local_snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def source_snapshot():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (LIVE, ARCHIVE)}


def archived_json(reference):
    """Read the exact public member independently of the product data loader."""
    with tarfile.open(ARCHIVE) as archive:
        name = "r148-review-evidence/" + reference["archive_relative_path"]
        member = archive.getmember(name)
        assert member.isfile()
        raw = archive.extractfile(member).read()
    assert hashlib.sha256(raw).hexdigest() == reference["sanitized_sha256"]
    return json.loads(raw)


def expected_variant(study_id, variant):
    source = json.loads(LIVE.read_text())
    if study_id == CONTROLLED:
        controlled = source["controlled_v2"]["results"]
        if variant in {"base-alpaca", "base-native"}:
            protocol = (
                "alpaca_full_prompt_v1"
                if variant == "base-alpaca"
                else "native_chat_answer_only_v1"
            )
            return controlled["baseline"][protocol]["rows"], []
        arm = controlled["arms"][variant]
        history = archived_json(arm["optimizer_update_evidence"]["full_log"])
        return arm["evaluation"]["rows"], history
    smoke = source["product_smoke"]
    rows = smoke["results"]["evaluations"][variant]["rows"]
    state = archived_json(smoke["trainer_state"])
    return rows, [
        row for row in state["log_history"] if "loss" in row
    ] if variant == "trained" else []


def assert_clean(page):
    assert not page.exception, [item.message for item in page.exception]


def widget(page, study_id, suffix):
    return next(
        item
        for item in page.selectbox
        if study_id in str(item.key) and str(item.key).endswith(suffix)
    )


def curve_frames(block):
    """AppTest exposes Vega charts as protobufs; decode their actual Arrow data."""
    frames = []
    for chart in block.get("vega_lite_chart"):
        candidates = [chart.proto.data, *[item.data for item in chart.proto.datasets]]
        for data in candidates:
            if data.data:
                frames.append(convert_arrow_bytes_to_pandas_df(data.data))
    return frames


def training_tab(page):
    return next(tab for tab in page.tabs if tab.label == "训练曲线")


def assert_actual_curve(page, history):
    expected_steps = [row["step"] for row in history]
    expected_losses = [row["loss"] for row in history]
    frames = curve_frames(training_tab(page))
    matches = [frame for frame in frames if "loss" in frame.columns and len(frame) == len(history)]
    assert matches, "A chart must use the complete real optimizer loss sequence."
    assert any(
        frame["loss"].tolist() == pytest.approx(expected_losses)
        and (frame["step"].tolist() if "step" in frame else frame.index.tolist()) == expected_steps
        for frame in matches
    )


def boot(monkeypatch, tmp_path, *, display):
    from src.workbench.business_evaluation import LocalEvaluationRuntime
    from src.workbench.gguf_evaluation import GGUFLocalEvaluationRuntime
    from src.workbench.training_runs import TrainingRunService
    from ui import config, queries, runtime
    from ui.components import workspace

    def no_execution(*args, **kwargs):
        pytest.fail("Browsing an archive must not load a model or start training.")

    monkeypatch.setattr(LocalEvaluationRuntime, "__init__", no_execution)
    monkeypatch.setattr(GGUFLocalEvaluationRuntime, "__init__", no_execution)
    monkeypatch.setattr(TrainingRunService, "start", no_execution)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", REAL_ROOT)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    page = AppTest.from_file(str(REAL_ROOT / "ui/app.py"), default_timeout=45).run()
    assert_clean(page)
    page.button(key="showcase-history").click().run()
    assert_clean(page)
    # Button navigation is proven above; anchor AppTest's next widget reruns.
    page.switch_page("views/08_Showcase.py").run()
    assert_clean(page)
    return page


def choose_card(page, study_id):
    for identity in ("r149", CONTROLLED, PRODUCT):
        assert page.button(key=f"archive-select-{identity}")
    page.button(key=f"archive-select-{study_id}").click().run()
    assert_clean(page)
    assert page.selectbox(key="archive-selection").value == study_id


@pytest.mark.parametrize("study_id", [CONTROLLED, PRODUCT])
@pytest.mark.parametrize("display", [False, True])
def test_history_cards_show_real_training_and_question_outputs_without_writes(
    monkeypatch, tmp_path, study_id, display
):
    before = local_snapshot(tmp_path)
    sources = source_snapshot()
    page = boot(monkeypatch, tmp_path, display=display)
    choose_card(page, study_id)
    primary = widget(page, study_id, "-primary").value
    rows, history = expected_variant(study_id, primary)
    if study_id == PRODUCT:
        widget(page, study_id, "-primary").set_value("base").run()
        assert_clean(page)
        rows, history = expected_variant(study_id, "base")
        assert history == []
        assert not curve_frames(training_tab(page))
        widget(page, study_id, "-primary").set_value("trained").run()
        assert_clean(page)
        primary = "trained"
        rows, history = expected_variant(study_id, primary)
    assert_actual_curve(page, history)
    if study_id == PRODUCT:
        assert any(
            "训练精度 FP32" in item.value and "推理精度 BF16" in item.value
            for item in training_tab(page).caption
        )
    reference = rows[-1]
    widget(page, study_id, "-case").set_value((reference["case_id"], reference["form"])).run()
    assert_clean(page)
    codes = [element.value.strip() for element in page.code]
    assert reference["prompt"].strip() in codes
    assert reference["output"].strip() in codes
    assert reference["expected"].strip() in codes
    if study_id == PRODUCT:
        widget(page, study_id, "-primary").set_value("base").run()
        assert_clean(page)
        assert not curve_frames(training_tab(page)), (
            "A previous trained curve must not remain on base."
        )
    assert local_snapshot(tmp_path) == before
    assert source_snapshot() == sources


def test_native_selection_uses_its_own_prompt_and_json_case(monkeypatch, tmp_path):
    before = local_snapshot(tmp_path)
    sources = source_snapshot()
    page = boot(monkeypatch, tmp_path, display=True)
    choose_card(page, CONTROLLED)
    native = "native_chat_answer_only_v1"
    widget(page, CONTROLLED, "-primary").set_value(native).run()
    assert_clean(page)
    widget(page, CONTROLLED, "-comparison").set_value("base-alpaca").run()
    assert_clean(page)
    widget(page, CONTROLLED, "-form").set_value("json").run()
    assert_clean(page)
    rows, history = expected_variant(CONTROLLED, native)
    reference = next(row for row in reversed(rows) if row["form"] == "json")
    widget(page, CONTROLLED, "-case").set_value((reference["case_id"], reference["form"])).run()
    assert_clean(page)
    other_rows, _ = expected_variant(CONTROLLED, "base-alpaca")
    other = next(row for row in other_rows if row["case_id"] == reference["case_id"])
    assert reference["prompt"] != other["prompt"]
    codes = [element.value.strip() for element in page.code]
    assert reference["prompt"].strip() in codes
    assert other["prompt"].strip() in codes
    assert reference["output"].strip() in codes
    assert_actual_curve(page, history)
    assert local_snapshot(tmp_path) == before
    assert source_snapshot() == sources
