"""Public delivery evidence is read-only; local registry identity remains exact."""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

REAL_ROOT = Path(__file__).resolve().parents[2]
PUBLIC = "公开交付档案"
LOCAL = "本地注册版本"


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def asset_identity():
    source = REAL_ROOT / "docs/validation/complete-output-handoff-live-2026-10-22.json"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def assert_clean(page):
    assert not page.exception, [entry.message for entry in page.exception]


def boot(monkeypatch, tmp_path, *, display=True, fetch=None):
    from src import tracking
    from ui import config, queries, runtime
    from ui.components import workspace

    calls = []

    def versions(*args, **kwargs):
        calls.append("versions")
        if fetch is None:
            pytest.fail("Public delivery must not query the local registry.")
        return fetch()

    def forbidden_tracker(*args, **kwargs):
        calls.append("tracker")
        pytest.fail("Browsing delivery or local registry must not initialize a writer.")

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'mlruns'}")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", REAL_ROOT)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])
    monkeypatch.setattr(queries, "fetch_model_versions", versions)
    monkeypatch.setattr(tracking, "get_tracker", forbidden_tracker)
    page = AppTest.from_file(str(REAL_ROOT / "ui/app.py"), default_timeout=45).run()
    assert_clean(page)
    page.switch_page("views/04_Registry.py").run()
    assert_clean(page)
    return page, calls


def test_display_defaults_to_real_public_artifacts_without_mlflow_or_writes(monkeypatch, tmp_path):
    from ui.showcase import load_showcase

    expected = load_showcase(REAL_ROOT)
    before = snapshot(tmp_path)
    source_before = asset_identity()
    # Public delivery must work even when MLflow itself is unavailable.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    page, calls = boot(monkeypatch, tmp_path)
    assert page.segmented_control(key="registry-source").value == PUBLIC
    text = "\n".join(
        str(entry.value)
        for collection in (page.text, page.code, page.caption, page.markdown, page.json)
        for entry in collection
    )
    for artifact in expected["artifacts"].values():
        for filename, details in artifact["weight_files"].items():
            assert filename in text
            assert f"{details['size']:,}" in text
            assert details["sha256"] in text
    assert expected["source"]["name"] in text
    assert expected["source"]["sha256"] in text
    assert calls == []
    assert snapshot(tmp_path) == before
    assert asset_identity() == source_before


def test_delivery_opens_complete_comparison_and_replaces_a_stale_study_selection(
    monkeypatch, tmp_path
):
    before = snapshot(tmp_path)
    page, calls = boot(monkeypatch, tmp_path)
    page.session_state["archive-selection"] = "r148-product-smoke"
    page.button(key="registry-delivery-open-comparison").click().run()
    assert_clean(page)
    assert page.selectbox(key="archive-selection").value == "r149"
    assert {"同题对照", "训练过程", "交付与成本", "来源与下载"}.issubset(
        {tab.label for tab in page.tabs}
    )
    assert calls == []
    assert snapshot(tmp_path) == before


def version_frame(aliases):
    return pd.DataFrame(
        [
            {
                "name": "local-model",
                "version": version,
                "current_stage": "None",
                "aliases": alias,
                "status": "READY",
                "created": "2026-10-22",
                "run_id": f"local-run-{version}",
            }
            for version, alias in aliases
        ]
    )


@pytest.mark.parametrize(
    "aliases,champion,challenger",
    [
        (
            [
                (4, "pre-champion"),
                (3, "pre-challenger"),
                (2, "deployed, champion"),
                (1, "challenger"),
            ],
            "v2",
            "v1",
        ),
        ([(4, "pre-champion"), (3, "pre-challenger")], "—", "—"),
    ],
)
def test_local_registry_uses_complete_aliases_and_preserves_registered_versions(
    monkeypatch, tmp_path, aliases, champion, challenger
):
    page, calls = boot(monkeypatch, tmp_path, display=False, fetch=lambda: version_frame(aliases))
    assert page.segmented_control(key="registry-source").value == LOCAL
    assert next(item for item in page.metric if item.label.endswith("Champion")).value == champion
    assert (
        next(item for item in page.metric if item.label.endswith("Challenger")).value == challenger
    )
    assert (
        next(item for item in page.selectbox if item.label == "已注册模型").value == "local-model"
    )
    assert set(page.selectbox(key="alias_version").options) == {str(v) for v, _ in aliases}
    assert any(item.label == "🏷 设置" and not item.disabled for item in page.button)
    assert calls == ["versions"]


@pytest.mark.parametrize("broken", [False, True])
def test_switch_to_local_preserves_empty_and_read_failure_states(monkeypatch, tmp_path, broken):
    def fetch():
        if broken:
            raise RuntimeError("test registry unreadable")
        return pd.DataFrame()

    # A display-mode query is permitted only for a pre-existing file store.
    (tmp_path / "mlruns/.trash").mkdir(parents=True)
    (tmp_path / "mlruns/models").mkdir()
    before = snapshot(tmp_path)
    page, calls = boot(monkeypatch, tmp_path, fetch=fetch)
    assert calls == []
    page.segmented_control(key="registry-source").set_value(LOCAL).run()
    assert_clean(page)
    if broken:
        assert any("注册记录暂不可读" in entry.value for entry in page.warning)
        assert not any("test registry unreadable" in entry.value for entry in page.warning)
        assert not any("暂无已注册的模型" in entry.value for entry in page.info)
    else:
        messages = [entry.value for entry in (*page.info, *page.caption, *page.markdown)]
        assert any("暂无已注册的模型" in text or "尚未登记模型" in text for text in messages)
        assert not page.error
    assert calls == ["versions"]
    assert snapshot(tmp_path) == before
