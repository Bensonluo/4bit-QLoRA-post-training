"""A damaged archive must not hide other verified historical records."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ui.showcase import R148_SOURCE, R149_SOURCE

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "sft-answer-supervision-evidence-2026-10-22.tar.gz"
CONTROLLED = "r148-controlled-v2"
PRODUCT = "r148-product-smoke"


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "assets"
    directory = root / "docs/validation"
    directory.mkdir(parents=True)
    for name in (R148_SOURCE, R149_SOURCE, ARCHIVE):
        shutil.copyfile(ROOT / "docs/validation" / name, directory / name)
    return root


def boot(monkeypatch, tmp_path, assets):
    from ui import config, queries, runtime
    from ui.components import workspace

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path / "local")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{tmp_path / 'local/mlruns'}")
    monkeypatch.setattr(workspace, "SHOWCASE_ROOT", assets)
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    assert not page.exception
    page.switch_page("views/08_Showcase.py").run()
    assert not page.exception, [entry.message for entry in page.exception]
    return page


def card_keys(page):
    return {button.key for button in page.button if str(button.key).startswith("archive-select-")}


def warning_text(page):
    return "\n".join(item.value for item in page.warning)


def render_handoff(root):
    from ui.components.showcase import render_history

    render_history(root)


def test_missing_r149_keeps_both_r148_studies_and_allows_navigation(monkeypatch, tmp_path, assets):
    (assets / "docs/validation" / R149_SOURCE).unlink()
    page = boot(monkeypatch, tmp_path, assets)
    assert card_keys(page) == {f"archive-select-{CONTROLLED}", f"archive-select-{PRODUCT}"}
    assert "完整分类任务" in warning_text(page)
    assert page.selectbox(key="archive-selection").value == CONTROLLED
    page.button(key=f"archive-select-{PRODUCT}").click().run()
    assert not page.exception
    assert page.selectbox(key="archive-selection").value == PRODUCT
    assert any(tab.label == "训练曲线" for tab in page.tabs)
    assert not (tmp_path / "local").exists()


def test_missing_r148_keeps_r149_and_reports_both_unavailable(monkeypatch, tmp_path, assets):
    (assets / "docs/validation" / R148_SOURCE).unlink()
    page = boot(monkeypatch, tmp_path, assets)
    assert card_keys(page) == {"archive-select-r149"}
    assert "监督协议受控对照" in warning_text(page)
    assert "产品链路验证" in warning_text(page)
    assert page.selectbox(key="archive-selection").value == "r149"
    assert any(tab.label == "同题对照" for tab in page.tabs)
    assert not (tmp_path / "local").exists()


def test_corrupt_selected_study_falls_back_to_a_healthy_record(monkeypatch, tmp_path, assets):
    page = boot(monkeypatch, tmp_path, assets)
    page.button(key=f"archive-select-{CONTROLLED}").click().run()
    assert page.selectbox(key="archive-selection").value == CONTROLLED
    path = assets / "docs/validation" / R148_SOURCE
    data = json.loads(path.read_text())
    first_arm = next(iter(data["controlled_v2"]["results"]["arms"].values()))
    first_arm["evaluation"]["summary"]["all"]["accepted"] = 999
    path.write_text(json.dumps(data, ensure_ascii=False))
    page.run()
    assert not page.exception
    assert card_keys(page) == {"archive-select-r149", f"archive-select-{PRODUCT}"}
    assert "监督协议受控对照" in warning_text(page)
    assert page.selectbox(key="archive-selection").value == "r149"
    page.button(key=f"archive-select-{PRODUCT}").click().run()
    assert not page.exception
    assert page.selectbox(key="archive-selection").value == PRODUCT
    assert not (tmp_path / "local").exists()


def test_missing_shared_input_retains_product_scores_but_explains_detail_gap(
    monkeypatch, tmp_path, assets
):
    path = assets / "docs/validation" / R148_SOURCE
    data = json.loads(path.read_text())
    del data["controlled_v2"]
    path.write_text(json.dumps(data, ensure_ascii=False))
    page = boot(monkeypatch, tmp_path, assets)
    assert card_keys(page) == {"archive-select-r149", f"archive-select-{PRODUCT}"}
    assert any("16 道合成开发题 · 分组信息不可核验" in item.value for item in page.caption)
    page.button(key=f"archive-select-{PRODUCT}").click().run()
    assert not page.exception
    assert page.selectbox(key="archive-selection").value == PRODUCT
    assert "这份历史记录暂不可读" in warning_text(page)
    assert not (tmp_path / "local").exists()


def test_unreadable_study_detail_does_not_expose_server_path(monkeypatch, tmp_path, assets):
    from ui import study_archive

    page = boot(monkeypatch, tmp_path, assets)

    def unavailable(*args):
        raise ValueError("/Users/private-owner/secret-client/archive.tar.gz")

    monkeypatch.setattr(study_archive, "load_study", unavailable)
    page.button(key=f"archive-select-{CONTROLLED}").click().run()
    assert not page.exception
    assert "这份历史记录暂不可读" in warning_text(page)
    assert "private-owner" not in warning_text(page)
    assert "secret-client" not in warning_text(page)
    assert card_keys(page) == {
        "archive-select-r149",
        f"archive-select-{CONTROLLED}",
        f"archive-select-{PRODUCT}",
    }
    assert not (tmp_path / "local").exists()


def test_unreadable_handoff_detail_does_not_expose_server_path(monkeypatch, assets):
    from ui import showcase

    def unavailable(*args):
        raise OSError("/Users/private-owner/secret-client/report.json")

    monkeypatch.setattr(showcase, "load_showcase", unavailable)
    page = AppTest.from_function(render_handoff, args=(assets,)).run()
    assert not page.exception
    assert "公开历史记录暂不可读" in warning_text(page)
    assert "private-owner" not in warning_text(page)
    assert "secret-client" not in warning_text(page)


def test_all_sources_missing_gives_a_task_navigation_exit(monkeypatch, tmp_path, assets):
    for name in (R149_SOURCE, R148_SOURCE):
        (assets / "docs/validation" / name).unlink()
    page = boot(monkeypatch, tmp_path, assets)
    assert not card_keys(page)
    assert not any(item.label == "历史记录" for item in page.selectbox)
    assert "完整分类任务" in warning_text(page)
    assert "监督协议受控对照" in warning_text(page)
    assert "产品链路验证" in warning_text(page)
    assert not (tmp_path / "local").exists()
    page.button(key="archive-open-tasks").click().run()
    assert not page.exception
    assert any("先说业务目标，再看数据" in title.value for title in page.title)
