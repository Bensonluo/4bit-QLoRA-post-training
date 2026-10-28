"""Missing precision must never change whether a historical arm was trained."""

import json
import shutil
from pathlib import Path

import pytest

from tests.unit.test_archive_resilience_ui import (
    ARCHIVE,
    CONTROLLED,
    PRODUCT,
    ROOT,
    boot,
    card_keys,
    warning_text,
)
from tests.unit.test_study_archive_navigation_ui import curve_frames, training_tab
from ui.showcase import R148_SOURCE, R149_SOURCE
from ui.study_archive import load_study


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "assets"
    directory = root / "docs/validation"
    directory.mkdir(parents=True)
    for name in (R148_SOURCE, R149_SOURCE, ARCHIVE):
        shutil.copyfile(ROOT / "docs/validation" / name, directory / name)
    return root


def change(assets: Path, mutation):
    path = assets / "docs/validation" / R148_SOURCE
    data = json.loads(path.read_text())
    mutation(data)
    path.write_text(json.dumps(data, ensure_ascii=False))


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


@pytest.mark.parametrize("missing", ["null", "absent"])
def test_missing_controlled_precision_does_not_relabel_trained_arms(
    monkeypatch, tmp_path, assets, missing
):
    def mutate(data):
        settings = data["controlled_v2"]["settings"]
        if missing == "null":
            settings["dtype"] = None
        else:
            del settings["dtype"]

    change(assets, mutate)
    before = snapshot(assets)
    study = load_study(assets, CONTROLLED)
    original = load_study(ROOT, CONTROLLED)
    assert [v["role"] for v in study["variants"]] == [
        "baseline",
        "baseline",
        "trained",
        "trained",
        "trained",
    ]
    for actual, recorded in zip(study["variants"], original["variants"], strict=True):
        assert actual["training_precision"] is None and actual["evaluation_precision"] is None
        assert actual["training_history"] == recorded["training_history"]
        assert actual["rows"] == recorded["rows"]
        assert actual["baseline_id"] == recorded["baseline_id"]
    page = boot(monkeypatch, tmp_path, assets)
    page.button(key=f"archive-select-{CONTROLLED}").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("训练精度 未记录 · 推理精度 未记录" in item.value for item in page.caption)
    assert not any("这个基座在本次实验中未训练" in item.value for item in page.info)
    assert any(
        "loss" in frame.columns and len(frame) == 64 for frame in curve_frames(training_tab(page))
    )
    assert f"archive-select-{PRODUCT}" in card_keys(page)
    assert not (tmp_path / "local").exists()
    assert snapshot(assets) == before


@pytest.mark.parametrize("field", ["training", "base_eval", "trained_eval"])
def test_missing_product_precision_is_unknown_without_losing_role_or_curve(
    monkeypatch, tmp_path, assets, field
):
    def mutate(data):
        smoke = data["product_smoke"]
        if field == "training":
            del smoke["settings"]["actual_training_dtype"]
        else:
            model = "base" if field == "base_eval" else "trained"
            del smoke["results"]["evaluations"][model]["actual_dtype"]

    change(assets, mutate)
    before = snapshot(assets)
    study = load_study(assets, PRODUCT)
    original = load_study(ROOT, PRODUCT)
    assert [item["role"] for item in study["variants"]] == ["baseline", "trained"]
    assert study["variants"][1]["training_history"] == original["variants"][1]["training_history"]
    index = 0 if field == "base_eval" else 1
    name = "training_precision" if field == "training" else "evaluation_precision"
    assert study["variants"][index][name] is None
    page = boot(monkeypatch, tmp_path, assets)
    page.button(key=f"archive-select-{PRODUCT}").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert any("未记录" in item.value for item in page.caption)
    assert any(
        "loss" in frame.columns and len(frame) == 64 for frame in curve_frames(training_tab(page))
    )
    assert f"archive-select-{CONTROLLED}" in card_keys(page)
    assert not (tmp_path / "local").exists()
    assert snapshot(assets) == before


@pytest.mark.parametrize("invalid", ["", " ", 1, [], {}])
@pytest.mark.parametrize("study_id", [CONTROLLED, PRODUCT])
def test_invalid_precision_fails_detail_cleanly_but_other_study_remains(
    monkeypatch, tmp_path, assets, invalid, study_id
):
    def mutate(data):
        if study_id == CONTROLLED:
            data["controlled_v2"]["settings"]["dtype"] = invalid
        else:
            data["product_smoke"]["results"]["evaluations"]["trained"]["actual_dtype"] = invalid

    change(assets, mutate)
    before = snapshot(assets)
    with pytest.raises(ValueError, match="精度记录格式无效"):
        load_study(assets, study_id)
    other = PRODUCT if study_id == CONTROLLED else CONTROLLED
    assert load_study(assets, other)["id"] == other
    page = boot(monkeypatch, tmp_path, assets)
    page.button(key=f"archive-select-{study_id}").click().run()
    assert not page.exception
    assert "这份历史记录暂不可读" in warning_text(page)
    assert str(tmp_path) not in warning_text(page)
    page.button(key=f"archive-select-{other}").click().run()
    assert not page.exception
    assert any(item.label == "训练曲线" for item in page.tabs)
    assert not (tmp_path / "local").exists()
    assert snapshot(assets) == before
