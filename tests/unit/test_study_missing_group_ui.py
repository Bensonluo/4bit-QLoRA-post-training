"""A study remains browsable when optional group metadata is unavailable."""

import hashlib
import json
import shutil

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
from ui.showcase import R148_SOURCE, R149_SOURCE


@pytest.fixture
def copied_assets(tmp_path):
    root = tmp_path / "assets"
    directory = root / "docs/validation"
    directory.mkdir(parents=True)
    for name in (R148_SOURCE, R149_SOURCE, ARCHIVE):
        shutil.copyfile(ROOT / "docs/validation" / name, directory / name)
    return root


def digest_tree(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("study_id", [CONTROLLED, PRODUCT])
@pytest.mark.parametrize("missing_group", ["missing", "empty", "null", "number", "object"])
def test_missing_group_keeps_actual_case_and_training_without_writes(
    monkeypatch, tmp_path, copied_assets, study_id, missing_group
):
    path = copied_assets / "docs/validation" / R148_SOURCE
    data = json.loads(path.read_text())
    case = data["controlled_v2"]["data"]["development"][0]
    original_input = case["input"]
    original_answer = case["output"]
    if missing_group == "missing":
        del case["group_id"]
    else:
        case["group_id"] = {"empty": " ", "null": None, "number": 7, "object": {}}[missing_group]
    path.write_text(json.dumps(data, ensure_ascii=False))
    before = digest_tree(copied_assets)
    page = boot(monkeypatch, tmp_path, copied_assets)
    assert f"archive-select-{study_id}" in card_keys(page)
    page.button(key=f"archive-select-{study_id}").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert page.selectbox(key="archive-selection").value == study_id
    assert any("业务输入 · 分组 未记录或不可核验" in item.value for item in page.caption)
    assert any(item.value.strip() == original_input.strip() for item in page.code)
    assert any(item.value.strip() == original_answer.strip() for item in page.code)
    assert any(item.label == "训练曲线" for item in page.tabs)
    assert not (tmp_path / "local").exists()
    assert digest_tree(copied_assets) == before


@pytest.mark.parametrize("study_id", [CONTROLLED, PRODUCT])
def test_missing_shared_case_input_explains_detail_gap_without_fabrication(
    monkeypatch, tmp_path, copied_assets, study_id
):
    path = copied_assets / "docs/validation" / R148_SOURCE
    data = json.loads(path.read_text())
    del data["controlled_v2"]["data"]["development"][0]["input"]
    path.write_text(json.dumps(data, ensure_ascii=False))
    before = digest_tree(copied_assets)
    page = boot(monkeypatch, tmp_path, copied_assets)
    assert f"archive-select-{study_id}" in card_keys(page)
    page.button(key=f"archive-select-{study_id}").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert "这份历史记录暂不可读" in warning_text(page)
    assert str(tmp_path) not in warning_text(page)
    assert not (tmp_path / "local").exists()
    assert digest_tree(copied_assets) == before
