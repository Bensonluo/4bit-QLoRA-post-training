"""Invalid numeric evidence cannot crash independent archive browsing."""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from tests.unit.test_archive_resilience_ui import (
    ARCHIVE,
    CONTROLLED,
    PRODUCT,
    ROOT,
    boot,
    card_keys,
    render_handoff,
    warning_text,
)
from ui.showcase import R148_SOURCE, R149_SOURCE, available_history_catalog, load_showcase
from ui.study_archive import LOGS, PREFIX, PROTOCOLS, STATE, load_study


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "assets"
    directory = root / "docs/validation"
    directory.mkdir(parents=True)
    for name in (R148_SOURCE, R149_SOURCE, ARCHIVE):
        shutil.copyfile(ROOT / "docs/validation" / name, directory / name)
    return root


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        if p.is_file()
        else None
        for p in root.rglob("*")
    }


@pytest.mark.parametrize(
    "change", ["huge_loss", "huge_metric", "huge_duration", "huge_directory", "huge_weight"]
)
def test_unreadable_r149_numeric_evidence_keeps_other_archives_and_home(
    monkeypatch, tmp_path, assets, change
):
    from streamlit.testing.v1 import AppTest

    path = assets / "docs/validation" / R149_SOURCE
    data = json.loads(path.read_text())
    invalid = 10**400
    if change == "huge_loss":
        data["actual_lr_and_optimizer_steps"][0]["loss"] = invalid
    elif change == "huge_metric":
        data["actual_report"]["models"][0]["metrics"]["exact_match"] = invalid
    elif change == "huge_duration":
        data["costs"]["preparation_seconds"] = invalid
    elif change == "huge_directory":
        data["artifact_inventory"]["artifacts"]["hf"]["directory_bytes"] = invalid
    else:
        artifact = data["artifact_inventory"]["artifacts"]["hf"]
        name = next(iter(artifact["weight_files"]))
        artifact["weight_files"][name]["size"] = invalid
        data["export_receipts"]["hf"]["output_files"][name]["size"] = invalid
        artifact["directory_bytes"] = invalid
    path.write_text(json.dumps(data, ensure_ascii=False))
    before = snapshot(assets)
    entries, unavailable = available_history_catalog(assets)
    assert [entry["id"] for entry in entries] == [CONTROLLED, PRODUCT]
    assert [entry["id"] for entry in unavailable] == ["r149"]
    with pytest.raises(ValueError):
        load_showcase(assets)
    page = boot(monkeypatch, tmp_path, assets)
    assert card_keys(page) == {f"archive-select-{CONTROLLED}", f"archive-select-{PRODUCT}"}
    assert "完整分类任务" in warning_text(page)
    assert str(tmp_path) not in warning_text(page)
    home = AppTest.from_function(render_handoff, args=(assets,), default_timeout=30).run()
    assert not home.exception
    assert any("暂不可读或核验失败" in warning.value for warning in home.warning)
    assert not (tmp_path / "local").exists()
    assert snapshot(assets) == before


def corrupt_log_and_rebind_archive(root: Path, study_id: str) -> None:
    directory = root / "docs/validation"
    source = directory / R148_SOURCE
    data = json.loads(source.read_text())
    archive = directory / ARCHIVE
    target = LOGS[PROTOCOLS[0]] if study_id == CONTROLLED else STATE
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        content = {
            member.name: tar.extractfile(member).read() for member in members if member.isfile()
        }
    target_name = PREFIX + target
    log = json.loads(content[target_name])
    if study_id == CONTROLLED:
        log[0]["loss"] = 10**400
        reference = data["controlled_v2"]["results"]["arms"][PROTOCOLS[0]][
            "optimizer_update_evidence"
        ]["full_log"]
    else:
        log["log_history"][0]["loss"] = 10**400
        reference = data["product_smoke"]["trainer_state"]
    content[target_name] = json.dumps(log, ensure_ascii=False).encode()
    manifest_name = PREFIX + "manifest.json"
    manifest = json.loads(content[manifest_name])
    digest = hashlib.sha256(content[target_name]).hexdigest()
    manifest[target]["sha256"] = digest
    manifest[target]["size"] = len(content[target_name])
    reference["sanitized_sha256"] = digest
    content[manifest_name] = json.dumps(manifest, ensure_ascii=False).encode()
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        for member in members:
            if member.isfile():
                payload = content[member.name]
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
            else:
                tar.addfile(member)
    archive.write_bytes(output.getvalue())
    data["detailed_evidence_archive"]["bytes"] = len(output.getvalue())
    data["detailed_evidence_archive"]["sha256"] = hashlib.sha256(output.getvalue()).hexdigest()
    source.write_text(json.dumps(data, ensure_ascii=False))


@pytest.mark.parametrize("study_id,healthy", [(CONTROLLED, PRODUCT), (PRODUCT, CONTROLLED)])
def test_rebound_archive_with_huge_loss_shows_error_and_preserves_other_study(
    monkeypatch, tmp_path, assets, study_id, healthy
):
    corrupt_log_and_rebind_archive(assets, study_id)
    before = snapshot(assets)
    with pytest.raises(ValueError, match="无法读取或核验"):
        load_study(assets, study_id)
    assert load_study(assets, healthy)["id"] == healthy
    page = boot(monkeypatch, tmp_path, assets)
    page.button(key=f"archive-select-{study_id}").click().run()
    assert not page.exception, [item.message for item in page.exception]
    assert "这份历史记录暂不可读" in warning_text(page)
    assert str(tmp_path) not in warning_text(page)
    page.button(key=f"archive-select-{healthy}").click().run()
    assert not page.exception
    assert any(item.label == "训练曲线" for item in page.tabs)
    assert not (tmp_path / "local").exists()
    assert snapshot(assets) == before


@pytest.mark.parametrize("study_id,healthy", [(CONTROLLED, PRODUCT), (PRODUCT, CONTROLLED)])
@pytest.mark.parametrize("invalid", [10**400, float("nan"), float("inf"), -1, 0, True])
def test_invalid_recorded_learning_rate_cannot_reach_history(study_id, healthy, invalid, assets):
    source = assets / "docs/validation" / R148_SOURCE
    data = json.loads(source.read_text())
    if study_id == CONTROLLED:
        data["controlled_v2"]["settings"]["learning_rate"] = invalid
    else:
        data["product_smoke"]["settings"]["configs"]["training"]["learning_rate"] = invalid
    source.write_text(json.dumps(data, ensure_ascii=False))
    before = snapshot(assets)
    with pytest.raises(ValueError, match="无法读取或核验"):
        load_study(assets, study_id)
    assert load_study(assets, healthy)["id"] == healthy
    assert snapshot(assets) == before
