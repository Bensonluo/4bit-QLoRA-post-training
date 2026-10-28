"""Reusing full sources must follow the current parsed view, not only file bytes."""

import pytest

from src.workbench.intake_service import IntakeService
from tests.unit.test_composed_intake import _workbook_bytes, analysis_for, files


def full_composed_session(tmp_path, name, data, **options):
    service = IntakeService(tmp_path / "intake")
    session = service.create("预测工单类型", *files(3)["main"], scope="full")
    session = service.add_source(
        session.session_id, session.revision, "labels", name, data, scope="full", **options
    )
    session = service.apply_analysis(session, analysis_for(session))
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_sources(session.session_id, session.revision)
    return service, session


def test_switching_sheet_with_identical_workbook_does_not_reuse_old_full_view(tmp_path):
    workbook = _workbook_bytes(
        ("一月", ("id", "category"), [(str(i), "咨询") for i in range(3)]),
        ("二月", ("id", "category"), [(str(i), "投诉") for i in range(3)]),
    )
    service, session = full_composed_session(tmp_path, "labels.xlsx", workbook, sheet="一月")
    assert {row.target for row in session.full_data.preview.rows} == {"咨询"}
    previous_digest = session.full_data.sources["labels"].digest
    changed = service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "labels.xlsx",
        workbook,
        scope="full",
        sheet="二月",
    )
    assert changed.sources["labels"].digest == previous_digest
    assert changed.sources["labels"].sheet == "二月"
    assert changed.full_data.sources["labels"].sheet == "二月"
    assert changed.full_data.sources["main"] == session.full_data.sources["main"]
    changed = service.apply_analysis(changed, analysis_for(changed))
    changed = service.confirm(changed.session_id, changed.revision)
    changed = service.validate_full_sources(
        changed.session_id,
        changed.revision,
    )
    assert changed.full_data.sources["labels"].sheet == "二月"
    assert {row.target for row in changed.full_data.preview.rows} == {"投诉"}


def test_same_bytes_new_encoding_replaces_cached_full_view(tmp_path):
    data = ("id,category\n" + "".join(f"{i},咨询\n" for i in range(3))).encode()
    service, session = full_composed_session(tmp_path, "labels.csv", data, encoding="utf-8")
    old = session.full_data.sources["labels"]
    changed = service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "labels.csv",
        data,
        scope="full",
        encoding="latin-1",
    )
    assert changed.sources["labels"].digest == old.digest
    assert changed.sources["labels"].rows[0].values["category"] != old.rows[0].values["category"]
    assert changed.full_data.sources["labels"].encoding == "latin-1"
    changed = service.apply_analysis(changed, analysis_for(changed))
    changed = service.confirm(changed.session_id, changed.revision)
    changed = service.validate_full_sources(changed.session_id, changed.revision)
    assert changed.full_data.sources["labels"].encoding == "latin-1"
    assert {row.target for row in changed.full_data.preview.rows} == {
        changed.sources["labels"].rows[0].values["category"]
    }


def test_identical_view_keeps_original_full_source_even_when_adding_sample(tmp_path):
    workbook = _workbook_bytes(
        ("一月", ("id", "category"), [(str(i), "咨询") for i in range(3)]),
        ("二月", ("id", "category"), [(str(i), "投诉") for i in range(3)]),
    )
    service, session = full_composed_session(tmp_path, "labels.xlsx", workbook, sheet="一月")
    changed = service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "labels-renamed.xlsx",
        workbook,
        scope="sample",
        sheet="一月",
    )
    assert changed.full_data.sources["labels"].sheet == "一月"
    changed = service.apply_analysis(changed, analysis_for(changed))
    changed = service.confirm(changed.session_id, changed.revision)
    changed = service.validate_full_sources(changed.session_id, changed.revision)
    assert {row.target for row in changed.full_data.preview.rows} == {"咨询"}


def test_switching_back_reuses_the_explicitly_selected_full_view(tmp_path):
    workbook = _workbook_bytes(
        ("一月", ("id", "category"), [(str(i), "咨询") for i in range(3)]),
        ("二月", ("id", "category"), [(str(i), "投诉") for i in range(3)]),
    )
    service, session = full_composed_session(tmp_path, "labels.xlsx", workbook, sheet="一月")
    changed = service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "labels.xlsx",
        workbook,
        scope="full",
        sheet="二月",
    )
    restored = service.add_source(
        changed.session_id,
        changed.revision,
        "labels",
        "labels.xlsx",
        workbook,
        scope="full",
        sheet="一月",
    )
    assert restored.full_data.sources["labels"].sheet == "一月"
    restored = service.apply_analysis(restored, analysis_for(restored))
    restored = service.confirm(restored.session_id, restored.revision)
    restored = service.validate_full_sources(restored.session_id, restored.revision)
    assert restored.full_data.sources["labels"].sheet == "一月"
    assert {row.target for row in restored.full_data.preview.rows} == {"咨询"}


def test_new_sample_view_cannot_reuse_different_old_full_view(tmp_path):
    workbook = _workbook_bytes(
        ("一月", ("id", "category"), [(str(i), "咨询") for i in range(3)]),
        ("二月", ("id", "category"), [(str(i), "投诉") for i in range(3)]),
    )
    service, session = full_composed_session(tmp_path, "labels.xlsx", workbook, sheet="一月")
    changed = service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "labels.xlsx",
        workbook,
        scope="sample",
        sheet="二月",
    )
    assert "labels" not in changed.full_data.sources
    changed = service.apply_analysis(changed, analysis_for(changed))
    changed = service.confirm(changed.session_id, changed.revision)
    with pytest.raises(ValueError, match="labels"):
        service.validate_full_sources(changed.session_id, changed.revision)


def test_composed_sheet_change_does_not_reuse_prior_blind_verification(tmp_path):
    workbook = _workbook_bytes(
        ("一月", ("id", "category"), [(str(i), "咨询") for i in range(3)]),
        ("二月", ("id", "category"), [(str(i), "投诉") for i in range(3)]),
    )
    service, session = full_composed_session(tmp_path, "labels.xlsx", workbook, sheet="一月")
    session = service.confirm_full_data(session.session_id, session.revision)
    challenge = service.start_label_verification(session.session_id, session.revision)
    result = service.submit_label_verification(
        session.session_id,
        challenge["verification_id"],
        {item["row_id"]: "咨询" for item in challenge["items"]},
    )
    assert result["verdict"] == "verified"
    session = service.load(session.session_id)
    changed = service.add_source(
        session.session_id,
        session.revision,
        "labels",
        "labels.xlsx",
        workbook,
        scope="full",
        sheet="二月",
    )
    changed = service.apply_analysis(changed, analysis_for(changed))
    changed = service.confirm(changed.session_id, changed.revision)
    changed = service.validate_full_sources(changed.session_id, changed.revision)
    loaded = service.load(changed.session_id)
    assert loaded.full_data.sources["labels"].sheet == "二月"
    assert {row.target for row in loaded.full_data.preview.rows} == {"投诉"}
    assert loaded.label_verification["stale"] is True
