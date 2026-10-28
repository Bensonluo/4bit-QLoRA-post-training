"""Blind verification must bind the parsed supervision, not only raw file bytes."""

from io import BytesIO

import pytest
from openpyxl import Workbook

from src.workbench.intake_service import IntakeService
from src.workbench.training_runs import _require_verified_labels
from tests.unit.test_full_data import approved


def workbook():
    book = Workbook()
    first = book.active
    first.title = "一月"
    second = book.create_sheet("二月")
    for sheet, prefix in ((first, "甲"), (second, "乙")):
        sheet.append(["编号", "客户描述", "类别", "处理结果"])
        for index in range(6):
            sheet.append([f"{prefix}{index}", f"{prefix}问题{index}", "质量", "补发"])
    output = BytesIO()
    book.save(output)
    return output.getvalue()


def confirm_sheet(service, session, data, name):
    session = service.validate_full_data(
        session.session_id, session.revision, "records.xlsx", data, sheet=name
    )
    return service.confirm_full_data(session.session_id, session.revision)


def verify(service, session):
    challenge = service.start_label_verification(session.session_id, session.revision)
    targets = {row.row_id: row.target for row in session.full_data.preview.rows}
    result = service.submit_label_verification(
        session.session_id,
        challenge["verification_id"],
        {row["row_id"]: targets[row["row_id"]] for row in challenge["items"]},
    )
    assert result["verdict"] == "verified"
    return challenge


def test_same_workbook_changed_sheet_does_not_carry_verified_verdict(tmp_path):
    service = IntakeService(tmp_path / "intake")
    first = confirm_sheet(service, approved(service), workbook(), "一月")
    data = (
        service.root / first.session_id / "full" / f"{first.full_data.source.digest}.xlsx"
    ).read_bytes()
    verify(service, first)
    first = service.load(first.session_id)
    assert first.label_verification["verdict"] == "verified"
    second = confirm_sheet(service, first, data, "二月")
    assert second.full_data.source.digest == first.full_data.source.digest
    assert second.full_data.preview.rows[0].input != first.full_data.preview.rows[0].input
    assert service.load(second.session_id).label_verification.get("stale") is True
    with pytest.raises(ValueError, match="盲标核验"):
        _require_verified_labels(service.load(second.session_id), action="启动")
    restored = confirm_sheet(service, second, data, "一月")
    assert service.load(restored.session_id).label_verification["verdict"] == "verified"


def test_pending_challenge_cannot_be_submitted_against_different_sheet(tmp_path):
    service = IntakeService(tmp_path / "intake")
    data = workbook()
    first = confirm_sheet(service, approved(service), data, "一月")
    pending = service.start_label_verification(first.session_id, first.revision)
    answers = {item["row_id"]: "质量" for item in pending["items"]}
    second = confirm_sheet(service, first, data, "二月")
    assert service.load(second.session_id).label_verification["stale"] is True
    assert service.load(second.session_id).label_verification["previous_verdict"] == "未完成"
    with pytest.raises(ValueError, match="盲标核验"):
        _require_verified_labels(service.load(second.session_id), action="启动")
    with pytest.raises(ValueError, match="核验失效"):
        service.submit_label_verification(second.session_id, pending["verification_id"], answers)
    assert service.load(second.session_id).label_verification["stale"] is True
    current = service.start_label_verification(second.session_id, second.revision)
    assert current["verification_id"] != pending["verification_id"]
    assert all("乙问题" in item["input"] for item in current["items"])


def test_legacy_raw_only_verdict_is_not_silently_trusted(tmp_path):
    import sqlite3

    service = IntakeService(tmp_path / "intake")
    first = confirm_sheet(service, approved(service), workbook(), "一月")
    verified = verify(service, first)
    with sqlite3.connect(service.database) as connection:
        connection.execute(
            "UPDATE label_verifications SET full_source_digest=? WHERE verification_id=?",
            (first.full_data.source.digest, verified["verification_id"]),
        )
    loaded = service.load(first.session_id)
    assert loaded.label_verification["stale"] is True
    assert loaded.label_verification["previous_verdict"] == "verified"
    with pytest.raises(ValueError, match="盲标核验"):
        _require_verified_labels(loaded, action="启动")
    new = service.start_label_verification(first.session_id, first.revision, sample_size=1)
    assert new["items"][0]["input"] not in {item["input"] for item in verified["items"]}


def test_legacy_raw_only_pending_is_stale_and_cannot_be_submitted(tmp_path):
    import sqlite3

    service = IntakeService(tmp_path / "intake")
    first = confirm_sheet(service, approved(service), workbook(), "一月")
    pending = service.start_label_verification(first.session_id, first.revision)
    with sqlite3.connect(service.database) as connection:
        connection.execute(
            "UPDATE label_verifications SET full_source_digest=? WHERE verification_id=?",
            (first.full_data.source.digest, pending["verification_id"]),
        )
    loaded = service.load(first.session_id)
    assert loaded.label_verification["stale"] is True
    assert loaded.label_verification["previous_verdict"] == "未完成"
    with pytest.raises(ValueError, match="盲标核验"):
        _require_verified_labels(loaded, action="启动")
    with pytest.raises(ValueError, match="核验失效"):
        service.submit_label_verification(
            first.session_id,
            pending["verification_id"],
            {row["row_id"]: "质量" for row in pending["items"]},
        )
    assert service.load(first.session_id).label_verification["stale"] is True


def test_read_without_any_verification_does_not_hash_whole_data(tmp_path, monkeypatch):
    import src.workbench.intake_service as intake_service

    service = IntakeService(tmp_path / "intake")
    session = confirm_sheet(service, approved(service), workbook(), "一月")

    def unexpected_hash(_):
        raise AssertionError("ordinary task read must not hash all data")

    monkeypatch.setattr(intake_service, "_label_source_view_digest", unexpected_hash)
    assert service.load(session.session_id).label_verification is None
