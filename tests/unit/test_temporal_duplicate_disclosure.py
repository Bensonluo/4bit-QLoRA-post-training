"""Dataset duplicate warnings must describe the records actually used in each split."""

import json
from pathlib import Path

import pytest

from src.workbench.intake_service import IntakeService
from tests.unit.test_temporal_intake_entrypoints import DATA, temporal_session


@pytest.mark.parametrize(
    ("duplicated_row", "expected_split", "expected_impact"),
    [
        ("t", "train", "训练集中有 1 条重复记录，训练等效于给这些例题加权"),
        ("v", "validation", "训练集没有重复例题；验证集有 1 条、测试集有 0 条"),
        ("h", "test", "训练集没有重复例题；验证集有 0 条、测试集有 1 条"),
    ],
)
def test_duplicate_disclosure_counts_only_published_examples_and_names_the_split(
    tmp_path, duplicated_row, expected_split, expected_impact
):
    lines = DATA.decode().splitlines()
    selected = next(line for line in lines[1:] if line.startswith(f"{duplicated_row},"))
    excluded = next(line for line in lines[1:] if line.startswith("x,"))
    data = ("\n".join([*lines, selected, excluded]) + "\n").encode()
    service = IntakeService(tmp_path / "intake")
    session = temporal_session(service, data=data)
    session = service.materialize_dataset(session.session_id, session.revision)
    artifact = session.dataset
    stats = artifact.statistics
    assert stats["included_rows"] == 4
    assert stats["excluded_rows"] == 3
    assert stats["rendered_exact_duplicate_rows"] == 1
    assert stats["source_exact_duplicate_rows"] == 1
    assert stats["rendered_exact_duplicate_rows_by_split"] == {
        split: int(split == expected_split) for split in ("train", "validation", "test")
    }
    note = stats["duplicate_note"]
    assert expected_impact in note
    assert "4 条纳入记录去重后只有 3 道独立例题" in note
    if expected_split != "train":
        assert "训练等效于" not in note
    manifest = json.loads(Path(artifact.paths["manifest"]).read_text())
    assert manifest["metadata"]["statistics"] == stats


def test_only_excluded_duplicates_do_not_create_training_warning(tmp_path):
    lines = DATA.decode().splitlines()
    excluded = next(line for line in lines[1:] if line.startswith("x,"))
    service = IntakeService(tmp_path / "intake")
    session = temporal_session(service, data=("\n".join([*lines, excluded]) + "\n").encode())
    session = service.materialize_dataset(session.session_id, session.revision)
    stats = session.dataset.statistics
    assert stats["included_rows"] == 3 and stats["excluded_rows"] == 3
    assert stats["rendered_exact_duplicate_rows"] == 0
    assert stats["source_exact_duplicate_rows"] == 0
    assert "duplicate_note" not in stats
