"""A preflight validates every immutable split without rereading the package."""

from pathlib import Path

import pytest

from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.workbench.intake_service import IntakeService
from src.workbench.training_preflight import _verified_partitions
from tests.unit.test_business_evaluation import model_paths as model_paths
from tests.unit.test_full_data import FULL, approved


@pytest.fixture
def materialized(tmp_path):
    intake = IntakeService(tmp_path / "intake")
    session = approved(intake)
    session = intake.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = intake.confirm_full_data(session.session_id, session.revision)
    return intake.materialize_dataset(session.session_id, session.revision)


def test_preflight_reads_each_manifest_and_split_once(materialized, monkeypatch):
    artifact = materialized.dataset
    files = {Path(path).resolve() for path in artifact.paths.values()}
    reads = dict.fromkeys(files, 0)
    real_read = Path.read_bytes

    def counting_read(path):
        resolved = path.resolve()
        if resolved in reads:
            reads[resolved] += 1
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read)
    before = {path: real_read(path) for path in files}
    report = {"issues": []}
    partitions = _verified_partitions(materialized, report)
    assert set(partitions) == {"train", "validation", "test"}
    assert all(partitions.values())
    assert report["issues"] == []
    assert {path.name: count for path, count in reads.items()} == {
        "manifest.json": 1,
        "train.jsonl": 1,
        "validation.jsonl": 1,
        "test.jsonl": 1,
    }
    assert {path: real_read(path) for path in files} == before


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_preflight_still_rejects_each_modified_split(materialized, split):
    artifact = materialized.dataset
    path = Path(artifact.paths[split])
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        _verified_partitions(materialized, {"issues": []})
    path.write_bytes(original)
    assert _verified_partitions(materialized, {"issues": []})[split]


def test_complete_package_matches_legacy_read_apis(materialized):
    artifact = materialized.dataset
    registry = LocalDatasetRegistry(artifact.registry_root)
    manifest, partitions = registry.load_split_package(artifact.name, artifact.version)
    assert manifest == registry.get_split_manifest(artifact.name, artifact.version)
    for split in ("train", "validation", "test"):
        assert partitions[split] == registry.load_split(artifact.name, artifact.version, split)


@pytest.mark.parametrize("fixed_suite", [False, True])
def test_actual_comparison_reads_package_once_and_uses_verified_cases(
    tmp_path, monkeypatch, fixed_suite, request
):
    from src.workbench.business_evaluation import BusinessEvaluationService, EvaluationProtocol
    from src.workbench.evaluation_suites import EvalSuiteService, evaluation_cases
    from tests.unit.test_business_evaluation import fixture_factory

    # Use the actual business-comparison fixtures and immutable partitions.
    intake = IntakeService(tmp_path / "intake")
    session = approved(intake)
    session = intake.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = intake.confirm_full_data(session.session_id, session.revision)
    session = intake.materialize_dataset(session.session_id, session.revision)
    if fixed_suite:
        reference = EvalSuiteService(tmp_path / "suites").freeze(session)
        session = session.model_copy(deep=True)
        session.dataset.evaluation_suite = reference
    expected_cases = evaluation_cases(session)
    models = request.getfixturevalue("model_paths")
    runtime = fixture_factory(session, [])
    files = {Path(path).resolve() for path in session.dataset.paths.values()}
    counts = dict.fromkeys(files, 0)
    real_read = Path.read_bytes

    def counting_read(path):
        resolved = path.resolve()
        if resolved in counts:
            counts[resolved] += 1
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read)
    report = BusinessEvaluationService(tmp_path / "evaluations").compare(
        session, models, EvaluationProtocol("classification_exact"), runtime_factory=runtime
    )
    assert report.status == "completed"
    assert len(report.models) == 2
    assert report.dataset["row_count"] == len(expected_cases["records"])
    if fixed_suite:
        assert report.protocol["evaluation_key"] == expected_cases["evaluation_key"]
    assert {path.name: count for path, count in counts.items()} == {
        "manifest.json": 1,
        "train.jsonl": 1,
        "validation.jsonl": 1,
        "test.jsonl": 1,
    }


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_comparison_refuses_any_modified_partition_before_loading_model(
    materialized, tmp_path, split
):
    from src.workbench.business_evaluation import (
        BusinessEvaluationService,
        EvaluationModel,
        EvaluationProtocol,
    )

    artifact = materialized.dataset
    path = Path(artifact.paths[split])
    path.write_bytes(path.read_bytes() + b" ")
    invoked = []

    def runtime(model):
        invoked.append(model)
        raise AssertionError("model must not load for an invalid partition")

    with pytest.raises(ValueError, match="hash mismatch"):
        BusinessEvaluationService(tmp_path / "eval").compare(
            materialized,
            [EvaluationModel("one", "unused"), EvaluationModel("two", "unused")],
            EvaluationProtocol("classification_exact"),
            runtime_factory=runtime,
        )
    assert not invoked


def test_missing_registry_is_not_created_by_reading(materialized, tmp_path):
    missing = tmp_path / "missing-parent" / "registry"
    changed = materialized.model_copy(deep=True)
    changed.dataset.registry_root = str(missing)
    with pytest.raises(ValueError, match="存储目录不存在"):
        _verified_partitions(changed, {"issues": []})
    assert not missing.parent.exists()


@pytest.fixture(params=[False, True], ids=["ordinary", "fixed_suite"])
def comparison_with_session(tmp_path, request):
    from src.workbench.business_evaluation import (
        BusinessEvaluationService,
        EvaluationProtocol,
        Generation,
    )
    from src.workbench.evaluation_suites import EvalSuiteService
    from tests.unit.test_business_evaluation import fixture_factory

    intake = IntakeService(tmp_path / "intake")
    session = approved(intake)
    session = intake.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = intake.confirm_full_data(session.session_id, session.revision)
    session = intake.materialize_dataset(session.session_id, session.revision)
    if request.param:
        reference = EvalSuiteService(tmp_path / "suites").freeze(session)
        session = session.model_copy(deep=True)
        session.dataset.evaluation_suite = reference
    models = request.getfixturevalue("model_paths")
    report = BusinessEvaluationService(tmp_path / "evaluations").compare(
        session,
        models,
        EvaluationProtocol("classification_exact"),
        runtime_factory=fixture_factory(session, [], lambda *_: Generation("错误答案")),
    )
    return session, report


def test_real_diagnostics_reads_each_verified_file_once(comparison_with_session, monkeypatch):
    from src.workbench.evaluation_diagnostics import EvaluationDiagnostics

    session, report = comparison_with_session
    files = {Path(path).resolve() for path in session.dataset.paths.values()}
    real_read = Path.read_bytes
    before = {path: real_read(path) for path in files}
    counts = dict.fromkeys(files, 0)

    def counting_read(path):
        resolved = path.resolve()
        if resolved in counts:
            counts[resolved] += 1
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", counting_read)
    diagnostic = EvaluationDiagnostics(report, session)
    assert diagnostic.read_cases()["total"] >= 1
    assert {path.name: count for path, count in counts.items()} == {
        "manifest.json": 1,
        "train.jsonl": 1,
        "validation.jsonl": 1,
        "test.jsonl": 1,
    }
    assert {path: real_read(path) for path in files} == before


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_real_diagnostics_refuses_any_modified_partition(comparison_with_session, split):
    from src.workbench.evaluation_diagnostics import EvaluationDiagnostics

    session, report = comparison_with_session
    path = Path(session.dataset.paths[split])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        EvaluationDiagnostics(report, session)


def test_diagnostics_does_not_create_missing_registry(comparison_with_session, tmp_path):
    from src.workbench.evaluation_diagnostics import EvaluationDiagnostics

    session, report = comparison_with_session
    missing = tmp_path / "missing-parent" / "registry"
    session = session.model_copy(deep=True)
    session.dataset.registry_root = str(missing)
    with pytest.raises(ValueError, match="存储目录不存在"):
        EvaluationDiagnostics(report, session)
    assert not missing.parent.exists()
