"""Confirmed full data must survive grouping, immutable export and revisions."""

from pathlib import Path

import pytest

from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.workbench.intake_models import (
    DataRecipe,
    FieldBinding,
    FieldRole,
    IntakeAnalysis,
    TaskSpec,
)
from src.workbench.intake_service import IntakeService, next_action
from src.workbench.materialize import dataset_is_current

SAMPLE = b"customer,order,text,label\nsample-c,sample-o,sample text,yes\n"
FULL = (
    b"customer,order,text,label\n"
    b"c1,o1,a,yes\n"
    b"c1,o2,b,yes\n"
    b"c2,o2,c,yes\n"
    b"c3,o3,d,yes\n"
    b"c4,o4,e,yes\n"
    b"c5,o5,d,yes\n"
    b"c6,o6,f,yes\n"
    b"c6,o6,f,yes\n"
    b"c7,o7,g,yes\n"
)


def _analysis(group_columns):
    return IntakeAnalysis(
        task=TaskSpec(
            goal="根据业务文本分类",
            usage_input="用户提交的文本",
            desired_output="业务类别",
            row_meaning="一次业务记录",
            supervision_source="人工类别",
            success_criteria=["正确分类"],
            field_roles=[
                FieldRole(
                    column="customer",
                    role="group" if "customer" in group_columns else "metadata",
                    reason="客户",
                ),
                FieldRole(
                    column="order",
                    role="group" if "order" in group_columns else "metadata",
                    reason="订单",
                ),
                FieldRole(
                    column="text", role="input", reason="业务文本", available_at_prediction=True
                ),
                FieldRole(column="label", role="target", reason="人工答案"),
            ],
        ),
        findings=[],
        recipe=DataRecipe(
            instruction="分类",
            inputs=[FieldBinding(column="text", label="内容")],
            targets=[FieldBinding(column="label", label="类别", value_kind="categorical")],
            group_columns=group_columns,
            split_rationale="相同客户、订单不能跨分区",
        ),
        training_approach="SFT",
        next_steps=["查看全量预览"],
    )


@pytest.fixture
def service(tmp_path):
    return IntakeService(tmp_path / "intake")


def _full(service, *, data=FULL, group_columns=None, confirm=True):
    groups = ["customer", "order"] if group_columns is None else group_columns
    session = service.create("根据业务文本分类", "sample.csv", SAMPLE)
    session = service.apply_analysis(session, _analysis(groups))
    session = service.confirm(session.session_id, session.revision)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", data)
    if confirm:
        session = service.confirm_full_data(session.session_id, session.revision)
    return session


def _load(artifact):
    registry = LocalDatasetRegistry(artifact.registry_root)
    return {
        split: registry.load_split(artifact.name, artifact.version, split)
        for split in ("train", "validation", "test")
    }


def test_confirmed_full_data_is_materialized_without_row_loss_or_entity_leakage(service):
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    artifact = session.dataset
    assert next_action(session) == "ready_for_training_preflight"
    assert dataset_is_current(session)
    splits = _load(artifact)
    records = [row for part in splits.values() for row in part]
    assert all(splits.values())
    assert len(records) == len(session.full_data.source.rows) == 9
    assert {r["metadata"]["source_row_id"] for r in records} == {
        r.row_id for r in session.full_data.source.rows
    }
    assert all(r["metadata"]["source_digest"] == artifact.source_digest for r in records)
    by_row = {r["metadata"]["source_row_id"]: r for r in records}
    for expected in session.full_data.preview.rows:
        assert by_row[expected.row_id]["input"] == expected.input
        assert by_row[expected.row_id]["output"] == expected.target
    location = {}
    for split, part in splits.items():
        for row in part:
            for key in (
                ("customer", row["metadata"]["group"]["customer"]),
                ("order", row["metadata"]["group"]["order"]),
                ("input", row["input"]),
            ):
                assert location.setdefault(key, split) == split
    assert artifact.statistics["independent_groups"] == 5
    assert artifact.statistics["source_exact_duplicate_rows"] == 1
    assert artifact.statistics["rendered_exact_duplicate_rows"] == 2
    assert sum(artifact.statistics["row_counts"].values()) == 9
    assert artifact.statistics["stratified"] is False
    assert artifact.full_confirmed_revision == session.full_data.confirmed_revision
    assert service.load(session.session_id).dataset == artifact


def test_exported_config_uses_train_and_validation_without_resplitting(service):
    from config.base import DataConfig

    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    artifact = session.dataset
    config = DataConfig(**artifact.data_config)
    assert config.dataset_name == config.train_file == artifact.paths["train"]
    assert config.validation_file == artifact.paths["validation"]
    assert config.validation_split == 0
    assert config.max_samples is None
    assert config.format == "alpaca"
    assert artifact.paths["test"] not in artifact.data_config.values()
    assert all(Path(path).is_file() for path in artifact.paths.values())


def test_repeated_export_and_reconfirmation_reuse_immutable_content_version(service):
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision, seed=13)
    original = session.dataset
    before = {name: Path(path).read_bytes() for name, path in original.paths.items()}
    session = service.materialize_dataset(session.session_id, session.revision, seed=13)
    assert session.dataset.version == original.version
    session = service.validate_full_data(session.session_id, session.revision, "renamed.csv", FULL)
    assert session.dataset is None
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision, seed=13)
    assert session.dataset.version == original.version
    assert session.dataset.full_confirmed_revision > original.full_confirmed_revision
    assert before == {name: Path(path).read_bytes() for name, path in original.paths.items()}


@pytest.mark.parametrize("change", ["answer", "apply_analysis", "validate_full_data"])
def test_business_or_full_data_revision_invalidates_only_active_artifact(service, change):
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    artifact = session.dataset
    if change == "answer":
        session = service.answer(session.session_id, "修订任务")
    elif change == "apply_analysis":
        session = service.apply_analysis(session, _analysis(["customer", "order"]))
    else:
        session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    assert session.dataset is None
    assert next_action(session) != "ready_for_training_preflight"
    assert sum(map(len, _load(artifact).values())) == 9
    with pytest.raises(ValueError, match="确认当前全量"):
        service.materialize_dataset(session.session_id, session.revision)


@pytest.mark.parametrize(
    "options",
    [
        {"validation_fraction": 0},
        {"test_fraction": -0.1},
        {"validation_fraction": float("nan")},
        {"test_fraction": float("inf")},
        {"validation_fraction": True},
        {"validation_fraction": "0.1"},
        {"validation_fraction": 0.6, "test_fraction": 0.4},
        {"validation_fraction": 0.8, "test_fraction": 0.3},
        {"seed": 1.5},
    ],
)
def test_invalid_split_parameters_do_not_publish_or_change_session(service, options):
    session = _full(service)
    with pytest.raises(ValueError):
        service.materialize_dataset(session.session_id, session.revision, **options)
    assert service.load(session.session_id) == session
    assert not (service.root / "datasets").exists()


def test_without_group_fields_requires_explicit_independent_row_confirmation(service):
    session = _full(service, group_columns=[])
    with pytest.raises(ValueError, match="明确确认每行"):
        service.materialize_dataset(session.session_id, session.revision)
    session = service.materialize_dataset(
        session.session_id, session.revision, independent_rows_confirmed=True
    )
    assert session.dataset.statistics["independent_groups"] == 7
    input_location = {}
    for split, rows in _load(session.dataset).items():
        for row in rows:
            assert input_location.setdefault(row["input"], split) == split


@pytest.mark.parametrize("grouped", [True, False])
def test_fewer_than_three_independent_groups_blocks_instead_of_splitting_related_rows(
    service, grouped
):
    data = (
        "customer,order,text,label\nc1,o1,a,yes\nc1,o2,a,yes\nc2,o2,b,yes\n"
        if grouped
        else "customer,order,text,label\nc1,o1,a,yes\nc2,o2,a,yes\nc3,o3,b,yes\n"
    ).encode()
    session = _full(service, data=data, group_columns=None if grouped else [])
    with pytest.raises(ValueError, match="至少需要 3 个"):
        service.materialize_dataset(
            session.session_id, session.revision, independent_rows_confirmed=True
        )
    assert service.load(session.session_id).dataset is None


def test_unconfirmed_or_stale_report_cannot_be_materialized(service):
    session = _full(service, confirm=False)
    with pytest.raises(ValueError, match="确认当前全量"):
        service.materialize_dataset(session.session_id, session.revision)
    session = service.confirm_full_data(session.session_id, session.revision)
    with pytest.raises(ValueError, match="已变化"):
        service.materialize_dataset(session.session_id, session.revision - 1)
    session.full_data.preview.rows[0].target = "invented"
    session = service._save(session, session.revision)
    with pytest.raises(ValueError, match="预览与当前来源"):
        service.materialize_dataset(session.session_id, session.revision)


def test_rare_answer_landing_only_in_holdout_is_disclosed(service):
    """稀有答案整组落入保留分区:统计点名分区答案构成与训练集缺口,披露不重切。

    17 条 yes + 末行唯一一条 screen(18 个单行组),默认 seed 42 下 screen 整组
    落入测试集(14/2/2)——训练集从未见过该答案,逐字学习下模型无法输出没学过的值,
    而验证/测试照常打分。此前 statistics 对分区答案构成完全无声。
    """
    rare = (
        "customer,order,text,label\n"
        + "".join(f"c{i:02d},o{i:02d},question {i},yes\n" for i in range(1, 18))
        + "c18,o18,screen broken,screen\n"
    ).encode()
    session = _full(service, data=rare, group_columns=["customer"])
    session = service.materialize_dataset(session.session_id, session.revision)
    statistics = session.dataset.statistics
    assert statistics["row_counts"] == {"train": 14, "validation": 2, "test": 2}
    assert statistics["answer_counts_by_split"]["train"] == {"yes": 14}
    assert statistics["answer_counts_by_split"]["test"] == {"yes": 1, "screen": 1}
    assert statistics["train_missing_answers"] == {"screen": {"test": 1}}
    note = statistics["answer_coverage_note"]
    assert "1 类答案" in note and "screen×1（测试1 条）" in note, note
    assert "从未出现在训练集" in note and "逐字" in note and "照常打分" in note
    assert "没有自动重新切分" in note, note


def test_open_answer_space_skips_coverage_keys(service):
    """答案不同取值超过 20 种(开放文本形态):不逐值点名,缺键即如实边界。"""
    wide = (
        "customer,order,text,label\n"
        + "".join(f"c{i:02d},o{i:02d},question {i},answer {i}\n" for i in range(1, 23))
        + "c23,o23,question 23,answer 23\n"
    ).encode()
    session = _full(service, data=wide, group_columns=["customer"])
    session = service.materialize_dataset(session.session_id, session.revision)
    statistics = session.dataset.statistics
    assert statistics["row_counts"] == {"train": 19, "validation": 2, "test": 2}
    for key in ("answer_counts_by_split", "train_missing_answers", "answer_coverage_note"):
        assert key not in statistics, f"超过 20 种答案不应携带 {key}"
