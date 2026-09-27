"""Preflight uses local real tokenizer offsets and immutable full-data artifacts."""

from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("datasets")

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from src.data_flywheel.dataset_registry import LocalDatasetRegistry
from src.workbench.intake_service import IntakeService
from src.workbench.training_preflight import load_local_tokenizer, preflight_dataset
from tests.unit.test_data_materialize import FULL, _full


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "yes": 4}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


@pytest.fixture
def exported(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = _full(service)
    session = service.materialize_dataset(session.session_id, session.revision)
    return service, session


def _republish(session, mutate):
    artifact = session.dataset
    registry = LocalDatasetRegistry(artifact.registry_root)
    manifest = registry.get_split_manifest(artifact.name, artifact.version)
    splits = {
        key: registry.load_split(artifact.name, artifact.version, key)
        for key in ("train", "validation", "test")
    }
    mutate(splits, manifest["metadata"])
    old_version = artifact.version
    artifact.version = registry.register_splits(
        artifact.name, splits, metadata=manifest["metadata"]
    )
    artifact.paths = {
        key: path.replace(old_version, artifact.version) for key, path in artifact.paths.items()
    }
    artifact.data_config = {
        key: value.replace(old_version, artifact.version) if isinstance(value, str) else value
        for key, value in artifact.data_config.items()
    }


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("shared_pad_eos", [False, True])
def test_real_local_tokenizer_reports_actual_formatter_and_padding_supervision(
    exported, tokenizer, padding_side, shared_pad_eos
):
    from datasets import Dataset

    from src.data.loaders import AlpacaDataset
    from src.data.sft_collator import AttentionMaskCausalCollator

    _, session = exported
    tokenizer.padding_side = padding_side
    if shared_pad_eos:
        tokenizer.pad_token = tokenizer.eos_token
    report = preflight_dataset(session, tokenizer, 31)
    assert report["status"] == "passed"
    assert report["supervision_strategy"] == "full_prompt_padding_masked"
    assert len(report["rows"]) == 9
    assert {issue["code"] for issue in report["issues"]} == {"full_prompt_supervision"}
    artifact = session.dataset
    registry = LocalDatasetRegistry(artifact.registry_root)
    raw = registry.load_split(artifact.name, artifact.version, "train")
    loader = AlpacaDataset("unused")
    loader.dataset = Dataset.from_list(raw)
    formatted = loader.format_for_training(tokenizer, max_length=31)
    batch = AttentionMaskCausalCollator(tokenizer)(list(formatted))
    for index, row in enumerate(raw):
        observation = next(
            item for item in report["rows"] if item["row_id"] == row["metadata"]["source_row_id"]
        )
        labels = batch["labels"][index]
        assert observation["supervised_tokens"] == (labels[1:] != -100).sum().item()
        assert (labels[batch["attention_mask"][index] == 0] == -100).all()
        assert observation["terminal_eos_supervised"] is True
        assert observation["answer_tokens"] == observation["answer_supervised_tokens"] == 1
        assert observation["input_tokens"] > observation["answer_tokens"]
        assert observation["supervised_tokens"] > observation["answer_supervised_tokens"]
        assert observation["answer_boundary_verified"] is True


def test_right_truncation_that_removes_answer_blocks_with_exact_source_rows(exported, tokenizer):
    _, session = exported
    report = preflight_dataset(session, tokenizer, 6)
    assert report["status"] == "blocked"
    missing = {item["row_id"] for item in report["rows"] if item["answer_supervised_tokens"] == 0}
    assert missing == {row.row_id for row in session.full_data.source.rows}
    assert all(part["truncation_ratio"] == 1 for part in report["splits"].values())
    assert any(issue["code"] == "answer_lost" and issue["row_ids"] for issue in report["issues"])


def test_left_truncation_keeps_answer_but_requires_context_review(exported, tokenizer):
    _, session = exported
    tokenizer.truncation_side = "left"
    report = preflight_dataset(session, tokenizer, 6)
    assert report["status"] == "warnings"
    assert all(row["answer_supervised_tokens"] == 1 for row in report["rows"])
    assert all(row["answer_was_truncated"] is False for row in report["rows"])


def test_partial_answer_truncation_is_reported_without_claiming_complete_coverage(
    tmp_path, tokenizer
):
    service = IntakeService(tmp_path / "intake")
    session = _full(service, data=FULL.replace(b",yes\n", b",yes yes yes yes yes\n"))
    session = service.materialize_dataset(session.session_id, session.revision)
    complete = preflight_dataset(session, tokenizer, 64)
    length = complete["rows"][0]["input_tokens"] + 2
    report = preflight_dataset(session, tokenizer, length)
    assert report["status"] == "warnings"
    assert all(0 < row["answer_kept_tokens"] < row["answer_tokens"] for row in report["rows"])
    assert all(row["answer_was_truncated"] for row in report["rows"])


def test_one_token_leaves_no_shifted_loss_and_missing_pad_blocks(exported, tokenizer):
    _, session = exported
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(single="$A")
    report = preflight_dataset(session, tokenizer, 1)
    assert any(issue["code"] == "empty_supervision" for issue in report["issues"])
    tokenizer.pad_token = None
    assert preflight_dataset(session, tokenizer, 32)["status"] == "blocked"


def test_special_token_overhead_cannot_silently_exceed_requested_length(exported, tokenizer):
    report = preflight_dataset(exported[1], tokenizer, 1)
    assert report["status"] == "blocked"
    assert all(row["kept_tokens"] == 1 for row in report["rows"])
    assert any(issue["code"] == "empty_supervision" for issue in report["issues"])


def test_pad_equal_eos_preserves_real_end_token_supervision(exported, tokenizer):
    _, session = exported
    tokenizer.pad_token = tokenizer.eos_token
    report = preflight_dataset(session, tokenizer, 32)
    assert report["status"] == "passed"
    assert all(row["terminal_eos_supervised"] for row in report["rows"])
    assert all(row["answer_supervised_tokens"] == 1 for row in report["rows"])


def test_no_offsets_never_claims_verified_answer_coverage(exported, tokenizer):
    class NoOffsets:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, prompt, **options):
            if options.pop("return_offsets_mapping", False):
                raise NotImplementedError("slow tokenizer")
            return tokenizer(prompt, **options)

    _, session = exported
    report = preflight_dataset(session, NoOffsets(), 32)
    assert report["status"] == "warnings"
    assert all(
        row["answer_tokens"] is None and not row["answer_boundary_verified"]
        for row in report["rows"]
    )


@pytest.mark.parametrize("max_length", [0, -1, 1.5, True, "512"])
def test_invalid_max_length_is_rejected(exported, tokenizer, max_length):
    with pytest.raises(ValueError, match="正整数"):
        preflight_dataset(exported[1], tokenizer, max_length)


def test_hash_tampering_blocks_before_tokenization(exported, tokenizer):
    _, session = exported
    path = Path(session.dataset.paths["train"])
    path.write_bytes(path.read_bytes() + b"\n")
    report = preflight_dataset(session, tokenizer, 32)
    assert report["status"] == "blocked"
    assert report["rows"] == []
    assert report["issues"][-1]["code"] == "artifact_mismatch"


@pytest.mark.parametrize(
    "change", ["source_metadata", "missing_row", "test_as_validation", "wrong_loader"]
)
def test_valid_hash_does_not_hide_metadata_or_consumption_mismatch(exported, tokenizer, change):
    _, session = exported
    if change == "source_metadata":
        _republish(session, lambda rows, metadata: metadata.update(source_digest="wrong"))
    elif change == "missing_row":
        _republish(session, lambda rows, metadata: rows["train"].pop())
    elif change == "test_as_validation":
        session.dataset.data_config["validation_file"] = session.dataset.paths["test"]
    else:
        session.dataset.data_config["dataset_loader"] = "finance"
    report = preflight_dataset(session, tokenizer, 32)
    assert report["status"] == "blocked"
    assert report["rows"] == []


def test_hash_valid_but_cross_partition_group_leakage_is_blocked(exported, tokenizer):
    _, session = exported

    def leak(splits, _metadata):
        for source, records in splits.items():
            if any(row["metadata"]["source_row_id"] == "r000001" for row in records):
                index = next(
                    i
                    for i, row in enumerate(records)
                    if row["metadata"]["source_row_id"] == "r000001"
                )
                target = next(key for key in splits if key != source)
                splits[target].append(records.pop(index))
                return

    _republish(session, leak)
    report = preflight_dataset(session, tokenizer, 32)
    assert report["status"] == "blocked"
    assert any(issue["code"] == "cross_split_leakage" for issue in report["issues"])


def test_revision_invalidates_preflight_and_local_tokenizer_round_trip(
    exported, tokenizer, tmp_path
):
    service, session = exported
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    loaded = load_local_tokenizer(directory)
    assert preflight_dataset(session, loaded, 32)["status"] == "passed"
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    report = preflight_dataset(session, loaded, 32)
    assert report["status"] == "blocked"
    assert any(issue["code"] == "stale_dataset" for issue in report["issues"])
