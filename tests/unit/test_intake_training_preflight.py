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


class NoOffsets:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, prompt, **options):
        if options.pop("return_offsets_mapping", False):
            raise NotImplementedError("slow tokenizer")
        return self.tokenizer(prompt, **options)


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
    assert all(row["terminal_eos_supervised"] for row in report["rows"])


@pytest.mark.parametrize("auto_eos", [False, True])
@pytest.mark.parametrize("shared_pad_eos", [False, True])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_only_terminal_eos_truncated_blocks_even_when_every_answer_is_complete(
    exported, tokenizer, auto_eos, shared_pad_eos, with_offsets
):
    _, session = exported
    if not auto_eos:
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", tokenizer.bos_token_id)]
        )
    if shared_pad_eos:
        tokenizer.pad_token = tokenizer.eos_token
    complete = preflight_dataset(session, tokenizer, 64)
    assert complete["status"] == "passed"
    lengths = {row["full_tokens"] for row in complete["rows"]}
    assert len(lengths) == 1

    report = preflight_dataset(
        session, tokenizer if with_offsets else NoOffsets(tokenizer), lengths.pop() - 1
    )
    assert report["status"] == "blocked"
    assert not any(row["terminal_eos_supervised"] for row in report["rows"])
    if with_offsets:
        assert all(
            row["answer_tokens"] == row["answer_supervised_tokens"] == 1 for row in report["rows"]
        )
        assert not any(row["answer_was_truncated"] for row in report["rows"])
    else:
        assert not any(row["answer_boundary_verified"] for row in report["rows"])
    lost = [issue for issue in report["issues"] if issue["code"] == "terminal_eos_lost"]
    assert len(lost) == len(report["rows"])
    assert all(issue["severity"] == "blocking" and issue["row_ids"] for issue in lost)


@pytest.mark.parametrize("with_offsets", [False, True])
@pytest.mark.parametrize("answer,cut", [("yes [EOS] yes", 2), ("yes [EOS]", 1)])
def test_truncation_ending_on_interior_eos_does_not_claim_original_ending_survived(
    tmp_path, tokenizer, with_offsets, answer, cut
):
    service = IntakeService(tmp_path / "intake")
    session = _full(service, data=FULL.replace(b",yes\n", f",{answer}\n".encode()))
    session = service.materialize_dataset(session.session_id, session.revision)
    complete = preflight_dataset(session, tokenizer, 64)
    length = complete["rows"][0]["full_tokens"] - cut
    from src.data.loaders import tokenize_alpaca_record

    example = {"instruction": "分类", "input": "内容: a", "output": answer}
    actual = tokenize_alpaca_record(example, tokenizer, length)
    assert actual["input_ids"][-1] == tokenizer.eos_token_id
    report = preflight_dataset(session, tokenizer if with_offsets else NoOffsets(tokenizer), length)
    assert report["status"] == "blocked"
    if with_offsets:
        assert all(row["answer_supervised_tokens"] > 0 for row in report["rows"])
    assert not any(row["terminal_eos_supervised"] for row in report["rows"])
    assert any(issue["code"] == "terminal_eos_lost" for issue in report["issues"])


@pytest.mark.parametrize("with_offsets", [False, True])
def test_terminal_eos_at_causal_first_position_is_not_supervised(exported, tokenizer, with_offsets):
    tokenizer.truncation_side = "left"
    report = preflight_dataset(exported[1], tokenizer if with_offsets else NoOffsets(tokenizer), 1)
    assert report["status"] == "blocked"
    assert not any(row["terminal_eos_supervised"] for row in report["rows"])
    assert any(issue["code"] == "terminal_eos_lost" for issue in report["issues"])


@pytest.mark.parametrize("with_offsets", [False, True])
def test_eos_only_after_left_padding_still_has_loss_but_lacks_real_causal_context(
    exported, tokenizer, with_offsets
):
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    report = preflight_dataset(exported[1], tokenizer if with_offsets else NoOffsets(tokenizer), 1)
    assert report["status"] == "blocked"
    # The collator really computes PAD -> EOS loss. Keep that accounting honest
    # while refusing a record without any real predecessor/context token.
    assert all(row["supervised_tokens"] == 1 for row in report["rows"])
    assert all(row["terminal_eos_supervised"] is True for row in report["rows"])
    assert all(row["kept_tokens"] == 1 for row in report["rows"])
    assert any(issue["code"] == "missing_causal_context" for issue in report["issues"])


def test_partial_answer_truncation_is_reported_without_claiming_complete_coverage(
    tmp_path, tokenizer
):
    service = IntakeService(tmp_path / "intake")
    session = _full(service, data=FULL.replace(b",yes\n", b",yes yes yes yes yes\n"))
    session = service.materialize_dataset(session.session_id, session.revision)
    complete = preflight_dataset(session, tokenizer, 64)
    length = complete["rows"][0]["input_tokens"] + 2
    report = preflight_dataset(session, tokenizer, length)
    assert report["status"] == "blocked"
    assert all(0 < row["answer_kept_tokens"] < row["answer_tokens"] for row in report["rows"])
    assert all(row["answer_was_truncated"] for row in report["rows"])
    assert any(issue["code"] == "terminal_eos_lost" for issue in report["issues"])
    assert any(issue["code"] == "answer_partial" for issue in report["issues"])


@pytest.mark.parametrize("with_offsets", [False, True])
def test_left_truncation_with_partial_answer_keeps_eos_but_cannot_claim_complete_target(
    tmp_path, tokenizer, with_offsets
):
    service = IntakeService(tmp_path / "intake")
    session = _full(service, data=FULL.replace(b",yes\n", b",yes yes yes yes yes\n"))
    session = service.materialize_dataset(session.session_id, session.revision)
    tokenizer.truncation_side = "left"
    report = preflight_dataset(session, tokenizer if with_offsets else NoOffsets(tokenizer), 3)
    assert all(row["terminal_eos_supervised"] for row in report["rows"])
    assert not any(issue["code"] == "terminal_eos_lost" for issue in report["issues"])
    if with_offsets:
        assert report["status"] == "blocked"
        assert all(0 < row["answer_kept_tokens"] < row["answer_tokens"] for row in report["rows"])
        assert any(issue["code"] == "answer_partial" for issue in report["issues"])
    else:
        assert report["status"] == "warnings"
        assert all(row["answer_kept_tokens"] is None for row in report["rows"])
        assert not any(issue["code"] == "answer_partial" for issue in report["issues"])
        assert any(issue["code"] == "answer_offsets_unavailable" for issue in report["issues"])


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("with_offsets", [False, True])
def test_complete_visible_answer_must_also_survive_actual_causal_supervision(
    tmp_path, tokenizer, padding_side, with_offsets
):
    from src.data.loaders import tokenize_alpaca_record
    from src.data.sft_collator import AttentionMaskCausalCollator

    service = IntakeService(tmp_path / "intake")
    session = _full(service, data=FULL.replace(b",yes\n", b",yes yes\n"))
    session = service.materialize_dataset(session.session_id, session.revision)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = padding_side
    example = {"instruction": "分类", "input": "内容: a", "output": "yes yes"}
    actual = tokenize_alpaca_record(example, tokenizer, 3)
    assert actual["input_ids"] == [4, 4, tokenizer.eos_token_id]
    batch = AttentionMaskCausalCollator(tokenizer)([actual])
    actual_answer_labels = (batch["labels"][0, 1:] == 4).sum().item()
    assert actual_answer_labels == (2 if padding_side == "left" else 1)
    report = preflight_dataset(session, tokenizer if with_offsets else NoOffsets(tokenizer), 3)
    assert all(row["terminal_eos_supervised"] for row in report["rows"])
    if with_offsets:
        assert all(row["answer_kept_tokens"] == row["answer_tokens"] == 2 for row in report["rows"])
        assert not any(row["answer_was_truncated"] for row in report["rows"])
        assert all(
            row["answer_supervised_tokens"] == actual_answer_labels for row in report["rows"]
        )
        complete = padding_side == "left"
        assert report["status"] == ("warnings" if complete else "blocked")
        assert all(row["answer_supervision_complete"] is complete for row in report["rows"])
        assert any(issue["code"] == "answer_partial" for issue in report["issues"]) is not complete
    else:
        assert report["status"] == "warnings"
        assert all(row["answer_supervision_complete"] is None for row in report["rows"])
        assert not any(issue["code"] == "answer_partial" for issue in report["issues"])


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
    _, session = exported
    report = preflight_dataset(session, NoOffsets(tokenizer), 32)
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
