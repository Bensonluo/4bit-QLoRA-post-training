"""SFT must pass the selected, formatted partitions to the HF Trainer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from src.training.domain_trainer import DomainAdaptationTrainer
from src.training.sft_trainer import SFTTrainer


class RowTokenizer:
    """Encode an input's row identity so partition membership survives formatting."""

    eos_token_id = 9000
    pad_token_id = 0

    def __call__(self, prompts: str | list[str], **_kwargs: object) -> dict:
        def encode(prompt: str) -> list[int]:
            match = re.search(r"row-(\d+)", prompt)
            assert match is not None
            return [int(match.group(1))]

        if isinstance(prompts, str):
            return {"input_ids": encode(prompts), "attention_mask": [1]}
        return {
            "input_ids": [encode(prompt) for prompt in prompts],
            "attention_mask": [[1] for _ in prompts],
        }


def _write_rows(path: Path, identities: range) -> None:
    path.write_text(
        json.dumps(
            [
                {"instruction": "Classify this stock record", "input": f"row-{i}", "output": "yes"}
                for i in identities
            ]
        )
    )


@pytest.fixture
def local_dataset_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the production loaders, Dataset splitting, and formatters while
    # keeping HF dataset discovery/downloads out of this regression test.
    def load_local(path: str, **_kwargs: object) -> Dataset:
        return Dataset.from_list(json.loads(Path(path).read_text()))

    monkeypatch.setattr("src.data.loaders.load_dataset", load_local)


def _trainer(
    tmp_path: Path, source: Path, domain_name: str | None = None, **data_options: object
) -> SFTTrainer:
    config = dict(
        model_config=ModelConfig(name="unused-local-test-model"),
        training_config=TrainingConfig(output_dir=str(tmp_path / "output"), bf16=False),
        lora_config=LoRAConfig(),
        data_config=DataConfig(dataset_name=str(source), max_samples=None, **data_options),
        logging_config=LoggingConfig(),
    )
    trainer = (
        SFTTrainer(**config)
        if domain_name is None
        else DomainAdaptationTrainer(**config, domain_name=domain_name)
    )
    trainer.tokenizer = RowTokenizer()
    trainer.model = MagicMock()
    return trainer


def _trainer_inputs(trainer: SFTTrainer) -> dict:
    # Inspect the final Trainer boundary, not just split_dataset's return value.
    with (
        patch("src.training.sft_trainer.Trainer") as hf_trainer,
        patch("src.training.sft_trainer.AttentionMaskCausalCollator"),
    ):
        trainer.setup_trainer()
        return hf_trainer.call_args.kwargs


def _identities(dataset: Dataset) -> set[int]:
    return {row["input_ids"][0] for row in dataset}


@pytest.mark.parametrize(
    "dataset_type,domain_name",
    [("alpaca", None), ("medical_entity", None), ("alpaca", "legal"), ("finance", "finance")],
)
def test_final_trainer_receives_disjoint_formatted_splits(
    tmp_path: Path, local_dataset_loader: None, dataset_type: str, domain_name: str | None
) -> None:
    source = tmp_path / f"{dataset_type}.json"
    _write_rows(source, range(20))
    trainer = _trainer(tmp_path, source, domain_name=domain_name, validation_split=0.25)

    trainer.prepare_data()
    inputs = _trainer_inputs(trainer)

    train = inputs["train_dataset"]
    validation = inputs["eval_dataset"]
    assert len(train) == 15
    assert len(validation) == 5
    assert _identities(train).isdisjoint(_identities(validation))
    assert _identities(train) | _identities(validation) == set(range(20))
    assert "instruction" not in train.column_names
    assert "instruction" not in validation.column_names


@pytest.mark.parametrize(
    "dataset_type,domain_name",
    [("alpaca", None), ("medical_entity", None), ("alpaca", "legal"), ("finance", "finance")],
)
def test_explicit_validation_file_is_formatted_and_kept_separate(
    tmp_path: Path, local_dataset_loader: None, dataset_type: str, domain_name: str | None
) -> None:
    source = tmp_path / f"{dataset_type}.json"
    validation_source = tmp_path / "validation.json"
    _write_rows(source, range(10))
    _write_rows(validation_source, range(100, 104))
    trainer = _trainer(
        tmp_path,
        source,
        domain_name=domain_name,
        validation_split=0,
        validation_file=str(validation_source),
    )

    trainer.prepare_data()
    inputs = _trainer_inputs(trainer)

    assert _identities(inputs["train_dataset"]) == set(range(10))
    assert _identities(inputs["eval_dataset"]) == set(range(100, 104))


@pytest.mark.parametrize("domain_name", [None, "legal", "finance"])
def test_no_validation_keeps_all_source_rows_in_train(
    tmp_path: Path, local_dataset_loader: None, domain_name: str | None
) -> None:
    source = tmp_path / "alpaca.json"
    _write_rows(source, range(10))
    trainer = _trainer(tmp_path, source, domain_name=domain_name, validation_split=0)

    trainer.prepare_data()
    inputs = _trainer_inputs(trainer)

    assert _identities(inputs["train_dataset"]) == set(range(10))
    assert inputs["eval_dataset"] is None


@pytest.mark.parametrize("domain_name", [None, "finance"])
def test_materialized_finance_named_path_keeps_every_approved_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain_name: str | None
) -> None:
    import datasets.config
    import huggingface_hub.constants
    import torch

    from src.workbench.intake_service import IntakeService
    from tests.unit.test_data_materialize import _full

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf" / "datasets")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    service = IntakeService(tmp_path / "intake")
    full = "customer,order,text,label\n" + "".join(f"c{i},o{i},row-{i},yes\n" for i in range(20))
    session = _full(service, data=full.encode())
    session = service.materialize_dataset(
        session.session_id, session.revision, name="finance-customer-data"
    )
    artifact = session.dataset
    trainer = _trainer(tmp_path, Path(artifact.paths["train"]), domain_name=domain_name)
    trainer.data_config = DataConfig(**artifact.data_config)
    # No mocked load_dataset: these are the actual exported local JSONL files.
    trainer.prepare_data()
    inputs = _trainer_inputs(trainer)

    def expected_ids(split: str) -> set[int]:
        return {
            int(re.search(r"row-(\d+)", json.loads(line)["input"]).group(1))
            for line in Path(artifact.paths[split]).read_text().splitlines()
        }

    assert _identities(inputs["train_dataset"]) == expected_ids("train")
    assert _identities(inputs["eval_dataset"]) == expected_ids("validation")
    assert _identities(inputs["train_dataset"]).isdisjoint(expected_ids("test"))
    assert _identities(inputs["eval_dataset"]).isdisjoint(expected_ids("test"))
    assert len(inputs["train_dataset"]) == artifact.statistics["row_counts"]["train"]

    class CalibrationTokenizer(RowTokenizer):
        seen: list[str] = []

        def __call__(self, prompts, **kwargs):
            self.seen.append(prompts)
            return super().__call__(prompts, **kwargs)

    trainer.tokenizer = CalibrationTokenizer()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.lora_config.lora_ga_calibration_batches = 100
    callback = trainer._build_lora_ga_train_step()
    assert callable(callback)
    assert {
        int(re.search(r"row-(\d+)", prompt).group(1)) for prompt in trainer.tokenizer.seen
    } == expected_ids("train")


def test_explicit_loader_validation_and_legacy_selection() -> None:
    from src.data import AlpacaDataset, FinanceDataset
    from src.data.medical_dataset import MedicalEntityDataset
    from src.training.sft_trainer import _dataset_class_for

    assert DataConfig().dataset_loader is None
    with pytest.raises(ValueError, match="dataset_loader"):
        DataConfig(dataset_loader="invented")
    assert _dataset_class_for("finance/path") is FinanceDataset
    assert _dataset_class_for("medical_entity/path") is MedicalEntityDataset
    assert _dataset_class_for("finance/path", "alpaca") is AlpacaDataset
    assert _dataset_class_for("anything", "finance") is FinanceDataset
    assert _dataset_class_for("anything", "medical_entity") is MedicalEntityDataset
