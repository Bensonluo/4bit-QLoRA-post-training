"""Answer-only masks survive every training boundary and drive the actual CE loss."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from src.data.grpo_dataset import GRPODataset
from src.data.loaders import AlpacaDataset, FinanceDataset, PreferenceDataset
from src.data.medical_dataset import MedicalEntityDataset
from src.data.sft_collator import AttentionMaskCausalCollator
from src.data.sft_protocol import (
    ANSWER_PROTOCOL,
    LEGACY_PROTOCOL,
    NATIVE_PROTOCOL,
    tokenize_sft_record,
)
from src.training.domain_trainer import DomainAdaptationTrainer
from src.training.sft_trainer import SFTTrainer
from tests.unit.test_lora_ga_calibration import TinyCausalLoss
from tests.unit.test_sft_eos_contract import EXAMPLE, tokenizer


def test_data_protocol_defaults_preserve_legacy_and_reject_unpublished_native():
    assert DataConfig().sft_protocol == LEGACY_PROTOCOL
    assert DataConfig(sft_protocol=ANSWER_PROTOCOL).sft_protocol == ANSWER_PROTOCOL
    for invalid in (None, "", "answer_only", NATIVE_PROTOCOL, True):
        with pytest.raises(ValueError, match="sft_protocol"):
            DataConfig(sft_protocol=invalid)


@pytest.mark.parametrize(
    "ratio", [-0.1, 1.1, float("nan"), float("inf"), -float("inf"), True, False, "0", None, 1j]
)
def test_warmup_ratio_requires_finite_real_fraction(ratio):
    with pytest.raises(ValueError, match="warmup_ratio"):
        TrainingConfig(warmup_ratio=ratio)


@pytest.mark.parametrize(
    "ratio, expected_four, expected_hundred",
    [
        (0, [0.004, 0.003, 0.002], [0.004, 0.00396, 0.00392]),
        (0.03, [0, 0.004, 0.008 / 3], [0, 0.004 / 3, 0.008 / 3]),
        (0.5, [0, 0.002, 0.004], [0, 0.00008, 0.00016]),
        (1, [0, 0.001, 0.002], [0, 0.00004, 0.00008]),
    ],
)
def test_warmup_ratio_reaches_real_hf_scheduler_for_actual_step_count(
    tmp_path, ratio, expected_four, expected_hundred
):
    from transformers import Trainer

    trainer = SFTTrainer(
        model_config=ModelConfig(name="unused"),
        training_config=TrainingConfig(
            output_dir=str(tmp_path / "no-warmup"),
            bf16=False,
            warmup_ratio=ratio,
            learning_rate=0.004,
            lr_scheduler_type="linear",
        ),
        lora_config=LoRAConfig(),
        data_config=DataConfig(),
        logging_config=LoggingConfig(),
    )
    trainer.model = TinyCausalLoss(24, 12)
    trainer.tokenizer = tokenizer()
    trainer.train_dataset = [EXAMPLE]
    trainer.eval_dataset = None
    with patch("src.training.sft_trainer.Trainer") as hf_trainer:
        trainer.setup_trainer()
    arguments = hf_trainer.call_args.kwargs["args"]
    # Exercise HF's actual scheduler entry point without loading another model.
    # Rebuilding with a changed total also covers batch-size retry schedules.
    hf_trainer = Trainer.__new__(Trainer)
    hf_trainer.args = arguments
    for total, expected in [(4, expected_four), (100, expected_hundred)]:
        weight = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([weight], lr=arguments.learning_rate)
        hf_trainer.lr_scheduler = None
        scheduler = hf_trainer.create_scheduler(total, optimizer=optimizer)
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(2):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        assert rates == pytest.approx(expected)


@pytest.mark.parametrize(
    "fields",
    [
        {"sft_warmup_ratio": -0.1},
        {"sft_warmup_ratio": 1.1},
        {"sft_warmup_ratio": float("nan")},
        {"sft_warmup_ratio": float("inf")},
        {"sft_warmup_ratio": True},
        {"sft_warmup_ratio": None},
        {"sft_warmup_ratio": "0.5"},
        {"warmup_steps": 1},
        {"warmup_steps": 0.5},
    ],
)
def test_sft_training_arguments_reject_invalid_or_ambiguous_warmup(tmp_path, fields):
    from src.training.sft_trainer import _SFTRatioTrainingArguments

    with pytest.raises(ValueError, match="warmup"):
        _SFTRatioTrainingArguments(output_dir=str(tmp_path), bf16=False, **fields)


def test_data_config_does_not_import_training_dependencies():
    # Import this module directly: config.__init__ historically constructs
    # DPO's default ModelConfig, whose platform detection can import torch.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, importlib.util; "
            "spec = importlib.util.spec_from_file_location('isolated_config', 'config/base.py'); "
            "module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; "
            "spec.loader.exec_module(module); module.DataConfig(); "
            "assert 'torch' not in sys.modules; assert 'src.data' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("format", ["chat", "sharegpt", "dpo", "grpo"])
def test_answer_protocol_requires_alpaca_records(format):
    assert DataConfig(format=format).sft_protocol == LEGACY_PROTOCOL
    with pytest.raises(ValueError, match="alpaca"):
        DataConfig(format=format, sft_protocol=ANSWER_PROTOCOL)


@pytest.mark.parametrize("dataset_class", [PreferenceDataset, GRPODataset])
def test_preference_and_grpo_formatters_reject_sft_masks_before_loading(dataset_class):
    source = dataset_class("unused")
    with pytest.raises(ValueError, match="protocols"):
        source.format_for_training(tokenizer(), sft_protocol=ANSWER_PROTOCOL)
    assert source.dataset is None


@pytest.mark.parametrize("dataset_class", [AlpacaDataset, FinanceDataset, MedicalEntityDataset])
def test_formatters_return_proven_answer_and_eos_labels(dataset_class):
    tok = tokenizer()
    source = dataset_class("unused")
    source.dataset = Dataset.from_list([EXAMPLE])
    expected = tokenize_sft_record(EXAMPLE, tok, 24, protocol=ANSWER_PROTOCOL)
    actual = dict(source.format_for_training(tok, 24, sft_protocol=ANSWER_PROTOCOL)[0])
    assert actual == expected
    assert [label for label in actual["labels"] if label != -100] == [
        tok.convert_tokens_to_ids("answer"),
        tok.eos_token_id,
    ]


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("pad_equals_eos", [True, False])
def test_answer_mask_actual_ce_gradients_exclude_prompt_and_include_eos(side, pad_equals_eos):
    tok = tokenizer(side=side, pad_equals_eos=pad_equals_eos)
    row = tokenize_sft_record(EXAMPLE, tok, 21, protocol=ANSWER_PROTOCOL)
    before = deepcopy(row)
    batch = AttentionMaskCausalCollator(tok, sft_protocol=ANSWER_PROTOCOL)([row])
    assert row == before
    labels = batch["labels"][0]
    assert torch.all(labels[batch["attention_mask"][0] == 0] == -100)
    active = batch["attention_mask"][0].bool()
    assert labels[active].tolist()[:-2] == [-100] * (int(active.sum()) - 2)
    assert labels[active].tolist()[-2:] == [tok.convert_tokens_to_ids("answer"), tok.eos_token_id]
    model = TinyCausalLoss(24, len(tok))
    loss = model(**batch).loss
    loss.backward()
    assert torch.isfinite(loss)
    actual_gradient = torch.count_nonzero(model.logits.grad[:-1], dim=-1) > 0
    assert torch.equal(actual_gradient, labels[1:] != -100)
    eos_position = ((batch["input_ids"][0] == tok.eos_token_id) & active).nonzero().item()
    assert actual_gradient[eos_position - 1]
    # Mutating masked labels would change the loss: the real model's CE is
    # driven by exactly two targets, rather than a merely recorded mask.
    logits = model.logits.detach().clone()
    logits[eos_position - 1, tok.eos_token_id] = 4
    with torch.no_grad():
        model.logits.copy_(logits)
    assert model(**batch).loss < loss


@pytest.mark.parametrize("side", ["left", "right"])
def test_dynamic_batch_padding_keeps_each_records_answer_mask(side):
    tok = tokenizer(side=side)
    rows = [
        tokenize_sft_record(row, tok, protocol=ANSWER_PROTOCOL)
        for row in [EXAMPLE, EXAMPLE | {"output": "answer answer"}]
    ]
    before = deepcopy(rows)
    batch = AttentionMaskCausalCollator(tok, sft_protocol=ANSWER_PROTOCOL)(rows)
    assert rows == before
    for original, mask, labels in zip(rows, batch["attention_mask"], batch["labels"]):
        assert labels[mask == 1].tolist() == original["labels"]
        assert torch.all(labels[mask == 0] == -100)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "ragged",
        "float",
        "bool",
        "mismatch",
        "padding",
        "no_answer",
        "no_eos",
        "causal_first",
        "label_gap",
        "no_prompt",
        "nonterminal_eos",
    ],
)
def test_answer_collator_rejects_invalid_or_empty_supervision(mutation):
    tok = tokenizer()
    row = tokenize_sft_record(EXAMPLE, tok, 24, protocol=ANSWER_PROTOCOL)
    first = next(i for i, label in enumerate(row["labels"]) if label != -100)
    eos = first + 1
    if mutation == "missing":
        del row["labels"]
    elif mutation == "ragged":
        row["labels"].pop()
    elif mutation in {"float", "bool", "mismatch"}:
        row["labels"][first] = {"float": 11.0, "bool": True, "mismatch": 999}[mutation]
    elif mutation == "padding":
        row["labels"][-1] = tok.pad_token_id
    elif mutation == "no_answer":
        row["labels"][first] = -100
    elif mutation == "no_eos":
        row["labels"][eos] = -100
    elif mutation == "causal_first":
        row = {key: values[first:] for key, values in row.items()}
    elif mutation == "label_gap":
        row["labels"][first - 2] = row["input_ids"][first - 2]
    elif mutation == "no_prompt":
        row["labels"] = [
            token if mask else -100 for token, mask in zip(row["input_ids"], row["attention_mask"])
        ]
    elif mutation == "nonterminal_eos":
        row["input_ids"][eos + 1] = tok.convert_tokens_to_ids("question")
        row["attention_mask"][eos + 1] = 1
    with pytest.raises(ValueError):
        AttentionMaskCausalCollator(tok, sft_protocol=ANSWER_PROTOCOL)([row])


@pytest.mark.parametrize("trainer_class", [SFTTrainer, DomainAdaptationTrainer])
@pytest.mark.parametrize("loader", ["alpaca", "medical_entity"])
def test_train_validation_ga_and_formal_trainer_consume_identical_answer_masks(
    tmp_path, monkeypatch, trainer_class, loader
):
    records = [EXAMPLE, EXAMPLE]
    source = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    source.write_text(json.dumps(records))
    validation.write_text(json.dumps([EXAMPLE]))
    monkeypatch.setattr(
        "src.data.loaders.load_dataset",
        lambda path, **kwargs: Dataset.from_list(json.loads(Path(path).read_text())),
    )
    trainer = trainer_class(
        model_config=ModelConfig(name="unused", max_length=24),
        training_config=TrainingConfig(output_dir=str(tmp_path / "out"), batch_size=1, bf16=False),
        lora_config=LoRAConfig(lora_ga_calibration_batches=1),
        data_config=DataConfig(
            dataset_name=str(source),
            dataset_loader=loader,
            validation_split=0,
            validation_file=str(validation),
            sft_protocol=ANSWER_PROTOCOL,
        ),
        logging_config=LoggingConfig(),
    )
    tok = tokenizer()
    trainer.tokenizer = tok
    trainer.model = TinyCausalLoss(24, len(tok))
    trainer.prepare_data()
    expected = tokenize_sft_record(EXAMPLE, tok, 24, protocol=ANSWER_PROTOCOL)
    assert dict(trainer.train_dataset[0]) == dict(trainer.eval_dataset[0]) == expected
    assert len(trainer.train_dataset) == 2 and len(trainer.eval_dataset) == 1
    with patch("src.training.sft_trainer.Trainer") as hf_trainer:
        trainer.setup_trainer()
    formal_collator = hf_trainer.call_args.kwargs["data_collator"]
    formal_batch = formal_collator([trainer.train_dataset[0]])
    expected_model = TinyCausalLoss(24, len(tok))
    expected_model(**formal_batch).loss.backward()
    trainer._build_lora_ga_train_step()()
    actual_batch = trainer.model.batches[0]
    assert all(torch.equal(actual_batch[key], value) for key, value in formal_batch.items())
    assert torch.equal(trainer.model.logits.grad, expected_model.logits.grad)


def test_answer_formatter_never_silently_truncates_or_falls_back():
    tok = tokenizer()
    source = AlpacaDataset("unused")
    source.dataset = Dataset.from_list([EXAMPLE])
    with pytest.raises(ValueError, match="max_length"):
        source.format_for_training(tok, 2, sft_protocol=ANSWER_PROTOCOL)
    with pytest.raises(ValueError, match="protocol"):
        source.format_for_training(tok, 24, sft_protocol=None)
