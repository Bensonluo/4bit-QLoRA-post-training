"""LoRA-GA estimates the same masked training objective on the actual train partition."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset

from src.data.loaders import AlpacaDataset
from src.data.sft_collator import AttentionMaskCausalCollator
from src.training.sft_trainer import SFTTrainer
from tests.unit.test_sft_eos_contract import EXAMPLE, tokenizer


class TinyCausalLoss(torch.nn.Module):
    def __init__(self, length, vocabulary):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(length, vocabulary))
        self.batches = []

    def forward(self, input_ids, attention_mask, labels):
        self.batches.append(
            {
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "labels": labels.detach().clone(),
            }
        )
        scores = self.logits[: input_ids.shape[1]].unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        return SimpleNamespace(
            loss=F.cross_entropy(
                scores[:, :-1].reshape(-1, scores.shape[-1]), labels[:, 1:].reshape(-1)
            )
        )


def calibration_trainer(
    tokenizer, model, *, batch_size=1, batches=1, max_samples=None, validation_split=0
):
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.tokenizer, trainer.model = tokenizer, model
    trainer.data_config = SimpleNamespace(
        dataset_name="local-calibration",
        dataset_loader="alpaca",
        max_samples=max_samples,
        validation_split=validation_split,
    )
    trainer.lora_config = SimpleNamespace(lora_ga_calibration_batches=batches)
    trainer.training_config = SimpleNamespace(batch_size=batch_size, seed=42)
    trainer.model_config = SimpleNamespace(max_length=24)
    return trainer


@pytest.mark.parametrize("pad_equals_eos", [True, False])
@pytest.mark.parametrize("side", ["left", "right"])
def test_calibration_ce_gradients_mask_padding_but_preserve_eos(monkeypatch, pad_equals_eos, side):
    tok = tokenizer(pad_equals_eos=pad_equals_eos, side=side)
    monkeypatch.setattr(
        "src.data.loaders.load_dataset", lambda *args, **kwargs: Dataset.from_list([EXAMPLE])
    )
    actual = TinyCausalLoss(24, len(tok))
    trainer = calibration_trainer(tok, actual)
    trainer._build_lora_ga_train_step()()

    source = AlpacaDataset("local-calibration")
    source.dataset = Dataset.from_list([EXAMPLE])
    rows = list(source.format_for_training(tok, max_length=24))
    expected_batch = AttentionMaskCausalCollator(tok)(rows)
    expected = TinyCausalLoss(24, len(tok))
    expected(**expected_batch).loss.backward()
    batch = actual.batches[0]
    assert torch.equal(batch["labels"], expected_batch["labels"])
    assert torch.equal(actual.logits.grad, expected.logits.grad)
    padding = batch["attention_mask"] == 0
    assert padding.any() and torch.all(batch["labels"][padding] == -100)
    real_eos = (batch["input_ids"] == tok.eos_token_id) & ~padding
    assert real_eos.sum() == 1
    assert torch.all(batch["labels"][real_eos] == tok.eos_token_id)
    assert torch.count_nonzero(actual.logits.grad[real_eos[0].nonzero().item() - 1]) > 0


@pytest.mark.parametrize("calibration_batches", [2, 3, 10])
def test_calibration_respects_actual_train_split_source_limit_and_partial_batches(
    monkeypatch, tmp_path, calibration_batches
):
    from tests.unit.test_sft_data_partitions import RowTokenizer

    class PaddingRowTokenizer(RowTokenizer):
        def pad(self, features, **kwargs):
            return {
                key: torch.tensor([row[key] for row in features])
                for key in ("input_ids", "attention_mask")
            }

    class RecordingLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.identities, self.batch_sizes = [], []

        def forward(self, input_ids, attention_mask, labels):
            self.identities.extend(input_ids[:, 0].tolist())
            self.batch_sizes.append(len(input_ids))
            return SimpleNamespace(loss=self.weight * input_ids.sum())

    records = [{"instruction": "classify", "input": f"row-{i}", "output": "yes"} for i in range(20)]
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps(records))
    monkeypatch.setattr(
        "src.data.loaders.load_dataset", lambda *args, **kwargs: Dataset.from_list(records)
    )
    model = RecordingLoss()
    trainer = calibration_trainer(
        PaddingRowTokenizer(),
        model,
        batch_size=3,
        batches=calibration_batches,
        max_samples=10,
        validation_split=0.25,
    )
    trainer.data_config.dataset_name = str(source)
    trainer.data_config.validation_file = None
    trainer.model_config.max_length = 8
    trainer.prepare_data()
    train_ids = [row["input_ids"][0] for row in trainer.train_dataset]
    heldout_ids = {row["input_ids"][0] for row in trainer.eval_dataset}
    assert len(train_ids) == 7 and len(heldout_ids) == 3
    assert set(train_ids) | heldout_ids == set(range(10))

    trainer._build_lora_ga_train_step()()
    expected = train_ids[: calibration_batches * 3]
    assert model.identities == expected
    assert set(model.identities).isdisjoint(heldout_ids)
    assert all(identity < 10 for identity in model.identities)
    assert len(model.batch_sizes) <= calibration_batches
    assert model.batch_sizes == ([3, 3] if calibration_batches == 2 else [3, 3, 1])
