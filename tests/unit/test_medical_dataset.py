"""Unit tests for the medical entity dataset (src/data/medical_dataset.py).

Uses real `datasets.Dataset` objects (offline) with tiny local JSON fixtures;
the tokenizer is a lightweight double.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.data.medical_dataset import MedicalEntityDataset


def _write_fixture(tmp_path: Any) -> str:
    rows = [
        {
            "instruction": "从候选列表中选出匹配的标准名称",
            "input": f"输入实体: 药品{i}\n候选:\n1. ...",
            "output": '{"standard_name": "x"}',
            "metadata": {"entity_type": "drug", "difficulty": "easy"},
        }
        for i in range(2)
    ] + [
        {
            "instruction": "从候选列表中选出匹配的标准名称",
            "input": "输入实体: 医院\n候选:\n1. ...",
            "output": '{"standard_name": "y"}',
            "metadata": {"entity_type": "hospital", "difficulty": "hard"},
        }
    ]
    path = tmp_path / "medical.json"
    path.write_text(json.dumps(rows, ensure_ascii=False))
    return str(path)


class _FakeTokenizer:
    """Batch-callable tokenizer double returning fixed-width id lists."""

    def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": [[7, 8, 9] for _ in prompts],
            "attention_mask": [[1, 1, 1] for _ in prompts],
        }


class TestMedicalEntityDataset:
    def test_load_returns_all_rows(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        dataset = ds.load()
        assert dataset.num_rows == 3

    def test_difficulty_filter_keeps_only_matching(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path), difficulty_filter="hard")
        dataset = ds.load()
        assert dataset.num_rows == 1
        assert dataset[0]["metadata"]["difficulty"] == "hard"

    def test_max_samples_applies_after_filter(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path), max_samples=2)
        assert ds.load().num_rows == 2

    def test_filter_then_max_samples_combined(self, tmp_path: Any) -> None:
        # easy rows are 2; filter easy + max 1 → 1 (not 3-wide unfiltered slice)
        ds = MedicalEntityDataset(_write_fixture(tmp_path), difficulty_filter="easy", max_samples=1)
        assert ds.load().num_rows == 1
        assert ds.load()[0]["metadata"]["difficulty"] == "easy"

    def test_missing_file_raises(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(str(tmp_path / "nope.json"))
        with pytest.raises(FileNotFoundError, match="数据文件不存在"):
            ds.load()

    def test_format_for_training_renders_alpaca_template(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        ds.load()
        formatted = ds.format_for_training(_FakeTokenizer(), max_length=64)

        assert formatted.num_rows == 3
        assert "input_ids" in formatted.column_names
        # original text columns are removed after tokenization
        assert "instruction" not in formatted.column_names
        assert formatted[0]["input_ids"] == [7, 8, 9]

    def test_format_prompts_contain_all_three_sections(self, tmp_path: Any) -> None:
        seen_prompts: list[list[str]] = []

        class _CaptureTokenizer(_FakeTokenizer):
            def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, Any]:
                seen_prompts.append(prompts)
                return super().__call__(prompts, **kwargs)

        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        ds.load()
        ds.format_for_training(_CaptureTokenizer(), max_length=64)

        first_prompt = seen_prompts[0][0]
        assert "### Instruction:" in first_prompt
        assert "### Input:" in first_prompt
        assert "### Response:" in first_prompt

    def test_format_loads_implicitly_when_not_loaded(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        formatted = ds.format_for_training(_FakeTokenizer(), max_length=64)
        assert formatted.num_rows == 3


class TestBaseDatasetBehaviors:
    def test_split_dataset_loads_implicitly(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        assert ds.dataset is None
        train, val = ds.split_dataset(validation_split=0.5, seed=42)
        assert train.num_rows + val.num_rows == 3

    def test_split_sizes_match_fraction(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        ds.load()
        train, val = ds.split_dataset(validation_split=1 / 3, seed=42)
        assert train.num_rows == 2 and val.num_rows == 1

    def test_zero_validation_split_returns_full_dataset(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        ds.load()
        train, val = ds.split_dataset(validation_split=0)
        assert train.num_rows == 3
        assert val is None

    def test_split_is_seed_reproducible(self, tmp_path: Any) -> None:
        ds_a = MedicalEntityDataset(_write_fixture(tmp_path))
        ds_a.load()
        ds_b = MedicalEntityDataset(_write_fixture(tmp_path))
        ds_b.load()
        train_a, _ = ds_a.split_dataset(validation_split=0.5, seed=7)
        train_b, _ = ds_b.split_dataset(validation_split=0.5, seed=7)
        assert train_a["input"] == train_b["input"]

    def test_len_is_zero_before_load(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset(_write_fixture(tmp_path))
        assert len(ds) == 0
        ds.load()
        assert len(ds) == 3

    def test_repr_carries_class_and_path(self, tmp_path: Any) -> None:
        ds = MedicalEntityDataset("some/path.json", max_samples=10)
        text = repr(ds)
        assert "MedicalEntityDataset" in text and "some/path.json" in text
