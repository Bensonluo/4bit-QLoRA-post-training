"""Unit tests for GRPO dataset loader."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.grpo_dataset import GRPODataset


class TestGRPODataset:
    def test_load_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            examples = [
                {"prompt": "What is 2+2?", "answer": "4"},
                {"prompt": "What is 3+3?", "answer": "6"},
            ]
            with open(data_file, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            ds = GRPODataset(data_path=str(data_file))
            ds.load()
            assert len(ds.dataset) == 2
            assert ds.dataset["prompt"][0] == "What is 2+2?"
            assert ds.dataset["answer"][0] == "4"

    def test_custom_column_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            with open(data_file, "w") as f:
                f.write(json.dumps({"question": "Q", "ref": "A"}) + "\n")

            ds = GRPODataset(
                data_path=str(data_file),
                prompt_key="question",
                reference_key="ref",
            )
            ds.load()
            assert "prompt" in ds.dataset.column_names
            assert "reference" in ds.dataset.column_names

    def test_missing_prompt_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            with open(data_file, "w") as f:
                f.write(json.dumps({"answer": "4"}) + "\n")

            ds = GRPODataset(data_path=str(data_file))
            with pytest.raises(ValueError, match="must contain a 'prompt' column"):
                ds.load()

    def test_max_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            with open(data_file, "w") as f:
                for i in range(10):
                    f.write(json.dumps({"prompt": f"p{i}", "answer": str(i)}) + "\n")

            ds = GRPODataset(data_path=str(data_file), max_samples=3)
            ds.load()
            assert len(ds.dataset) == 3


class TestGRPODatasetLoadFailure:
    @patch("src.data.grpo_dataset.load_dataset", side_effect=Exception("HF down"))
    def test_double_failure_raises_runtime_error(self, _mock_load: MagicMock) -> None:
        ds = GRPODataset(data_path="nonexistent/path.jsonl")
        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            ds.load()


class TestGRPODatasetRenameAndNormalize:
    def test_answer_key_renamed_to_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            with open(data_file, "w") as f:
                f.write(json.dumps({"prompt": "Q", "gold": "A"}) + "\n")

            ds = GRPODataset(data_path=str(data_file), answer_key="gold")
            ds.load()
            assert "answer" in ds.dataset.column_names
            assert ds.dataset["answer"][0] == "A"
            assert "gold" not in ds.dataset.column_names

    def test_conversation_prompt_kept_as_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            convo = [{"role": "user", "content": "hi"}]
            with open(data_file, "w") as f:
                f.write(json.dumps({"prompt": convo, "answer": "x"}) + "\n")

            ds = GRPODataset(data_path=str(data_file))
            ds.load()
            assert ds.dataset["prompt"][0] == convo


class TestGRPODatasetFormatForTraining:
    def test_loads_lazily_when_dataset_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            with open(data_file, "w") as f:
                for ex in [{"prompt": "a", "answer": "1"}, {"prompt": "b", "answer": "2"}]:
                    f.write(json.dumps(ex) + "\n")

            ds = GRPODataset(data_path=str(data_file))
            formatted = ds.format_for_training(tokenizer=None)
            assert formatted is ds.dataset
            assert len(formatted) == 2

    def test_preloaded_dataset_returned_without_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "data.jsonl"
            with open(data_file, "w") as f:
                f.write(json.dumps({"prompt": "a", "answer": "1"}) + "\n")

            ds = GRPODataset(data_path=str(data_file))
            ds.load()
            with patch.object(ds, "load") as mock_load:
                result = ds.format_for_training(tokenizer="tok", max_length=99)
            mock_load.assert_not_called()
            assert result is ds.dataset
