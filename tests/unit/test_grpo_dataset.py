"""Unit tests for GRPO dataset loader."""

import json
import tempfile
from pathlib import Path

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
