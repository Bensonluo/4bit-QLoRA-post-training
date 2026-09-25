"""Unit tests for data preprocessing utilities (src/data/preprocessors.py).

DataCollator padding is verified against real torch tensors; formatting and
statistics helpers run on real datasets.Dataset objects.
"""

from typing import Any

import pytest
from datasets import Dataset

from src.data.preprocessors import (
    DataCollator,
    compute_statistics,
    format_instruction,
    print_dataset_statistics,
    tokenize_function,
)


class _Tok:
    pad_token_id = 0

    def __call__(self, texts: Any, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": [[1, 2] for _ in texts], "attention_mask": [[1, 1] for _ in texts]}


class TestDataCollator:
    def test_pads_to_max_length_with_pad_token(self) -> None:
        collator = DataCollator(_Tok(), max_length=6)

        batch = collator([{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}])

        assert batch["input_ids"] == [[1, 2, 3, 0, 0, 0]]
        assert len(batch["attention_mask"][0]) == 6

    def test_longest_padding_keeps_max_batch_len(self) -> None:
        collator = DataCollator(_Tok(), padding="longest", max_length=100)

        batch = collator([{"input_ids": [1]}, {"input_ids": [1, 2, 3]}])

        assert batch["input_ids"] == [[1, 0, 0], [1, 2, 3]]

    def test_truncates_sequences_over_max_length(self) -> None:
        collator = DataCollator(_Tok(), max_length=2)

        batch = collator([{"input_ids": [1, 2, 3, 4]}])

        assert batch["input_ids"] == [[1, 2]]

    def test_pad_to_multiple_of_rounds_up(self) -> None:
        collator = DataCollator(_Tok(), max_length=10, pad_to_multiple_of=8)

        batch = collator([{"input_ids": [1, 2, 3]}])

        # max_length pads 3 → 10, then the multiple rounds up to 16.
        assert len(batch["input_ids"][0]) == 16

    def test_pad_to_multiple_of_noop_on_exact_multiple(self) -> None:
        collator = DataCollator(_Tok(), max_length=8, pad_to_multiple_of=8)

        batch = collator([{"input_ids": [1, 2]}])

        assert len(batch["input_ids"][0]) == 8

    def test_labels_are_padded_like_inputs(self) -> None:
        collator = DataCollator(_Tok(), max_length=4)

        batch = collator([{"input_ids": [1, 2], "labels": [5, 6]}])

        assert batch["labels"] == [[5, 6, 0, 0]]

    def test_other_keys_passed_through_unpadded(self) -> None:
        collator = DataCollator(_Tok(), max_length=4)

        batch = collator([{"input_ids": [1], "weight": 0.5}, {"input_ids": [1], "weight": 2.0}])

        assert batch["weight"] == [0.5, 2.0]


class TestFormatInstruction:
    def test_alpaca_with_input(self) -> None:
        result = format_instruction("Summarize", "Article text", "Summary")
        assert "### Instruction:\nSummarize" in result
        assert "### Input:\nArticle text" in result
        assert "### Response:\nSummary" in result

    def test_alpaca_without_input(self) -> None:
        result = format_instruction("Summarize", "", "Summary")
        assert "### Input:" not in result

    def test_chat_with_input_and_output(self) -> None:
        result = format_instruction("Q", "ctx", "A", format_type="chat")
        assert result == "user: Q\n\nctx\nassistant: A"

    def test_chat_without_input_or_output(self) -> None:
        result = format_instruction("Q", format_type="chat")
        assert result == "user: Q"

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format type"):
            format_instruction("Q", format_type="bogus")


class TestTokenizeFunction:
    def test_forwards_column_and_kwargs(self) -> None:
        tok = _Tok()

        result = tokenize_function(
            {"text": ["a b", "c"]},
            tok,
            max_length=32,  # type: ignore[arg-type]
        )

        assert result["input_ids"] == [[1, 2], [1, 2]]


class TestComputeStatistics:
    def test_with_text_column(self) -> None:
        ds = Dataset.from_list([{"text": "a b c"}, {"text": "a"}])

        stats = compute_statistics(ds)

        assert stats["num_samples"] == 2
        assert stats["columns"] == ["text"]
        assert stats["avg_length"] == pytest.approx(2.0)
        assert stats["max_length"] == 3
        assert stats["min_length"] == 1

    def test_without_text_column(self) -> None:
        ds = Dataset.from_list([{"instruction": "x"}])

        stats = compute_statistics(ds)

        assert "avg_length" not in stats

    def test_print_statistics_runs(self) -> None:
        ds = Dataset.from_list([{"text": "a b"}])
        print_dataset_statistics(ds, name="Test")  # must not raise


class TestPrintDatasetStatistics:
    def test_no_text_column_skips_length_block(self) -> None:
        from src.data.preprocessors import print_dataset_statistics

        ds = Dataset.from_dict({"instruction": ["a"], "output": ["b"]})

        stats = compute_statistics(ds)
        assert "avg_length" not in stats

        # Must exit cleanly without touching the length fields.
        print_dataset_statistics(ds)
