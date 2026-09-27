"""Unit tests for dataset loaders (src/data/loaders.py).

`load()` paths patch `load_dataset` at the module source; `format_for_training`
runs against real `datasets.Dataset` objects so map/remove_columns behave
exactly as in training.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from src.data.loaders import (
    AlpacaDataset,
    FinanceDataset,
    PreferenceDataset,
    load_custom_dataset,
)


class _Tok:
    """Callable tokenizer double that records per-call kwargs."""

    eos_token_id = 4
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"text": text, **kwargs})
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def _mock_ds(n: int = 50) -> MagicMock:
    ds = MagicMock()
    ds.__len__.return_value = n
    selected = MagicMock()
    selected.__len__.return_value = 10
    ds.select.return_value = selected
    return ds


class TestAlpacaLoad:
    @patch("src.data.loaders.load_dataset")
    def test_loads_from_huggingface(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_ds(50)

        ds = AlpacaDataset(data_path="some/data", max_samples=None)

        result = ds.load()

        mock_load.assert_called_once_with("some/data", split="train")
        assert result is mock_load.return_value
        assert ds.dataset is mock_load.return_value

    @patch("src.data.loaders.load_dataset")
    def test_falls_back_to_local_json(self, mock_load: MagicMock) -> None:
        local = _mock_ds(5)
        mock_load.side_effect = [Exception("HF down"), local]

        ds = AlpacaDataset(data_path="data.json")
        ds.load()

        assert mock_load.call_count == 2
        assert mock_load.call_args == (("json",), {"data_files": "data.json", "split": "train"})

    @patch("src.data.loaders.load_dataset")
    def test_both_sources_fail_raises_runtime_error(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [Exception("HF down"), Exception("not json")]

        ds = AlpacaDataset(data_path="bad.json")

        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            ds.load()

    @patch("src.data.loaders.load_dataset")
    def test_max_samples_selects_subset(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_ds(50)

        ds = AlpacaDataset(data_path="some/data", max_samples=10)
        result = ds.load()

        mock_load.return_value.select.assert_called_once_with(range(10))
        assert result is mock_load.return_value.select.return_value

    @patch("src.data.loaders.load_dataset")
    def test_no_selection_when_under_limit(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_ds(5)

        AlpacaDataset(data_path="some/data", max_samples=10).load()

        mock_load.return_value.select.assert_not_called()


class TestAlpacaFormat:
    def _dataset(self, rows: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list(rows)

    def test_formats_with_input(self) -> None:
        ds = AlpacaDataset(data_path="x")
        ds.dataset = self._dataset(
            [{"instruction": "Explain stocks", "input": "AAPL", "output": "A stock is..."}]
        )
        tok = _Tok()

        formatted = ds.format_for_training(tok, max_length=64)  # type: ignore[arg-type]

        assert "### Instruction:" in tok.calls[0]["text"]
        assert "### Input:" in tok.calls[0]["text"]
        assert formatted.column_names == ["input_ids", "attention_mask", "labels"]
        row = formatted[0]
        assert row["labels"] == row["input_ids"]

    def test_formats_without_input_omits_section(self) -> None:
        ds = AlpacaDataset(data_path="x")
        ds.dataset = self._dataset([{"instruction": "Hi", "input": "", "output": "Hello"}])
        tok = _Tok()

        ds.format_for_training(tok, max_length=64)  # type: ignore[arg-type]

        assert "### Input:" not in tok.calls[0]["text"]
        assert "### Response:" in tok.calls[0]["text"]

    def test_tokenizer_kwargs_forwarded(self) -> None:
        ds = AlpacaDataset(data_path="x")
        ds.dataset = self._dataset([{"instruction": "Hi", "input": "", "output": "Yo"}])
        tok = _Tok()

        ds.format_for_training(tok, max_length=128)  # type: ignore[arg-type]

        assert tok.calls[0]["truncation"] is False
        assert tok.calls[0]["padding"] is False
        # EOS is added before explicit length handling so truncation is honest.
        assert "max_length" not in tok.calls[0]

    @patch.object(AlpacaDataset, "load")
    def test_lazy_loads_when_dataset_missing(self, mock_load: MagicMock) -> None:
        ds = AlpacaDataset(data_path="x")
        ds.dataset = None  # type: ignore[assignment]
        real = self._dataset([{"instruction": "Hi", "input": "", "output": "Yo"}])

        def _load_side_effect() -> Dataset:
            ds.dataset = real
            return real

        mock_load.side_effect = _load_side_effect

        ds.format_for_training(_Tok(), max_length=64)  # type: ignore[arg-type]
        mock_load.assert_called_once()


class TestFinanceDataset:
    @patch("src.data.loaders.load_dataset")
    def test_load_filters_to_finance_rows(self, mock_load: MagicMock) -> None:
        raw = Dataset.from_list(
            [
                {"instruction": "Explain stock valuation", "input": "", "output": " stocks!"},
                {"instruction": "Write a poem", "input": "", "output": "roses are red"},
            ]
        )
        mock_load.return_value = raw

        ds = FinanceDataset(data_path="some/data")
        result = ds.load()

        assert len(result) == 1
        assert "stock" in result[0]["instruction"]

    def test_filter_matches_any_field_case_insensitive(self) -> None:
        ds = FinanceDataset(data_path="x")
        rows = Dataset.from_list(
            [
                {"instruction": "plain", "input": "BITCOIN chart", "output": "see input"},
                {"instruction": "plain", "input": "plain", "output": "plain"},
            ]
        )

        filtered = ds._filter_finance(rows)

        assert len(filtered) == 1


class TestPreferenceDataset:
    @patch("src.data.loaders.load_dataset")
    def test_loads_pairs(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_ds(5)

        ds = PreferenceDataset(data_path="prefs")
        result = ds.load()

        assert result is mock_load.return_value

    @patch("src.data.loaders.load_dataset")
    def test_both_sources_fail_raises(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = [Exception("HF down"), Exception("nope")]

        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            PreferenceDataset(data_path="bad.json").load()

    def test_formats_preference_triples(self) -> None:
        ds = PreferenceDataset(data_path="x")
        ds.dataset = Dataset.from_list([{"prompt": "Q", "chosen": "good", "rejected": "bad"}])
        tok = _Tok()

        formatted = ds.format_for_training(tok, max_length=300)  # type: ignore[arg-type]

        assert formatted.column_names == [
            "prompt_input_ids",
            "prompt_attention_mask",
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        ]
        # Prompt gets a third of the budget; completions the full budget.
        max_lengths = [c["max_length"] for c in tok.calls]
        assert max_lengths == [100, 300, 300]
        assert [c["padding"] for c in tok.calls] == [False, False, False]

    @patch("src.data.loaders.load_dataset")
    def test_load_truncates_to_max_samples(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_ds(50)

        ds = PreferenceDataset(data_path="prefs", max_samples=10)
        ds.load()

        mock_load.return_value.select.assert_called_once_with(range(10))
        assert ds.dataset is mock_load.return_value.select.return_value

    @patch("src.data.loaders.load_dataset")
    def test_format_lazy_loads_when_dataset_unset(self, mock_load: MagicMock) -> None:
        # format_for_training must not require an explicit load() first —
        # it lazily loads when self.dataset is None.
        mock_load.return_value = Dataset.from_list(
            [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]
        )
        ds = PreferenceDataset(data_path="prefs")

        formatted = ds.format_for_training(_Tok(), max_length=300)  # type: ignore[arg-type]

        mock_load.assert_called_once_with("prefs", split="train")
        assert "chosen_input_ids" in formatted.column_names


class TestLoadCustomDataset:
    @patch("src.data.loaders.load_dataset")
    def test_loads_json_with_limit(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_ds(50)

        result = load_custom_dataset("file.json", max_samples=10)

        mock_load.assert_called_once_with("json", data_files="file.json", split="train")
        mock_load.return_value.select.assert_called_once_with(range(10))
        assert result is mock_load.return_value.select.return_value

    @patch("src.data.loaders.load_dataset")
    def test_no_limit_when_under(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_ds(5)

        load_custom_dataset("file.json", max_samples=10)

        mock_load.return_value.select.assert_not_called()


class TestFinanceDatasetEmptyLoad:
    @patch("src.data.loaders.load_dataset")
    def test_empty_dataset_skips_finance_filter(self, mock_load: MagicMock) -> None:
        empty = Dataset.from_dict({"instruction": [], "input": [], "output": []})
        mock_load.return_value = empty

        ds = FinanceDataset(data_path="some/data")
        result = ds.load()

        # Empty base dataset is falsy → filter block skipped, no crash.
        assert len(result) == 0
