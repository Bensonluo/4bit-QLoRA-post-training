"""Unit tests for src/evaluation (metrics, qualitative, comparisons).

All model/tokenizer interactions use fakes — no weights are loaded.
"""

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from datasets import Dataset
from pytest import approx

from src.evaluation.metrics import compute_accuracy, compute_perplexity


class _FakeEncoding(dict):
    """dict subclass (like BatchEncoding) that supports .to(device)."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        super().__init__(input_ids=input_ids)

    def to(self, device: str) -> "_FakeEncoding":
        return self


class _FakeTokenizer:
    """Tokenizer fake: fixed input_ids; decode() serves queued responses."""

    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, ids: tuple[int, ...] = (5, 6, 7)) -> None:
        self._ids = torch.tensor([list(ids)])
        self._responses: list[str] = []

    def queue_responses(self, *responses: str) -> None:
        self._responses = list(responses)

    def __call__(self, text: str, **_: Any) -> _FakeEncoding:
        return _FakeEncoding(self._ids.clone())

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if self._responses:
            return self._responses.pop(0)
        return "decoded"


class _FakeLossModel:
    """Model fake returning a fixed loss (for perplexity)."""

    device = "cpu"

    def __init__(self, loss_value: float) -> None:
        self._loss = loss_value
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(loss=torch.tensor(self._loss))


class _FakeLogitsModel:
    """Model fake returning logits whose argmax is a fixed prediction row."""

    device = "cpu"

    def __init__(self, predictions: list[int], vocab: int = 16) -> None:
        seq = len(predictions)
        logits = torch.zeros(1, seq, vocab)
        for pos, token in enumerate(predictions):
            logits[0, pos, token] = 1.0
        self._logits = logits

    def eval(self) -> None:
        pass

    def __call__(self, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(logits=self._logits)


class _FakeGenModel:
    """Model fake whose generate() returns a fixed token row."""

    device = "cpu"

    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] | None = None

    def eval(self) -> None:
        pass

    def generate(self, **kwargs: Any) -> torch.Tensor:
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3]])


# ─── compute_perplexity ──────────────────────────────────────────


class TestComputePerplexity:
    def test_constant_loss_gives_exp_of_loss(self) -> None:
        model = _FakeLossModel(loss_value=1.0)
        tokenizer = _FakeTokenizer()
        dataset = Dataset.from_dict({"text": ["a b", "c d"]})

        ppl = compute_perplexity(model, dataset, tokenizer)

        assert ppl == approx(math.e)
        assert model.eval_called

    def test_instruction_key_is_accepted(self) -> None:
        model = _FakeLossModel(loss_value=0.0)
        tokenizer = _FakeTokenizer()
        dataset = Dataset.from_dict({"instruction": ["do the thing"]})

        assert compute_perplexity(model, dataset, tokenizer) == approx(1.0)

    def test_rows_without_text_or_instruction_are_skipped(self) -> None:
        model = _FakeLossModel(loss_value=1.0)
        tokenizer = _FakeTokenizer()
        dataset = Dataset.from_dict({"other": ["x"], "text": ["real text"], "junk": ["y"]})

        # Only the one text row contributes; result identical to single-row case.
        assert compute_perplexity(model, dataset, tokenizer) == approx(math.e)

    def test_row_lacking_both_keys_hits_continue(self) -> None:
        # A real Dataset has uniform columns, so use a plain row list to
        # express "this row has neither text nor instruction".
        calls = {"n": 0}

        class _CountingLossModel(_FakeLossModel):
            def __call__(self, **kw: Any) -> SimpleNamespace:
                calls["n"] += 1
                return super().__call__(**kw)

        model = _CountingLossModel(loss_value=1.0)
        tokenizer = _FakeTokenizer()
        rows: list[dict[str, str]] = [
            {"text": "counted"},
            {"unrelated": "skipped — no text/instruction key"},
        ]

        assert compute_perplexity(model, rows, tokenizer) == approx(math.e)
        # Only the text row reached the forward pass; the other hit `continue`.
        assert calls["n"] == 1

    @patch("src.evaluation.metrics.console")
    def test_progress_printed_every_hundred_rows(self, mock_console: MagicMock) -> None:
        model = _FakeLossModel(loss_value=1.0)
        tokenizer = _FakeTokenizer()
        dataset = Dataset.from_dict({"text": [f"row {i}" for i in range(100)]})

        compute_perplexity(model, dataset, tokenizer)

        mock_console.print.assert_any_call("  Processed 100 examples...")


# ─── compute_accuracy ────────────────────────────────────────────


class TestComputeAccuracy:
    def test_perfect_next_token_predictions(self) -> None:
        # input_ids (5,6,7); predictions shifted-match targets (6,7) → 2/2.
        tokenizer = _FakeTokenizer(ids=(5, 6, 7))
        model = _FakeLogitsModel(predictions=[6, 7, 0])
        dataset = Dataset.from_dict({"text": ["whatever"]})

        assert compute_accuracy(model, dataset, tokenizer) == approx(1.0)

    def test_half_correct_predictions(self) -> None:
        # predictions [6, 0, 0] → only first shifted position matches → 1/2.
        tokenizer = _FakeTokenizer(ids=(5, 6, 7))
        model = _FakeLogitsModel(predictions=[6, 0, 0])
        dataset = Dataset.from_dict({"text": ["whatever"]})

        assert compute_accuracy(model, dataset, tokenizer) == approx(0.5)

    def test_no_text_rows_gives_zero(self) -> None:
        model = _FakeLogitsModel(predictions=[6, 7, 0])
        tokenizer = _FakeTokenizer()
        dataset = Dataset.from_dict({"other": ["no text key"]})

        assert compute_accuracy(model, dataset, tokenizer) == 0.0


# ─── comparisons ─────────────────────────────────────────────────


class TestCompareModels:
    # load_merged_model is imported INSIDE the functions, so patch at the source.
    @patch("src.evaluation.comparisons.compute_perplexity")
    @patch("src.models.load_merged_model")
    def test_compare_models_improvement_math(
        self, mock_load: MagicMock, mock_ppl: MagicMock
    ) -> None:
        from src.evaluation.comparisons import compare_models

        tuned, base = _FakeLossModel(1.0), _FakeLossModel(1.0)
        mock_load.side_effect = [tuned, base]
        mock_ppl.side_effect = [2.0, 4.0]  # tuned, base
        tokenizer = _FakeTokenizer()
        dataset = Dataset.from_dict({"text": ["x"]})

        result = compare_models("tuned-path", "base-path", dataset, tokenizer)

        assert result["perplexity"]["tuned"] == "2.00"
        assert result["perplexity"]["base"] == "4.00"
        # (4 - 2) / 4 * 100 = +50.0%
        assert result["perplexity"]["improvement"] == "+50.0%"

    @patch("src.evaluation.comparisons.compute_perplexity")
    @patch("src.models.load_model_and_tokenizer")
    @patch("src.models.load_merged_model")
    def test_compare_models_falls_back_to_adapter_loading(
        self,
        mock_load_merged: MagicMock,
        mock_load_adapter: MagicMock,
        mock_ppl: MagicMock,
    ) -> None:
        from config.base import ModelConfig
        from src.evaluation.comparisons import compare_models

        tuned, base = _FakeLossModel(1.0), _FakeLossModel(1.0)
        # Merged load of the tuned checkpoint fails (adapter-only directory) →
        # fallback path; the base model loads merged as usual.
        mock_load_merged.side_effect = [RuntimeError("no merged weights"), base]
        mock_load_adapter.return_value = (tuned, _FakeTokenizer())
        mock_ppl.side_effect = [2.0, 4.0]

        result = compare_models(
            "tuned-path", "base-path", Dataset.from_dict({"text": ["x"]}), _FakeTokenizer()
        )

        # Fallback built a ModelConfig pointing at the adapter checkpoint.
        fallback_config = mock_load_adapter.call_args[0][0]
        assert isinstance(fallback_config, ModelConfig)
        assert fallback_config.name == "tuned-path"
        # The fallback-loaded model is the one evaluated first.
        assert mock_ppl.call_args_list[0][0][0] is tuned
        assert result["perplexity"]["improvement"] == "+50.0%"

    @patch("src.models.load_merged_model")
    def test_side_by_side_generation_runs(self, mock_load: MagicMock) -> None:
        from src.evaluation.comparisons import side_by_side_generation

        model = _FakeGenModel()
        mock_load.side_effect = [model, _FakeGenModel()]
        tokenizer = _FakeTokenizer()
        tokenizer.queue_responses("base says hi", "tuned says hello")

        side_by_side_generation("t", "b", ["hi"], tokenizer)  # no exception
