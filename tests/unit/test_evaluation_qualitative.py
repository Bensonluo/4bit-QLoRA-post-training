"""Unit tests for qualitative generation evaluation (src/evaluation/qualitative.py).

Model/tokenizer are MagicMock doubles; prompts come from real in-memory
datasets.Dataset objects. The interactive REPL is driven by faking
console.input side effects.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from datasets import Dataset

import src.evaluation.qualitative as qual
from src.evaluation.qualitative import generate_samples, interactive_generation


def _tokenizer(decode_value: Any = "answer") -> MagicMock:
    """Tokenizer double: __call__ → dict-yielding mock; decode → given value."""
    tok = MagicMock()
    tok.return_value.to.return_value = {"input_ids": [1, 2, 3]}
    tok.decode.return_value = decode_value
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    return tok


def _model() -> MagicMock:
    model = MagicMock()
    model.generate.return_value = [[9, 9, 9]]
    return model


class TestGenerateSamples:
    def test_prompt_key_dataset(self) -> None:
        ds = Dataset.from_dict({"prompt": ["hello", "world"]})
        model, tok = _model(), _tokenizer("hello answer")

        out = generate_samples(model, tok, ds, num_samples=2)

        # Each sample draws its own dataset row; row 0's response starts with
        # its prompt (stripped), row 1's does not (kept whole).
        assert out == [
            {"prompt": "hello", "response": "answer"},
            {"prompt": "world", "response": "hello answer"},
        ]
        # Generation kwargs pinned: sampling on, ids passed through.
        kwargs = model.generate.call_args.kwargs
        assert kwargs["do_sample"] is True
        assert kwargs["max_length"] == 256
        assert kwargs["pad_token_id"] == 0
        assert kwargs["eos_token_id"] == 1
        model.eval.assert_called()

    def test_instruction_key_gets_alpaca_template(self) -> None:
        ds = Dataset.from_dict({"instruction": ["Summarize this"]})
        model, tok = _model(), _tokenizer("summary")

        out = generate_samples(model, tok, ds)

        assert out[0]["prompt"] == "### Instruction:\nSummarize this\n\n### Response:\n"

    def test_text_key_truncated_to_100_chars(self) -> None:
        long_text = "x" * 250
        ds = Dataset.from_dict({"text": [long_text]})
        model, tok = _model(), _tokenizer(" continuation")

        out = generate_samples(model, tok, ds)

        assert out[0]["prompt"] == "x" * 100
        # Prefix stripping removes exactly the prompt characters.
        assert out[0]["response"] == "continuation".strip()

    def test_short_text_used_whole(self) -> None:
        ds = Dataset.from_dict({"text": ["short text"]})
        out = generate_samples(_model(), _tokenizer("short text + more"), ds)
        assert out[0]["prompt"] == "short text"

    def test_unknown_keys_are_skipped(self) -> None:
        ds = Dataset.from_dict({"label": [1, 2]})
        out = generate_samples(_model(), _tokenizer(), ds, num_samples=2)
        assert out == []
        model = _model()
        generate_samples(model, _tokenizer(), ds)
        model.generate.assert_not_called()

    def test_num_samples_capped_at_dataset_size(self) -> None:
        ds = Dataset.from_dict({"prompt": ["a", "b", "c"]})
        model = _model()
        out = generate_samples(model, _tokenizer("r"), ds, num_samples=99)
        assert len(out) == 3
        assert model.generate.call_count == 3

    def test_non_string_decode_is_stringified(self) -> None:
        sentinel = object()  # not a str → str() fallback branch
        ds = Dataset.from_dict({"prompt": ["p"]})
        out = generate_samples(_model(), _tokenizer(sentinel), ds)
        assert out[0]["response"] == str(sentinel).strip()

    def test_response_without_prompt_prefix_kept_whole(self) -> None:
        ds = Dataset.from_dict({"prompt": ["p"]})
        out = generate_samples(_model(), _tokenizer("totally different"), ds)
        assert out[0]["response"] == "totally different"


class TestInteractiveGeneration:
    def _run(
        self, inputs: list[str], decode_value: Any = "generated text"
    ) -> tuple[MagicMock, MagicMock]:
        model, tok = _model(), _tokenizer(decode_value)
        with patch.object(qual, "console") as mock_console:
            mock_console.input.side_effect = inputs
            interactive_generation(model, tok, max_length=128)
        return model, mock_console

    def test_generates_then_quits(self) -> None:
        model, mock_console = self._run(["what is ML?", "quit"])

        assert model.generate.call_count == 1
        kwargs = model.generate.call_args.kwargs
        assert kwargs["max_length"] == 128
        assert kwargs["do_sample"] is True
        printed = "".join(str(c) for c in mock_console.print.call_args_list)
        assert "Exiting" in printed

    def test_empty_prompt_loops_without_generating(self) -> None:
        model, _ = self._run(["", "q"])

        model.generate.assert_not_called()

    def test_all_exit_words_break_loop(self) -> None:
        for word in ("quit", "exit", "q"):
            model, _ = self._run([word])
            model.generate.assert_not_called()

    def test_prompt_prefix_stripped_from_response(self) -> None:
        model, mock_console = self._run(["hello", "quit"], decode_value=None)

        # decode returns MagicMock (not str) → response printed unstripped.
        model.generate.assert_called_once()
        assert mock_console.print.call_count >= 3

    def test_generation_uses_alpaca_format(self) -> None:
        model, tok = _model(), _tokenizer()
        with patch.object(qual, "console") as mock_console:
            mock_console.input.side_effect = ["task", "quit"]
            interactive_generation(model, tok)

        # The tokenizer received the Alpaca-formatted prompt, not the raw text.
        assert tok.call_args.args[0].startswith("### Instruction:\ntask")

    def test_response_printed_for_valid_prompt(self) -> None:
        _, mock_console = self._run(["hi", "quit"], decode_value="a reply")
        printed = "".join(
            str(c.args[0]) if c.args else str(c) for c in mock_console.print.call_args_list
        )
        assert "a reply" in printed

    def test_prompt_prefix_stripped_before_printing(self) -> None:
        # decode returns prompt+answer → only the answer is printed.
        formatted = "### Instruction:\nhello\n\n### Response:\n"
        _, mock_console = self._run(["hello", "quit"], decode_value=formatted + "an answer")

        printed = "".join(
            str(c.args[0]) if c.args else str(c) for c in mock_console.print.call_args_list
        )
        assert "an answer" in printed
        assert "### Instruction:" not in printed
