"""Unit tests for base model utilities (src/models/base.py).

transformers/rich classes are patched at source; no weights are loaded.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from src.models.base import BaseModelHandler, get_model_size, print_model_info


class _FakeParam:
    def __init__(self, n: int, device: str = "cpu") -> None:
        self._n = n
        self.device = device

    def numel(self) -> int:
        return self._n


class _FakeModel:
    def __init__(self, *sizes: int, quantized: bool = False) -> None:
        self._params = [_FakeParam(s) for s in sizes]
        self.config = SimpleNamespace()
        if quantized:
            self.config.quantization_config = "nf4-double"

    def parameters(self):
        # Generator, like torch nn.Module.parameters() — print_model_info
        # calls next() on it, which requires an iterator.
        yield from self._params

    def save_pretrained(self, out: str) -> None:
        self.saved_to = out


class _FakeTokenizer:
    vocab_size = 151_936

    def save_pretrained(self, out: str) -> None:
        self.saved_to = out


class TestBaseModelHandler:
    def test_save_model_writes_both_artifacts(self, tmp_path: Any) -> None:
        model, tok = _FakeModel(10), _FakeTokenizer()
        BaseModelHandler(model, tok).save_model(str(tmp_path / "out"))  # type: ignore[arg-type]
        assert model.saved_to == str(tmp_path / "out")
        assert tok.saved_to == str(tmp_path / "out")
        assert (tmp_path / "out").exists()  # dir created

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_from_pretrained_loads_pair(self, mock_m: MagicMock, mock_t: MagicMock) -> None:
        mock_m.return_value = "M"
        mock_t.return_value = "T"
        handler = BaseModelHandler.from_pretrained("/models/qwen")
        mock_m.assert_called_once_with("/models/qwen")
        assert (handler.model, handler.tokenizer) == ("M", "T")


class TestGetModelSize:
    def test_billions(self) -> None:
        count, size = get_model_size(_FakeModel(1_500_000_000))  # type: ignore[arg-type]
        assert count == 1_500_000_000 and size == "1.50B"

    def test_millions(self) -> None:
        _, size = get_model_size(_FakeModel(2_500_000))  # type: ignore[arg-type]
        assert size == "2.50M"

    def test_thousands(self) -> None:
        _, size = get_model_size(_FakeModel(500_000))  # type: ignore[arg-type]
        assert size == "500.00K"  # 5e5 < 1e6 → K tier

    def test_small_k(self) -> None:
        _, size = get_model_size(_FakeModel(1_234))  # type: ignore[arg-type]
        assert size == "1.23K"

    def test_small_int_raw(self) -> None:
        count, size = get_model_size(_FakeModel(42))  # type: ignore[arg-type]
        assert (count, size) == (42, "42")


class TestPrintModelInfo:
    @patch("rich.console.Console")
    def test_prints_params_vocab_device_quantization(self, mock_console_cls: MagicMock) -> None:
        console = MagicMock()
        mock_console_cls.return_value = console
        model = _FakeModel(1_500_000_000, quantized=True)
        print_model_info(model, _FakeTokenizer())  # type: ignore[arg-type]

        text = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "1.50B" in text
        assert "151,936" in text
        assert "cpu" in text  # device from first param
        assert "nf4-double" in text  # quantization config surfaced

    @patch("rich.console.Console")
    def test_no_quantization_line_when_plain(self, mock_console_cls: MagicMock) -> None:
        console = MagicMock()
        mock_console_cls.return_value = console
        print_model_info(_FakeModel(10), _FakeTokenizer())  # type: ignore[arg-type]
        text = " ".join(str(c.args[0]) for c in console.print.call_args_list)
        assert "Quantization" not in text
