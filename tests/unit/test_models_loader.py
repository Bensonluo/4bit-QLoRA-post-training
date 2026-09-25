"""Unit tests for platform-aware model loading (src/models/loader.py).

Platform branches (CUDA/MPS/CPU) are exercised by patching the platform
and distributed-context lookups; no real models are downloaded.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.models.loader as loader
from config.base import ModelConfig
from src.models.loader import (
    _get_quantization_config,
    load_base_model_for_dpo,
    load_model,
    load_model_and_tokenizer,
    load_tokenizer,
)


def _platform(device: str = "mps") -> MagicMock:
    """Fake PlatformInfo for the requested device class."""
    p = MagicMock()
    p.device = device
    p.description = f"test-{device}"
    p.is_cuda = device == "cuda"
    p.is_mps = device == "mps"
    return p


def _dist(distributed: bool = False) -> MagicMock:
    """Fake DistributedInfo."""
    d = MagicMock()
    d.is_distributed = distributed
    d.rank = 0
    d.world_size = 4
    d.local_rank = 2
    return d


def _load(
    device: str, config_overrides: dict[str, Any] | None = None, distributed: bool = False
) -> Any:
    """Run load_model under patched platform/distributed context.

    The real get_platform is patched in BOTH namespaces: ModelConfig's
    __post_init__ (which disables quantization on MPS/CPU) and the loader's.
    """
    overrides = config_overrides or {}
    with (
        patch("src.utils.platform_utils.get_platform", return_value=_platform(device)),
        patch("src.models.loader.get_platform", return_value=_platform(device)),
        patch("src.models.loader.get_torch_dtype", return_value="bfloat16"),
        patch(
            "src.training.distributed.get_distributed_info",
            return_value=_dist(distributed),
        ),
    ):
        config = ModelConfig(name="test-model", **overrides)
        return load_model(config)


class TestLoadTokenizer:
    @patch("src.models.loader.AutoTokenizer")
    def test_pad_token_already_set(self, mock_tok_cls: MagicMock) -> None:
        tok = mock_tok_cls.from_pretrained.return_value
        tok.pad_token = "<pad>"

        result = load_tokenizer("test-model")

        mock_tok_cls.from_pretrained.assert_called_once_with("test-model", trust_remote_code=True)
        assert result is tok
        tok.add_special_tokens.assert_not_called()

    @patch("src.models.loader.AutoTokenizer")
    def test_pad_falls_back_to_eos(self, mock_tok_cls: MagicMock) -> None:
        tok = mock_tok_cls.from_pretrained.return_value
        tok.pad_token = None
        tok.eos_token = "</s>"

        load_tokenizer("test-model")

        assert tok.pad_token == "</s>"
        tok.add_special_tokens.assert_not_called()

    @patch("src.models.loader.AutoTokenizer")
    def test_adds_pad_token_when_no_eos(self, mock_tok_cls: MagicMock) -> None:
        tok = mock_tok_cls.from_pretrained.return_value
        tok.pad_token = None
        tok.eos_token = None

        load_tokenizer("test-model")

        tok.add_special_tokens.assert_called_once_with({"pad_token": "[PAD]"})


class TestGetQuantizationConfig:
    def test_4bit_nf4_double_quant(self) -> None:
        cfg = _get_quantization_config(4)
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True

    def test_8bit(self) -> None:
        cfg = _get_quantization_config(8)
        assert cfg.load_in_8bit is True

    def test_invalid_bits_raise(self) -> None:
        with pytest.raises(ValueError, match="Unsupported quantization"):
            _get_quantization_config(3)

    def test_unavailable_raises_runtime_error(self) -> None:
        with (
            patch.object(loader, "_BITSANDBYTES_AVAILABLE", False),
            pytest.raises(RuntimeError, match="bitsandbytes"),
        ):
            _get_quantization_config(4)


@patch("src.models.loader.AutoModelForCausalLM")
class TestLoadModel:
    def test_mps_loads_then_moves_to_mps(self, mock_cls: MagicMock) -> None:
        loaded = mock_cls.from_pretrained.return_value
        model = _load("mps", {"use_flash_attention": False})

        mock_cls.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, torch_dtype="bfloat16"
        )
        loaded.to.assert_called_once_with("mps")
        # load_model returns the moved model, not the raw from_pretrained one.
        assert model is loaded.to.return_value

    def test_mps_flash_attention_downgraded(self, mock_cls: MagicMock) -> None:
        # use_flash_attention defaults True; MPS must drop attn_implementation.
        _load("mps")

        kwargs = mock_cls.from_pretrained.call_args.kwargs
        assert "attn_implementation" not in kwargs

    def test_cuda_single_gpu_quantized(self, mock_cls: MagicMock) -> None:
        _load("cuda", {"quantization_bits": 4})

        kwargs = mock_cls.from_pretrained.call_args.kwargs
        assert kwargs["device_map"] == "auto"
        assert kwargs["quantization_config"].load_in_4bit is True
        assert kwargs["attn_implementation"] == "flash_attention_2"

    def test_cuda_full_precision_omits_quantization(self, mock_cls: MagicMock) -> None:
        _load("cuda", {"quantization_bits": None})

        kwargs = mock_cls.from_pretrained.call_args.kwargs
        assert "quantization_config" not in kwargs

    def test_cuda_distributed_pins_local_rank(self, mock_cls: MagicMock) -> None:
        _load("cuda", {"quantization_bits": 4}, distributed=True)

        kwargs = mock_cls.from_pretrained.call_args.kwargs
        # One full copy per rank — never "auto" under DDP/DeepSpeed.
        assert kwargs["device_map"] == {"": 2}

    def test_cpu_omits_device_map(self, mock_cls: MagicMock) -> None:
        _load("cpu", {"use_flash_attention": False})

        kwargs = mock_cls.from_pretrained.call_args.kwargs
        assert "device_map" not in kwargs


class TestLoadModelAndTokenizer:
    @patch("src.models.loader.print_model_info")
    @patch("src.models.loader.load_model")
    @patch("src.models.loader.load_tokenizer")
    @patch("src.models.loader.get_platform")
    def test_composes_tokenizer_model_and_info(
        self,
        mock_pf: MagicMock,
        mock_lt: MagicMock,
        mock_lm: MagicMock,
        mock_info: MagicMock,
    ) -> None:
        mock_lm.return_value = "model"
        mock_lt.return_value = "tok"

        model, tok = load_model_and_tokenizer(ModelConfig(name="test-model"))

        assert (model, tok) == ("model", "tok")
        mock_lt.assert_called_once_with(model_name="test-model", trust_remote_code=True)
        mock_info.assert_called_once_with("model", "tok")


class TestLoadBaseModelForDpo:
    @patch("src.models.loader.load_model_and_tokenizer")
    def test_freezes_all_parameters(self, mock_load: MagicMock) -> None:
        param = MagicMock()
        param.requires_grad = True
        model = MagicMock()
        model.parameters.return_value = iter([param])
        mock_load.return_value = (model, "tok")

        m, t = load_base_model_for_dpo(ModelConfig(name="test-model"))

        assert m is model
        assert t == "tok"
        assert param.requires_grad is False


class TestBitsAndBytesImportFallback:
    def test_import_error_disables_bnb_flag(self) -> None:
        """Module-level ImportError fallback sets _BITSANDBYTES_AVAILABLE=False."""
        import importlib
        import sys
        import types

        import transformers

        # transformers is a _LazyModule that re-resolves deleted names on
        # demand, so `del transformers.BitsAndBytesConfig` cannot make the
        # from-import fail. Simulate a distribution without it instead: swap
        # a plain namespace holding exactly the names loader.py imports at
        # module level (same faking pattern the tracking tests use for mlflow).
        keep = ("AutoModelForCausalLM", "AutoTokenizer", "PreTrainedModel", "PreTrainedTokenizer")
        stub = types.SimpleNamespace(**{name: getattr(transformers, name) for name in keep})
        assert not hasattr(stub, "BitsAndBytesConfig")

        try:
            with patch.dict(sys.modules, {"transformers": stub}):
                importlib.reload(loader)
                assert loader._BITSANDBYTES_AVAILABLE is False
        finally:
            importlib.reload(loader)  # restore normal state for other tests
        assert loader._BITSANDBYTES_AVAILABLE is True
