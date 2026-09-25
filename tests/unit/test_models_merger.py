"""Unit tests for LoRA merge utilities (src/models/merger.py).

The PEFT wrapper classes are faked via ``__new__`` (no constructor) so the
``isinstance(model, PeftModel)`` branches are exercised without loading real
weights; loaders are patched at their source modules because merger.py
imports them locally inside the functions.
"""

from pathlib import Path
from subprocess import CalledProcessError
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import PeftModel

from src.models.merger import (
    compare_models_before_after,
    export_to_gguf,
    load_merged_model,
    merge_adapter_to_dir,
    merge_adapters_weighted,
    merge_lora_into_base,
)


class _FakePeft(PeftModel):
    """Subclass that exists only to satisfy isinstance checks."""

    pass


def _fake_peft(merged: Any = None) -> Any:
    """Build a PeftModel instance without running its constructor."""
    model = _FakePeft.__new__(_FakePeft)
    merged_model = merged if merged is not None else MagicMock()
    model.merge_and_unload = MagicMock(return_value=merged_model)  # type: ignore[method-assign]
    model.load_adapter = MagicMock()  # type: ignore[method-assign]
    model.save_pretrained = MagicMock()  # type: ignore[method-assign]
    return model


class _Param:
    def __init__(self, n: int, trainable: bool = True) -> None:
        self.n = n
        self.requires_grad = trainable

    def numel(self) -> int:
        return self.n


class _FakeHFModel:
    def __init__(self, *params: _Param) -> None:
        self._params = list(params)
        self.save_pretrained = MagicMock()

    def parameters(self):  # type: ignore[no-untyped-def]
        yield from self._params


class TestMergeAdapterToDir:
    @patch("transformers.AutoTokenizer")
    @patch("peft.AutoPeftModelForCausalLM")
    def test_auto_resolves_base_from_adapter_config(
        self, mock_auto_peft: MagicMock, mock_tok_cls: MagicMock, tmp_path: Any
    ) -> None:
        merged = MagicMock()
        peft_model = _fake_peft(merged=merged)
        mock_auto_peft.from_pretrained.return_value = peft_model
        adapter_dir, out_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")

        result = merge_adapter_to_dir(adapter_dir, out_dir)

        # Base model id resolved from adapter_config.json — no override passed.
        mock_auto_peft.from_pretrained.assert_called_once_with(
            adapter_dir, torch_dtype=torch.bfloat16
        )
        mock_tok_cls.from_pretrained.assert_called_once_with(adapter_dir)
        peft_model.merge_and_unload.assert_called_once()
        merged.save_pretrained.assert_called_once()
        assert result == str(Path(out_dir).resolve())

    @patch("transformers.AutoTokenizer")
    @patch("peft.AutoPeftModelForCausalLM")
    def test_dtype_override_forwarded(
        self, mock_auto_peft: MagicMock, mock_tok_cls: MagicMock, tmp_path: Any
    ) -> None:
        mock_auto_peft.from_pretrained.return_value = _fake_peft()

        merge_adapter_to_dir(str(tmp_path / "a"), str(tmp_path / "o"), dtype="float16")

        assert mock_auto_peft.from_pretrained.call_args.kwargs["torch_dtype"] is torch.float16

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    @patch("peft.PeftModel")
    def test_explicit_base_override_loads_base_and_attaches_adapter(
        self,
        mock_peft_cls: MagicMock,
        mock_base_cls: MagicMock,
        mock_tok_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        peft_model = _fake_peft()
        mock_peft_cls.from_pretrained.return_value = peft_model
        adapter_dir, out_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")

        merge_adapter_to_dir(adapter_dir, out_dir, base_model_name="Qwen/Qwen2.5-0.5B")

        mock_base_cls.from_pretrained.assert_called_once_with(
            "Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16
        )
        mock_peft_cls.from_pretrained.assert_called_once_with(
            mock_base_cls.from_pretrained.return_value, adapter_dir
        )
        mock_tok_cls.from_pretrained.assert_called_once_with("Qwen/Qwen2.5-0.5B")

    @patch("transformers.AutoTokenizer")
    @patch("peft.AutoPeftModelForCausalLM")
    def test_non_peft_model_saved_as_is(
        self, mock_auto_peft: MagicMock, mock_tok_cls: MagicMock, tmp_path: Any
    ) -> None:
        plain = MagicMock()  # not a PeftModel
        mock_auto_peft.from_pretrained.return_value = plain

        merge_adapter_to_dir(str(tmp_path / "a"), str(tmp_path / "o"))

        plain.merge_and_unload.assert_not_called()
        plain.save_pretrained.assert_called_once()

    def test_missing_auto_peft_raises_helpful_error(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `from peft import AutoPeftModelForCausalLM` inside the function
        # raises ImportError when peft is absent/too old — merger converts it
        # into an actionable install hint instead of leaking the ImportError.
        monkeypatch.delattr("peft.AutoPeftModelForCausalLM")

        with pytest.raises(RuntimeError, match="peft>=0.7"):
            merge_adapter_to_dir(str(tmp_path / "a"), str(tmp_path / "o"))


class TestMergeLoraIntoBase:
    def test_peft_model_merges_saves_and_returns(self, tmp_path: Any) -> None:
        merged = MagicMock()
        model = _fake_peft(merged=merged)
        tokenizer = MagicMock()
        out = str(tmp_path / "merged")

        result = merge_lora_into_base(model, "adapters", out, tokenizer=tokenizer)

        model.load_adapter.assert_called_once_with("adapters")
        assert result is merged
        merged.save_pretrained.assert_called_once_with(Path(out))
        tokenizer.save_pretrained.assert_called_once_with(Path(out))

    def test_empty_adapter_path_skips_load(self, tmp_path: Any) -> None:
        model = _fake_peft()

        merge_lora_into_base(model, "", str(tmp_path / "out"))

        model.load_adapter.assert_not_called()

    def test_non_peft_model_returned_unmerged(self, tmp_path: Any) -> None:
        plain = MagicMock()  # not a PeftModel — saved without merging

        result = merge_lora_into_base(plain, "adapters", str(tmp_path / "out"))

        assert result is plain
        plain.save_pretrained.assert_called_once()


class TestExportToGguf:
    @patch("subprocess.run")
    def test_success_invokes_llama_cpp_convert(self, mock_run: MagicMock) -> None:
        export_to_gguf("model_dir", "out.gguf", quantization="q5_k_m")

        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "llama.cpp/convert.py"
        assert cmd[2] == "model_dir"
        assert cmd[-1] == "q5_k_m"
        mock_run.assert_called_once_with(cmd, check=True)

    @patch("subprocess.run")
    def test_convert_failure_does_not_raise(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = CalledProcessError(1, "convert.py")

        export_to_gguf("model_dir", "out.gguf")  # must not raise


class TestLoadMergedModel:
    @patch("transformers.AutoModelForCausalLM")
    def test_loads_with_auto_device_and_dtype(self, mock_cls: MagicMock) -> None:
        result = load_merged_model("merged_dir")

        mock_cls.from_pretrained.assert_called_once_with(
            "merged_dir", device_map="auto", torch_dtype="auto"
        )
        assert result is mock_cls.from_pretrained.return_value


class TestCompareModelsBeforeAfter:
    def test_prints_parameter_comparison(self) -> None:
        base = _FakeHFModel(_Param(1000), _Param(50, trainable=True))
        tuned = _FakeHFModel(_Param(1000), _Param(20, trainable=False))

        compare_models_before_after(base, tuned)  # must not raise

    def test_zero_parameter_models(self) -> None:
        compare_models_before_after(_FakeHFModel(), _FakeHFModel())


class TestMergeAdaptersWeighted:
    def _make_adapter_dir(self, tmp_path: Any, name: str, base: str = "Qwen/Qwen2.5-0.5B") -> str:
        import json

        d = tmp_path / name
        d.mkdir()
        (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
        return str(d)

    def test_single_adapter_rejected(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            merge_adapters_weighted([str(tmp_path / "only")], str(tmp_path / "out"))

    def test_unknown_combination_type_rejected(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="combination_type"):
            merge_adapters_weighted(["a", "b"], str(tmp_path / "out"), combination_type="bogus")

    def test_weight_count_mismatch_rejected(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="weights length"):
            merge_adapters_weighted(["a", "b", "c"], str(tmp_path / "out"), weights=[1.0, 1.0])

    def test_density_required_for_ties(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="requires density"):
            merge_adapters_weighted(
                ["a", "b"], str(tmp_path / "out"), combination_type="ties", density=None
            )

    def test_density_range_validated(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="density"):
            merge_adapters_weighted(
                ["a", "b"], str(tmp_path / "out"), combination_type="ties", density=1.5
            )

    def test_majority_sign_method_validated(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="majority_sign_method"):
            merge_adapters_weighted(
                ["a", "b"],
                str(tmp_path / "out"),
                combination_type="ties",
                density=0.2,
                majority_sign_method="bogus",
            )

    def test_base_model_mismatch_rejected(self, tmp_path: Any) -> None:
        a = self._make_adapter_dir(tmp_path, "a", base="Qwen/Qwen2.5-0.5B")
        b = self._make_adapter_dir(tmp_path, "b", base="Qwen/Qwen2.5-1.5B")

        with pytest.raises(ValueError, match="share one base model"):
            merge_adapters_weighted([a, b], str(tmp_path / "out"), density=0.2)

    def test_missing_adapter_config_rejected(self, tmp_path: Any) -> None:
        (tmp_path / "empty").mkdir()
        a = self._make_adapter_dir(tmp_path, "a")

        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            merge_adapters_weighted(
                [a, str(tmp_path / "empty")], str(tmp_path / "out"), density=0.2
            )

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    @patch("peft.PeftModel")
    def test_happy_path_ties_combines_and_merges(
        self,
        mock_peft_cls: MagicMock,
        mock_auto_cls: MagicMock,
        mock_tok_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        a = self._make_adapter_dir(tmp_path, "a")
        b = self._make_adapter_dir(tmp_path, "b")
        peft_model = _fake_peft()
        peft_model.set_adapter = MagicMock()
        peft_model.add_weighted_adapter = MagicMock()
        mock_peft_cls.from_pretrained.return_value = peft_model
        out_dir = str(tmp_path / "merged")

        result = merge_adapters_weighted(
            [a, b], out_dir, weights=[0.7, 0.3], combination_type="ties", density=0.2
        )

        # First adapter attaches via from_pretrained, second via load_adapter.
        mock_peft_cls.from_pretrained.assert_called_once()
        assert mock_peft_cls.from_pretrained.call_args.kwargs["adapter_name"] == "adapter_0"
        peft_model.load_adapter.assert_called_once_with(b, adapter_name="adapter_1")
        peft_model.add_weighted_adapter.assert_called_once_with(
            adapters=["adapter_0", "adapter_1"],
            weights=[0.7, 0.3],
            adapter_name="merged",
            combination_type="ties",
            density=0.2,
            majority_sign_method="total",
            svd_rank=None,
            svd_clamp=None,
        )
        peft_model.set_adapter.assert_called_once_with("merged")
        peft_model.merge_and_unload.assert_called_once_with(adapter_names=["merged"])
        peft_model.merge_and_unload.return_value.save_pretrained.assert_called_once()
        mock_tok_cls.from_pretrained.assert_called_once_with(a)
        assert result == str(Path(out_dir).resolve())

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    @patch("peft.PeftModel")
    def test_none_weights_materialize_equal(
        self,
        mock_peft_cls: MagicMock,
        mock_auto_cls: MagicMock,
        mock_tok_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        a = self._make_adapter_dir(tmp_path, "a")
        b = self._make_adapter_dir(tmp_path, "b")
        peft_model = _fake_peft()
        peft_model.set_adapter = MagicMock()
        peft_model.add_weighted_adapter = MagicMock()
        mock_peft_cls.from_pretrained.return_value = peft_model

        merge_adapters_weighted([a, b], str(tmp_path / "out"), density=0.2)

        assert peft_model.add_weighted_adapter.call_args.kwargs["weights"] == [1.0, 1.0]

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    @patch("peft.PeftModel")
    def test_linear_needs_no_density(
        self,
        mock_peft_cls: MagicMock,
        mock_auto_cls: MagicMock,
        mock_tok_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        a = self._make_adapter_dir(tmp_path, "a")
        b = self._make_adapter_dir(tmp_path, "b")
        peft_model = _fake_peft()
        peft_model.set_adapter = MagicMock()
        peft_model.add_weighted_adapter = MagicMock()
        mock_peft_cls.from_pretrained.return_value = peft_model

        merge_adapters_weighted(
            [a, b], str(tmp_path / "out"), combination_type="linear", density=None
        )

        peft_model.add_weighted_adapter.assert_called_once()

    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModelForCausalLM")
    @patch("peft.PeftModel")
    def test_tokenizer_falls_back_to_base_when_adapter_dir_has_none(
        self,
        mock_peft_cls: MagicMock,
        mock_auto_cls: MagicMock,
        mock_tok_cls: MagicMock,
        tmp_path: Any,
    ) -> None:
        a = self._make_adapter_dir(tmp_path, "a")
        b = self._make_adapter_dir(tmp_path, "b")
        peft_model = _fake_peft()
        peft_model.set_adapter = MagicMock()
        peft_model.add_weighted_adapter = MagicMock()
        mock_peft_cls.from_pretrained.return_value = peft_model
        # Adapter dirs saved without a tokenizer raise; the base id must rescue.
        fallback_tok = MagicMock()
        mock_tok_cls.from_pretrained.side_effect = [OSError("no tokenizer files"), fallback_tok]

        result = merge_adapters_weighted(
            [a, b], str(tmp_path / "out"), combination_type="ties", density=0.5
        )

        assert mock_tok_cls.from_pretrained.call_args_list[1].args[0] == "Qwen/Qwen2.5-0.5B"
        fallback_tok.save_pretrained.assert_called_once()
        assert result == str(Path(tmp_path / "out").resolve())
