"""可对话模型发现（src/inference/discovery.py）与聊天引擎延迟导入守卫的单元测试。

覆盖：adapter 发现（含嵌套 checkpoint、坏配置降级）、merged 目录识别、
mtime 排序、limit 截断、空 outputs、重模块零导入守卫。
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import sys
import time
import types
from pathlib import Path

import pytest

from src.inference.chat_engine import generate_reply, stream_reply
from src.inference.discovery import ChatModelOption, discover_chat_models, looks_merged


def _make_adapter(out_dir: Path, base: str | None = "Qwen/Qwen2.5-0.5B-Instruct") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"base_model_name_or_path": base} if base else {"alpha_pattern": {}}
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (out_dir / "adapter_model.safetensors").write_bytes(b"x")
    return out_dir


def _make_merged(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text("{}", encoding="utf-8")
    (out_dir / "model.safetensors").write_bytes(b"x")
    return out_dir


class TestDiscoverChatModels:
    def test_adapter_found_with_base(self, tmp_path) -> None:
        adapter = _make_adapter(tmp_path / "outputs" / "run-a", base="Qwen/Qwen3-1.7B")
        result = discover_chat_models(tmp_path)
        assert len(result) == 1
        assert result[0].kind == "adapter"
        assert result[0].path == adapter.resolve()
        assert result[0].base_model == "Qwen/Qwen3-1.7B"

    def test_nested_checkpoint_adapter_found(self, tmp_path) -> None:
        _make_adapter(tmp_path / "outputs" / "run-b" / "checkpoint-100")
        result = discover_chat_models(tmp_path)
        assert [o.path.name for o in result] == ["checkpoint-100"]

    def test_corrupted_adapter_config_lists_with_base_none(self, tmp_path) -> None:
        d = tmp_path / "outputs" / "run-c"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{not json", encoding="utf-8")
        result = discover_chat_models(tmp_path)
        assert len(result) == 1
        assert result[0].base_model is None  # 条目仍在（UI 会要求手动补底座）

    def test_merged_dir_detected(self, tmp_path) -> None:
        merged = _make_merged(tmp_path / "outputs" / "merged" / "run-d")
        result = discover_chat_models(tmp_path)
        assert len(result) == 1
        assert result[0].kind == "merged"
        assert result[0].path == merged.resolve()
        assert result[0].base_model is None

    def test_config_without_weights_not_merged(self, tmp_path) -> None:
        d = tmp_path / "outputs" / "merged" / "empty"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        assert discover_chat_models(tmp_path) == []

    def test_pytorch_bin_counts_as_weights(self, tmp_path) -> None:
        d = tmp_path / "outputs" / "merged" / "legacy"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "pytorch_model.bin").write_bytes(b"x")
        assert discover_chat_models(tmp_path)[0].kind == "merged"

    def test_empty_outputs_returns_empty(self, tmp_path) -> None:
        (tmp_path / "outputs").mkdir()
        assert discover_chat_models(tmp_path) == []

    def test_missing_outputs_returns_empty(self, tmp_path) -> None:
        assert discover_chat_models(tmp_path) == []

    def test_sorted_newest_first(self, tmp_path) -> None:
        old = _make_adapter(tmp_path / "outputs" / "old-run")
        new = _make_adapter(tmp_path / "outputs" / "new-run")
        past = time.time() - 10000
        os.utime(old, (past, past))
        result = discover_chat_models(tmp_path)
        assert [p.name for p in [o.path for o in result]] == [
            new.name,
            old.name,
        ]

    def test_limit_caps_results(self, tmp_path) -> None:
        for i in range(5):
            _make_adapter(tmp_path / "outputs" / f"run-{i}")
        result = discover_chat_models(tmp_path, limit=3)
        assert len(result) == 3


class TestChatModelOptionLabel:
    def test_adapter_label_shows_base(self) -> None:
        opt = ChatModelOption(
            kind="adapter", path=Path("/x/outputs/test-quick"), base_model="Qwen/Qwen3-1.7B"
        )
        assert opt.label == "🔧 test-quick（底座 Qwen/Qwen3-1.7B）"

    def test_merged_label_no_base(self) -> None:
        opt = ChatModelOption(kind="merged", path=Path("/x/outputs/merged/run"), base_model=None)
        assert opt.label == "📦 run"


class TestLooksMerged:
    """Training Lab 一键合并的幂等判定与 Chat 页发现共用 looks_merged。"""

    def test_config_plus_safetensors_is_merged(self, tmp_path) -> None:
        d = tmp_path / "merged"
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"x")
        assert looks_merged(d) is True

    def test_config_plus_pytorch_bin_is_merged(self, tmp_path) -> None:
        d = tmp_path / "merged"
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "pytorch_model.bin").write_bytes(b"x")
        assert looks_merged(d) is True

    def test_config_without_weights_not_merged(self, tmp_path) -> None:
        d = tmp_path / "partial"
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        assert looks_merged(d) is False

    def test_weights_without_config_not_merged(self, tmp_path) -> None:
        d = tmp_path / "orphan"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"x")
        assert looks_merged(d) is False

    def test_adapter_dir_with_config_and_weights_not_merged(self, tmp_path) -> None:
        """adapter 目录也可能有 config.json——adapter_config.json 在则不是合并模型。"""
        d = tmp_path / "adapter"
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"x")
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert looks_merged(d) is False

    def test_missing_dir_not_merged(self, tmp_path) -> None:
        assert looks_merged(tmp_path / "nope") is False


class _FakeStreamer:
    """镜像 transformers.TextIteratorStreamer 的最小 API（queue + stop 信号）。"""

    stop_signal = None

    def __init__(self, tokenizer, skip_prompt=False, timeout=None, **decode_kwargs):
        self.text_queue = queue.Queue()
        self.timeout = timeout

    def on_finalized_text(self, text, stream_end=False):
        self.text_queue.put(text)
        if stream_end:
            self.text_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        return value


class _FakeModel:
    def __init__(self, chunks: list[str] | None = None, error: Exception | None = None):
        self.device = "cpu"
        self._chunks = chunks
        self._error = error

    def generate(self, *args, **kwargs):
        if self._error is not None:
            raise self._error
        streamer = kwargs["streamer"]
        # 真实 generate 的最后一块文本随 stream_end=True 一起下发
        for chunk in self._chunks[:-1]:
            streamer.on_finalized_text(chunk)
        streamer.on_finalized_text(self._chunks[-1], stream_end=True)


class _FakeTensor:
    def __init__(self, n: int):
        self.shape = (1, n)

    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": _FakeTensor(7), "attention_mask": _FakeTensor(7)}


class TestStreamReply:
    def _patch_heavy(self, monkeypatch):
        """stream_reply 在函数内 import torch / transformers——换成可控假件。"""
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TextIteratorStreamer = _FakeStreamer
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        fake_torch = types.SimpleNamespace(no_grad=lambda: contextlib.nullcontext())
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def test_stream_reply_yields_chunks_in_order(self, monkeypatch) -> None:
        self._patch_heavy(monkeypatch)
        model = _FakeModel(chunks=["你好", "，", "世界"])
        out = list(stream_reply(model, _FakeTokenizer(), [{"role": "user", "content": "hi"}]))
        assert out == ["你好", "，", "世界"]

    def test_generate_reply_joins_chunks(self, monkeypatch) -> None:
        self._patch_heavy(monkeypatch)
        model = _FakeModel(chunks=["你好", "，", "世界"])
        text = generate_reply(model, _FakeTokenizer(), [{"role": "user", "content": "hi"}])
        assert text == "你好，世界"

    def test_thread_death_raises_runtime_error_with_cause(self, monkeypatch) -> None:
        """生成线程异常死掉时队列收不到停止信号——靠 timeout 打破并带出原因。"""
        self._patch_heavy(monkeypatch)
        model = _FakeModel(error=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            list(
                stream_reply(
                    model, _FakeTokenizer(), [{"role": "user", "content": "hi"}], timeout=0.3
                )
            )


class TestNoHeavyImports:
    def test_discovery_module_imports_no_ml_stack(self) -> None:
        """discovery 必须保持纯标准库——UI 页面 import 它时不拉 torch。"""
        import subprocess
        import sys as _sys

        code = (
            "import sys; import src.inference.discovery; "
            "heavy = {'torch', 'transformers', 'peft', 'datasets'} & set(sys.modules); "
            "print('HEAVY:' + ','.join(sorted(heavy)) if heavy else 'CLEAN')"
        )
        out = subprocess.run(
            [_sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        assert out.stdout.strip().endswith("CLEAN")

    def test_chat_engine_module_import_is_lazy(self) -> None:
        """chat_engine 模块级 import 不得拉 torch/transformers/peft。"""
        import subprocess
        import sys as _sys

        code = (
            "import sys; import src.inference.chat_engine; "
            "heavy = {'torch', 'transformers', 'peft'} & set(sys.modules); "
            "print('HEAVY:' + ','.join(sorted(heavy)) if heavy else 'CLEAN')"
        )
        out = subprocess.run(
            [_sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            check=True,
        )
        assert out.stdout.strip().endswith("CLEAN")
