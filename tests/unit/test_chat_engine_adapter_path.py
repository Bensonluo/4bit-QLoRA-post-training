"""06 Chat 手输 adapter ~ 展开(R116,R115 obs-2 家族收口):行为钉。

选点依据(r116-scout 裁决,主会话一手复核全采纳):
06_Chat.py 手输「LoRA adapter 路径」字段(页面层零校验)→ _cached_load →
load_chat_model → ``PeftModel.from_pretrained(model, adapter_path)`` 原样
透传——~/... 被 HF 当 repo id 拒收(先例:domains/medical_entity/eval/models.py
RealFinetunedModel._load 对 adapter 的同款展开与注释;src/models/merger.py
模块头注释原文「HF treats them as repo ids and rejects them」),且重试救不了
(每次重试同一个 ~ 路径)。picker 路径不含 ~(discovery 的 .resolve() 绝对真路径,
一手核实:adapter 分支 cfg_path.parent.resolve() / merged 分支 d.resolve()),
唯一 ~ 入口 = 手输框;页面无存在性检查,from_pretrained 喂入点唯一——修
chat_engine 单点即修全链。底座侧 R115 已覆盖(expand_user_ref(base_model)),
本轮收口 adapter 侧,家族至此齐守。

钉型:行为钉(monkeypatch recorder,仿 test_models_merger.py tilde 钉先例——
patch src.models.loader.load_model_and_tokenizer 返回 fake + patch
peft.PeftModel 为 recorder:load_chat_model 的函数级 from-import 在调用时
解析到 patch)+ HF 名逐字节极性(披露:旧实现本就透传,守卫极性非 RED)+
双 ~ 合并旅程 + None 分支保活极性(06_Chat 手输框 ``.strip() or None`` 归一
喂 None 真实存在)。接线钉在 test_hf_refs.py::TestWiring(子串锚不引行号,
负极性防重内联,R114 nit-1 同型)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import peft
import pytest

import src.models.loader as loader_mod
from src.inference.chat_engine import load_chat_model


class _FakeBase:
    """load_model_and_tokenizer 的替身:eval() no-op 即够。"""

    def eval(self) -> _FakeBase:
        return self


@pytest.fixture()
def engine_recorder(monkeypatch: pytest.MonkeyPatch) -> Any:
    """patch 掉底座加载与 peft 包装,记录两者实际收到的引用。"""
    calls: dict[str, list[str]] = {"bases": [], "adapters": []}

    def fake_loader(cfg: Any) -> tuple[_FakeBase, object]:
        calls["bases"].append(cfg.name)
        return _FakeBase(), object()

    class FakePeft:
        def __init__(self, _model: Any) -> None:
            pass

        @classmethod
        def from_pretrained(cls, _model: Any, adapter_path: str) -> FakePeft:
            calls["adapters"].append(adapter_path)
            return cls(_model)

        def merge_and_unload(self) -> _FakeBase:
            return _FakeBase()

    monkeypatch.setattr(loader_mod, "load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr(peft, "PeftModel", FakePeft)
    return calls


def test_tilde_adapter_expanded_before_peft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_recorder: Any
) -> None:
    """RED 主钉:手输 ~/... 必须先展开再喂 PeftModel——未展开即 repo-id
    拒收,06 Chat 页加载按钮每次重试都收到同一个 ~ 路径,救不了。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    # R117 setup 更新:展开后的绝对路径如今过预检,须真实存在且含 marker
    adapter_dir = tmp_path / "my-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", "~/my-adapter")

    assert engine_recorder["adapters"] == [str(tmp_path / "my-adapter")], (
        "adapter 引用未展开就喂给了 PeftModel"
    )


def test_hf_repo_id_adapter_byte_identical(engine_recorder: Any) -> None:
    """极性钉:adapter 亦可为 HF repo id——展开不得腐蚀合法 HF 名
    (R115 逐字节透传义务在 adapter 侧同样成立;旧实现本就透传,守卫极性)。"""
    load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/org-adapter")

    assert engine_recorder["adapters"] == ["Qwen/org-adapter"]


def test_base_and_adapter_both_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_recorder: Any
) -> None:
    """合并旅程:底座与 adapter 双 ~/... 同轮双展开——底座侧是 R115 已修
    表达式,本钉同时守两侧(底座回退也会让此钉变红)。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    # R117 setup 更新:展开后的绝对路径如今过预检,须真实存在且含 marker
    adapter_dir = tmp_path / "my-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    load_chat_model("~/.cache/base", "~/my-adapter")

    assert engine_recorder["bases"] == [str(tmp_path / ".cache" / "base")]
    assert engine_recorder["adapters"] == [str(tmp_path / "my-adapter")]


def test_none_adapter_skips_peft(engine_recorder: Any) -> None:
    """分支保活极性:adapter=None(06_Chat 手输框 ``.strip() or None`` 的真实
    喂入)不得触碰 PeftModel——纯底座对话路径。"""
    load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", None)

    assert engine_recorder["adapters"] == []
