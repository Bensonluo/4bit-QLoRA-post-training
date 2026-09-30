"""06 Chat 手输 adapter 预检 fail-fast(R117):行为钉。

选点依据(r117-scout 裁决,主会话一手复核采纳,价值故事修正):
r116-scout 称「难懂错误」被一手实测推翻——venv peft 对坏 adapter 输入
(不存在绝对路径/无 config 目录/HF 名)一律抛 ``ValueError: Can't find
'adapter_config.json' at '<原路径>'``,离线快失败、并不 cryptic。真实价值:
① **fail-fast 时序**——现引擎先 ``load_model_and_tokenizer``(未缓存底座
下载=分钟级)后 PeftModel,坏路径要等底座加载完才报错,且 ``st.cache_resource``
不缓存失败 → 每次重试重付等待;② 中文修法文案;③ merged 目录误填指引。
可达性(诚实):picker-陈旧场景不可达(discover 在每次 rerun 先于按钮 handler
重扫,被删 option 从 labels 消失);真实受众 = CUSTOM 分支手输者
(同事分发型旅程:outputs/ 外 adapter 拷到本地填入)。

守卫设计:仅**绝对路径**预检(``Path.is_absolute()`` 纯谓词,不做字符串变异
——R115 禁 str(Path(x)) 变异,谓词无变异);非绝对(HF repo id / 相对路径)
透传不检查——R116 的 HF 名逐字节极性不得被此守卫破坏。相对路径保持今日
行为(迟到报错维持,修复面如实披露)。预检块必须 hoist 到底座加载**之前**,
否则 fail-fast 不成立。R116 既有 tilde 钉喂的 ``~/my-adapter`` 展开后不存在,
setup 已补 mkdir+marker(既有断言原文不动,披露于轮报)。

钉型:行为钉(recorder 复用 R116 test_chat_engine_adapter_path.py 同錯形:
patch loader+peft.PeftModel;**fail-fast 本体以 ``bases == []`` 钉死**——
底座加载器不得被调用)+ HF 名/相对路径极性(绿披露)+ 真 adapter 防误伤(绿)。
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


def _make_adapter_dir(root: Path, name: str = "my-adapter") -> Path:
    """真 adapter 目录:discovery 同款 marker(adapter_config.json)。"""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    return d


def test_nonexistent_absolute_adapter_fails_fast(engine_recorder: Any) -> None:
    """RED 主钉:手输不存在的绝对路径必须在底座加载**之前**报错——现实现
    要先等底座下载/加载完才在 peft 抛错,重试重付等待。原路径回显。"""
    with pytest.raises(FileNotFoundError, match=r"路径不存在或不是目录：/nonexistent/xyz-adapter"):
        load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", "/nonexistent/xyz-adapter")

    assert engine_recorder["bases"] == [], "fail-fast 本体:底座加载器不得被调用"
    assert engine_recorder["adapters"] == []


def test_dir_without_marker_fails_with_guidance(tmp_path: Path, engine_recorder: Any) -> None:
    """RED:存在但缺 adapter_config.json 的目录(典型=merged 模型误填)——
    报错须含 marker 名与修法指引(改填底座框),且同样在底座加载之前。"""
    empty = tmp_path / "not-an-adapter"
    empty.mkdir()

    # match 覆盖 marker 名 + 修法指引(obs-4 折叠:文案漂移须红)
    with pytest.raises(ValueError, match=r"缺 adapter_config.json.*底座模型"):
        load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", str(empty))

    assert engine_recorder["bases"] == [], "fail-fast 本体:底座加载器不得被调用"


def test_hf_repo_id_adapter_bypasses_precheck(engine_recorder: Any) -> None:
    """极性(绿披露):HF repo id 非绝对路径,不得被预检拦截——guard 过
    firing 会毁 R116 保护过的 HF 名路径。正常走完底座+peft。"""
    load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/org-adapter")

    assert engine_recorder["adapters"] == ["Qwen/org-adapter"]
    assert engine_recorder["bases"] != [], "HF 名路径底座加载必须照常发生"


def test_relative_adapter_path_keeps_today_behavior(engine_recorder: Any) -> None:
    """边界(绿披露):相对路径不预检、透传 peft——迟到报错维持,修复面
    如实披露(不夸大为全形态 fail-fast)。"""
    load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", "outputs/sft/run-x")

    assert engine_recorder["adapters"] == ["outputs/sft/run-x"]
    assert engine_recorder["bases"] != [], "相对路径路径底座加载必须照常发生"


def test_real_adapter_dir_passes_precheck(tmp_path: Path, engine_recorder: Any) -> None:
    """防误伤(绿):真 adapter 目录(含 marker)必须过预检直达 peft——本
    变更最大风险 = 守卫误伤真 adapter(同事分发型旅程的正路径)。"""
    d = _make_adapter_dir(tmp_path)

    load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", str(d))

    assert engine_recorder["adapters"] == [str(d)]
    assert engine_recorder["bases"] != []


def test_tilde_nonexistent_composition_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine_recorder: Any
) -> None:
    """组合旅程(r117-reviewer nit-1 采纳,绿披露):手输 ~/不存在 —— 展开→绝对→
    预检 FileNotFoundError,且回显**原始 ~ 形态**(非展开形态),底座不加载。
    {绝对,~}×{存在,缺失}矩阵最后一个未钉单元;修正后价值故事的主力受众
    旅程(同事分发 adapter 拷到本地手输)。"""
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError, match=r"路径不存在或不是目录：~/nope"):
        load_chat_model("Qwen/Qwen2.5-0.5B-Instruct", "~/nope")

    assert engine_recorder["bases"] == [], "fail-fast 本体:底座加载器不得被调用"
    assert engine_recorder["adapters"] == []
