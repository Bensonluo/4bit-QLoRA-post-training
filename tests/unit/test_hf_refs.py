"""HF 引用 ~ 展开守卫(R115,R114 obs-1 模式类收口):expand_user_ref 纯函数钉 + 三站点接线钉。

选点依据(r115-scout 裁决,主会话一手复核全采纳;含对 R114「四站点」框架的勘误):
``str(Path(x).expanduser())`` 在把 ~ 展开的同时会做两类不该对 HF 引用做的事:
1. Windows 下把 ``/`` 换成 ``\\``——HF repo id(Qwen/Qwen2.5)被毁,from_pretrained
   拒收(R114 reviewer obs-1;merger/chat_engine/eval-models 三站点同款模式类);
2. 归一化病理输入——一手实测(macOS):'a/./b'→'a/b'、'Qwen//Qwen'→'Qwen/Qwen'、
   'models/'→'models'——对本地目录无害,但证明「非 ~ 输入逐字节透传」是真实
   不变量,且使本钉在 macOS 即可区分新旧实现(旧实现回退当场红,无需模拟
   Windows——钉行为不钉实现)。

站点勘误(对 R114 轮报「四站点齐修」表述):registry.py:190 是本地目录消费者
(.is_dir() 守卫,喂 MLflow artifacts,Windows 反斜杠对本地路径是正确形式),
仅为 ~ 展开先例,不在腐蚀类。真腐蚀类 = 3 站点 4 表达式:
- src/inference/chat_engine.py:33(06 Chat 页加载链;loader.py 直收字符串,
  :33 是全链唯一腐蚀点,修它即修全链);
- src/models/merger.py:44(显式底座覆盖分支;None 分支由 peft 从 adapter_config
  自读 base 不经 Path,一手直读确认);
- domains/medical_entity/eval/models.py:23+:35(R114 新函数的两条返回路径)。

诚实定性(入轮报):非用户机器上的活 bug(config/windows.md:14 载 Windows 机
为 WSL2,POSIX 路径不可达);价值 = 文档声明的 native Windows 支持面上的正确性
(CLAUDE.md 明文 Windows 脚本入库=支持目标)+ Chat 页(06 非专家收尾步)在
native Windows 对每个 adapter 对话硬死。参照架构:
src/workbench/business_evaluation.py:131-134 先 is_dir() 再把原字符串传
snapshot_download——正是本修目标形态。

钉型:纯函数钉(逐字节透传极性 parametrize + ~ 展开)+ 接线钉×3(源码
call-shape 子串锚,不引行号——R104 nit-3 行号漂移教训;各带旧表达式负极性,
防重内联回退)。既有钉兼容性已逐条核实:test_models_merger.py 显式覆盖钉
(断言 from_pretrained 实参逐字节 "Qwen/Qwen2.5-0.5B")与 tilde 钉(HOME
monkeypatch 断言展开)均保绿;R114 的 test_medical_eval_base_resolution.py
5+2 钉同理(HF 名 no-op 极性 + ~ 展开两性质 helper 全保)。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.hf_refs import expand_user_ref

ROOT = Path(__file__).resolve().parents[2]
CHAT_ENGINE = ROOT / "src" / "inference" / "chat_engine.py"
MERGER = ROOT / "src" / "models" / "merger.py"
EVAL_MODELS = ROOT / "domains" / "medical_entity" / "eval" / "models.py"


class TestByteIdentity:
    @pytest.mark.parametrize(
        "ref",
        [
            "Qwen/Qwen2.5-0.5B-Instruct",  # HF repo id:极性基线(旧实现本就透传)
            "a/./b",  # 旧实现折叠成 a/b——逐字节钉在 macOS 当场红
            "Qwen//Qwen",  # 旧实现折叠成 Qwen/Qwen
            "models/",  # 旧实现剥掉尾斜杠
        ],
    )
    def test_non_tilde_ref_returned_byte_identical(self, ref: str) -> None:
        """非 ~ 起头的引用逐字节透传——HF 名在 Windows 被换分隔符、病理输入在
        macOS 被折叠,都是 Path() 归一化对 HF 引用值的非法改形。"""
        assert expand_user_ref(ref) == ref


class TestTildeExpansion:
    def test_tilde_ref_expanded_to_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """~ 展开保留(R114 先例):HF 把 ~/... 当 repo id 拒收,展开义务不变;
        bare ~ 展开为 home 本身。"""
        monkeypatch.setenv("HOME", str(tmp_path))
        assert expand_user_ref("~/base") == str(tmp_path / "base")
        assert expand_user_ref("~") == str(tmp_path)


class TestWiring:
    """接线钉:三站点 4 表达式必须经 helper——各带旧表达式负极性,防「保留
    helper、站点重内联」的部分解线逃过纯函数钉(R114 nit-1 接线钉同型)。"""

    def test_chat_engine_uses_helper(self) -> None:
        src = CHAT_ENGINE.read_text(encoding="utf-8")
        assert "expand_user_ref(base_model)" in src, "chat_engine 底座引用必须经 helper"
        assert "str(Path(base_model).expanduser())" not in src, "旧腐蚀表达式不得回归"

    def test_merger_uses_helper_for_base_override(self) -> None:
        src = MERGER.read_text(encoding="utf-8")
        assert "expand_user_ref(base_model_name)" in src, "merger 显式底座必须经 helper"
        assert "str(Path(base_model_name).expanduser())" not in src, "旧腐蚀表达式不得回归"

    def test_eval_models_uses_helper(self) -> None:
        src = EVAL_MODELS.read_text(encoding="utf-8")
        assert "expand_user_ref(base_model)" in src, "eval/models.py 显式 base 必须经 helper"
        assert "return expand_user_ref(base)" in src, "eval/models.py 返回路径必须经 helper"
        assert "str(Path(base).expanduser())" not in src, "旧腐蚀表达式不得回归"
