"""医疗领域评测底座解析 ~ 展开(R114 obs-1 主修):resolve_adapter_base 纯函数钉。

选点依据(r114-scout 裁决采纳,主会话一手复核成立):
domains/medical_entity/eval/models.py 的 RealFinetunedModel._load() 从
adapter_config.json/config.json 读出 base_model_name_or_path 后**不展开 ~**
直传 AutoTokenizer/AutoModelForCausalLM.from_pretrained——HF 把 ~/... 当
repo id 拒收(先例:src/models/merger.py:38-44 注释原文「HF treats them as
repo ids and rejects them」;src/inference/chat_engine.py:33、
src/tracking/registry.py:190 亦全部展开;grep 全仓 expanduser 在 domains/
零命中——eval/models.py 是唯一未展开的 from_pretrained 喂入点)。可达性:
00 页合并块的「底座路径覆盖」明文预期 ~/.cache/... 本地底座,此类 adapter
点页内评测按钮 → 子进程死于 hub 报错,且**重试救不了**(每次重试解析出
同一个 ~ 路径)。R113 obs-1 登记,本轮收口。

钉型:纯函数钉(免 torch——重依赖全在 _load 内,helper 模块级只依赖
json/pathlib,与 models.py 现风格一致)。原 _load 语义逐句平移:
显式 base_model 优先 → 依序找第一个**存在**的 config 文件(adapter_config.json
优先于 config.json;存在即 break,无论是否解析出键)→ 键取
base_model_name_or_path 回落 _name_or_path → 解析不出返回 None,
ValueError 留在 _load 原位。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from domains.medical_entity.eval.models import resolve_adapter_base

MODELS_SRC = (
    Path(__file__).resolve().parents[2] / "domains" / "medical_entity" / "eval" / "models.py"
)


def _write_config(directory: Path, filename: str, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


class TestTildeExpansion:
    def test_adapter_config_tilde_base_expanded_to_home(self, tmp_path: Path) -> None:
        """~/.cache/... 本地底座(00 页底座覆盖明文预期的形态)必须展开成
        以 home 起头的绝对路径——未展开即 HF repo-id 拒收,评测子进程必死。"""
        adapter = tmp_path / "adapter"
        _write_config(
            adapter,
            "adapter_config.json",
            {"base_model_name_or_path": "~/hf-cache/models--Qwen"},
        )

        base = resolve_adapter_base(str(adapter))

        assert base is not None
        assert base.startswith(str(Path.home())), f"未展开 ~: {base!r}"
        assert "~" not in base
        assert base.endswith("hf-cache/models--Qwen")

    def test_explicit_base_model_wins_and_expands(self, tmp_path: Path) -> None:
        """显式 base_model 参数优先于 adapter 目录里的任何 config(--base-model
        旗标的覆盖语义),且同样做 ~ 展开。"""
        adapter = tmp_path / "adapter"
        _write_config(adapter, "adapter_config.json", {"base_model_name_or_path": "Qwen/X"})

        base = resolve_adapter_base(str(adapter), "~/models/explicit-base")

        assert base == str(Path.home() / "models" / "explicit-base")


class TestPassthrough:
    def test_hf_name_returned_unchanged(self, tmp_path: Path) -> None:
        """HF 名(Qwen/...)不含 ~,expanduser 必须是 no-op——极性钉:修复
        不得把合法 HF 名改形。"""
        adapter = tmp_path / "adapter"
        _write_config(
            adapter,
            "adapter_config.json",
            {"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"},
        )

        assert resolve_adapter_base(str(adapter)) == "Qwen/Qwen2.5-0.5B-Instruct"


class TestFallbackChain:
    def test_config_json_name_or_path_used_when_no_adapter_config(self, tmp_path: Path) -> None:
        """合并导出产物(config.json + _name_or_path,无 adapter_config)是
        第二候选源;键回落链与原 _load 一致。"""
        merged = tmp_path / "merged"
        _write_config(merged, "config.json", {"_name_or_path": "~/models/merged-base"})

        base = resolve_adapter_base(str(merged))

        assert base == str(Path.home() / "models" / "merged-base")

    def test_empty_directory_returns_none(self, tmp_path: Path) -> None:
        """解析不出返回 None——ValueError 文案留在 _load 原位(调用方职责)。"""
        assert resolve_adapter_base(str(tmp_path / "empty")) is None

    def test_no_fallthrough_when_adapter_config_lacks_keys(self, tmp_path: Path) -> None:
        """break 语义钉(r114-reviewer nit-2 采纳):adapter_config.json 存在但
        两键皆缺时**不回落** config.json——第一存在文件即止是原 _load 语义
        (reviewer 活探针实证),未来改成「第一个解析出 base 的文件」会让
        此钉变红。"""
        adapter = tmp_path / "adapter"
        _write_config(adapter, "adapter_config.json", {"r": 8})
        _write_config(adapter, "config.json", {"_name_or_path": "Qwen/X"})

        assert resolve_adapter_base(str(adapter)) is None


class TestWiring:
    def test_load_calls_resolve_adapter_base(self) -> None:
        """接线钉(r114-reviewer nit-1 采纳):_load 必须经 helper 解析底座——
        防「保留 helper、_load 重内联旧循环」的部分解线逃过全部纯函数钉。"""
        src = MODELS_SRC.read_text(encoding="utf-8")
        m = re.search(r"def _load\(self\):[\s\S]*?(?=\n    def |\n\nclass |\Z)", src)
        assert m is not None, "RealFinetunedModel._load 必须在场"
        assert "resolve_adapter_base(self.model_path, self.base_model)" in m.group(0), (
            "_load 必须调用 resolve_adapter_base(抽取不得被部分回退)"
        )
