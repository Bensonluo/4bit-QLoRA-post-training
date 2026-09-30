"""00 下一步面板评测命令钉：R112 诚实限界 → R120 通用评测闭环换锚。

史（R112）：面板曾向 Wizard 数据训练完的用户推荐
`scripts/evaluate.py --model-path X --test-file Y`——--test-file 根本不是
该脚本（typer CLI）的合法选项，argparse/typer 即死；当年修正为
--dataset 形态并加「输出只在终端显示、不会出现在评测结果/模型对比两页」
的诚实限界 caption + 领域命令指路。

R120 换锚：通用评测闭环落地——wizard test 门内主路为「⚡ 用我的 test 集
评测」按钮（launch_entity_eval）+ 终端等价命令
`scripts/eval_entity_match.py --model-path X --test-file Y`（argparse
真实选项，结果落 domains/entity_matching/data/results/ 自动点亮 02/03
两页）。R112 三钉随旧块删除全部失锚——R120 定向集不含本文件，潜伏至
R121 全量义务轮才被抓（定向策略盲区的又一实证，轮报如实披露）。

R121 重锚三钉：
① 新命令块特征化（--model-path {arts.output_dir} + --test-file
   {_wizard_test} 插值，按钮主、命令辅双路惯例）；
② CLI 契约钉平移（面板旗标 ⊆ scripts/eval_entity_match.py argparse
   真实选项面——R111 当场抓 --test-file 坏旗标的机制在新命令上延续）；
③ 旧限界句反钉——「只在终端显示/不会出现在」必须退场：新命令的结果
   就进两页，旧边界句在新世界是谎言而非诚实；旧 evaluate.py 命令
   同步反钉（防回归到 typer 即死的坏命令）。
"""

import ast
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_LAB = ROOT / "ui" / "pages" / "00_Training_Lab.py"
GENERIC_EVAL_CLI = ROOT / "scripts" / "eval_entity_match.py"


def _source() -> str:
    return PAGE_LAB.read_text(encoding="utf-8")


def test_wizard_eval_command_block_characterization():
    """新命令块特征化钉：st.code 双 f-string 保持
    `scripts/eval_entity_match.py` + --model-path {arts.output_dir} +
    --test-file {_wizard_test} 插值——按钮（launch_entity_eval）与终端
    命令两条路指向同一 CLI，特征化防命令重写后契约钉②静默脱落。"""
    source = _source()
    assert 'f"python scripts/eval_entity_match.py "' in source, (
        "面板必须有通用评测终端等价命令（按钮主、命令辅）"
    )
    assert '"--model-path {arts.output_dir} --test-file {_wizard_test}"' in source, (
        "命令必须插值 output_dir 与 wizard test 路径（与按钮同参）"
    )


def test_wizard_eval_command_flags_match_cli_surface():
    """CLI 契约钉（R112 机制平移到新命令）：面板推荐的旗标必须 ⊆
    scripts/eval_entity_match.py 的 argparse 真实选项面——R111 的
    --test-file 坏旗标（typer 即死）事故在新命令上同样当场可抓，
    UI 推荐命令与 CLI 契约的跨文件对齐不靠人肉。ast 提取
    add_argument("--xxx") 首参常量，免重依赖导入（torch 链不进测试）。"""
    cli = GENERIC_EVAL_CLI.read_text(encoding="utf-8")
    opts: set[str] = set()
    for node in ast.walk(ast.parse(cli)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                for a in node.args:
                    if (
                        isinstance(a, ast.Constant)
                        and isinstance(a.value, str)
                        and a.value.startswith("--")
                    ):
                        opts.add(a.value)
    assert opts, "eval_entity_match.py 选项面提取不得为空(提取器失效即红)"
    source = _source()
    m = re.search(r'f"--model-path \{arts\.output_dir\} --test-file \{_wizard_test\}"', source)
    assert m is not None, "面板评测命令块(特征化钉①的锚)必须在场"
    flags = set(re.findall(r"--[a-z][a-z-]*", m.group(0)))
    assert flags, "命令旗标提取不得为空(提取器失效即红)"
    assert flags <= opts, f"UI 推荐旗标越出 CLI 真实选项面: {sorted(flags - opts)}"


def test_obsolete_terminal_only_caption_retired():
    """反钉：R112「只在终端显示/不会出现在两页」限界句必须退场——R120
    起通用评测结果就落 domains/entity_matching/data/results/，两页自动
    可见；旧边界句在新世界是谎言而非诚实。旧 evaluate.py 命令（typer
    即死的坏命令源头）一并反钉，防静默回归。"""
    source = _source()
    assert "只在终端显示" not in source, "旧「只在终端」限界句必须退场"
    assert "scripts/evaluate.py" not in source, (
        "旧 evaluate.py 推荐命令必须退场（由 eval_entity_match.py 取代）"
    )
