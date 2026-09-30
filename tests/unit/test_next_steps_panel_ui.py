"""00 下一步面板评测命令诚实限界(R112):坏命令修正 + 边界说明。

选点依据(r112-scout 裁决采纳,R111 轮报登记候选④的微切形态;完整
按钮式桥接经 scout 测评为 MEDIUM 改动,登记 R113 主切):00「🧭 下一步」
给 Wizard 数据训练完的用户推荐的评测命令原本是
`python scripts/evaluate.py --model-path X --test-file Y`——r112-reviewer
中途发现并经本轮一手复核:--test-file 根本不是该脚本的合法选项
(scripts/evaluate.py 是 typer CLI,选项仅 --model-path/--dataset/
--max-samples/--num-generations/--compare-with/--output),原命令在
argparse 即死(实证:No such option)。非「能跑但只进终端」的假朋友,
是彻底跑不起来的坏命令。本轮:命令修正为 --dataset 形态(吃本地
Alpaca JSON,loaders.py:100-105 HF 回落 load_dataset("json");实证
--model-path/--dataset 组合能过解析、死在路径检查即选项全被接受),
并在命令块后加诚实限界 caption——修好的命令输出仍只进终端
(scripts/evaluate.py 不写 domains/*/data/results/,R111 诚实红线),
指路真正能点亮「评测结果 / 模型对比」两页的领域评测命令。

领域命令形态一手核实(本轮直读,非抄 scout):
- domains/medical_entity/evaluate.py:119 `--model-path` 在场;
- 底座自动解析:eval/models.py 的 resolve_adapter_base()（RealFinetunedModel._load
  调用）未传 --base-model 时读 adapter 目录 adapter_config.json/config.json 的
  base_model_name_or_path,解析不出才 ValueError——命令无需配底座;
- 结果确实落页:evaluate.py:194 无条件 save_results(reports) →
  report.py:166 写 eval_detail_*.json 进 DOMAIN_ROOT/data/results/
  (report.py:12),正是 02/03 两页唯一数据源;
- 不诚实桥接已避坑:Wizard Alpaca test.json 不符 run_evaluation 候选
  schema(scout R113 成本档案),caption 不得展示领域命令吃 --test-file
  传 Wizard 文件的形态——领域评测用领域自己的测试集。

钉型裁量:源码钉。面板渲染需真实训练产物 fixture(runs 状态+adapter+
eval_sets 三重前置),与一段 caption 的保障不成比例;R111 obs-1 同裁量。
两钉:①限界 caption 必须落在评测命令块与注册段之间的面板区间,同时含
「只在终端」边界、两页点名、领域命令指路,且整个面板区间不得再出现
--test-file(坏旗标不得回归);②命令块特征化锚定(生而绿,披露:
--model-path 与 --dataset 插值在场),防钉①锚点静默脱落。
"""

import ast
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_LAB = ROOT / "ui" / "pages" / "00_Training_Lab.py"


def _source() -> str:
    return PAGE_LAB.read_text(encoding="utf-8")


def test_next_steps_eval_command_has_honesty_boundary_caption():
    """限界 caption 钉:评测命令块之后、注册段之前,caption 必须在场且
    四要素齐全——「只在终端」边界 + 两页点名 + 领域命令指路;且整个
    面板区间不得再出现 --test-file(r112-reviewer 发现的坏旗标,
    argparse 即死,不得回归)。r112-reviewer nit-1 采纳登记:domains/
    medical_entity/evaluate.py 自有合法 --test-file,未来面板文案若需
    合法提及它,届时连本断言一起改——当前区间无此形态,禁令利大于弊。"""
    source = _source()
    assert "scripts/evaluate.py --model-path" in source, "评测命令块必须在场"
    block = source.split("scripts/evaluate.py --model-path", 1)[1]
    # 面板区间锚:评测命令块到注册段之间(caption 挪到区间外即脱锚)
    scope_end = block.find("合并导出 / 注册")
    assert scope_end != -1, "面板结构锚(合并导出段)必须在场"
    panel = block[:scope_end]
    assert "--test-file" not in panel, "坏旗标 --test-file 不得回归面板(argparse 即死)"
    cap_pos = panel.find("st.caption(")
    assert cap_pos != -1, "评测命令后必须有诚实限界 caption(R112)"
    caption = panel[cap_pos : cap_pos + 600]
    assert "终端" in caption, "caption 必须说明输出只在终端"
    # 极性钉(r112-reviewer nit-2 采纳):token 在场≠方向对——逐字锚边界
    # 短语,倒置改写(「结果会出现在两页」)保 token 也过不了本钉
    assert "只在终端显示" in caption and "不会出现在" in caption, (
        "caption 必须逐字锚定「只在终端显示/不会出现在」边界方向"
    )
    assert "评测结果" in caption and "模型对比" in caption, (
        "caption 必须点名两页边界(命令结果不进这两页)"
    )
    assert "domains.medical_entity.evaluate" in caption, (
        "caption 必须给出领域评测命令指路(真正能点亮两页的路)"
    )


def test_next_steps_eval_command_block_characterization():
    """命令块特征化钉(生而绿,披露):st.code 插值 --model-path
    {arts.output_dir} 与 --dataset {arts.eval_sets['test']} 保持——
    钉①锚点依赖命令块在场,此钉防命令重写后锚点静默脱落;
    --dataset 是 scripts/evaluate.py 唯一吃本地数据集的合法选项。"""
    source = _source()
    assert "scripts/evaluate.py --model-path {arts.output_dir}" in source, (
        "评测命令必须保持 output_dir 插值"
    )
    assert "--dataset {arts.eval_sets['test']}" in source, (
        "评测命令必须以 --dataset 挂 wizard test 集(合法选项)"
    )


def test_next_steps_eval_command_flags_match_cli_surface():
    """CLI 契约钉(r112-reviewer E 项建议随轮采纳):面板 st.code 推荐的
    scripts/evaluate.py 旗标必须 ⊆ 脚本真实选项面——本钉在 R111 就能
    当场抓住 --test-file(argparse 即死),UI 推荐命令与 CLI 契约的跨文件
    对齐不再靠人肉。ast 解析提取 typer.Option("--xxx") 位置字符串,
    免重依赖导入(torch 链不进测试)。"""
    cli = (ROOT / "scripts" / "evaluate.py").read_text(encoding="utf-8")
    opts: set[str] = set()
    for node in ast.walk(ast.parse(cli)):
        if isinstance(node, ast.Call):
            for a in node.args:
                if (
                    isinstance(a, ast.Constant)
                    and isinstance(a.value, str)
                    and a.value.startswith("--")
                ):
                    opts.add(a.value)
    assert opts, "scripts/evaluate.py 选项面提取不得为空(提取器失效即红)"
    source = _source()
    cmd = re.search(r'f"python scripts/evaluate\.py ([^"]*)"\s*f"([^"]*)"', source)
    assert cmd is not None, "面板评测命令块(st.code 双 f-string)必须在场"
    flags = set(re.findall(r"--[a-z][a-z-]*", f"{cmd.group(1)} {cmd.group(2)}"))
    assert flags, "命令旗标提取不得为空(提取器失效即红)"
    assert flags <= opts, f"UI 推荐旗标越出 CLI 真实选项面: {sorted(flags - opts)}"
