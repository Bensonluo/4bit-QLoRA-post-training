"""00 下一步面板页内评测按钮(R113 主切):四态按钮 + launch_eval 接线。

选点依据(r113-scout 六问裁决采纳,R111 轮报登记候选④、R112 轮报定主切):
R112 只修好了终端回退命令(输出仍只进终端),非专家用户的「页内一键评测」
缺位——按钮的承诺「点亮评测结果/模型对比两页」在领域评测链上为真:
domains/medical_entity/evaluate.py:95 baseline 无条件 + :119 --model-path →
RealFinetunedModel + :194 save_results 无条件 → eval/report.py:166 写
eval_detail_*.json,02/03 页 load_eval_data 每 rerun glob 取最新
(domain_adapters.py:209-218)——「自动可看,无需导入」不撒谎。产物含
baseline+微调两模型,03 页 len(data)>=2 直接可用,不会落进 R111 刚分化的
len==1 单模型空态。

架构裁决(r113-scout B,一手复核成立):TrainingRunner 加 launch_eval()
方法,复用 _ACTIVE_PROCS_BY_ROOT/.run_meta.json/get_status/stop/delete 全套
——eval 行进训练同池,Activity 30s fragment 门(list_active 探测)免费看见
评测进度。cmd = python -m domains.medical_entity.evaluate --model-path X,
cwd=项目根(evaluate.py:72 相对路径读训练集,错 cwd 则 seen/unseen 静默
退化);无 --config(该脚本是 argparse 无此选项);env 注入 HF_ENDPOINT
镜像默认(底座 HF 名未缓存时的国内网络缓解)。

诚实红线对齐(R111/R112 同口径):
- 按钮不挂数用户 Wizard test.json(候选 schema 不符——domains/medical_entity/
  eval/runner.py:207 需 query/standard_name/code/candidates;勿与 src/tracking/
  runner.py 混淆,r113-reviewer nit-2),文案明说「领域自带测试集」(基线全量
  3,136 条已一手核实+采样 500 条);
- 失败态兜底 report.py:184-190 MLflow 后写洞(落盘后才试 log_eval_to_mlflow
  且只捕 ImportError——子进程可能退出码非 0 但产物已写、02 已点亮),
  失败文案须含「以页面为准」;
- R112 的终端命令 + 限界 caption 原样保留(按钮主、命令辅,两路都留),
  按钮块落 Chat 指路与 eval_sets 门之间——不依赖 wizard test 集存在。

钉型裁量:源码钉×3(接线/诚实文案/argparse 契约)+ 旅程钉×1(真实
AppTest 点按钮→launch_eval 以 (run_id, output_dir) 被调;launch_eval 是
薄方法级打桩点,无需重 fixture——R111 obs-1 的「合并按钮重 fixture」
裁量不适用)。旅程 fixture 用 R104 活 pid 先例的变体:returncode:0 预置
(状态走 finished → 面板渲染),PROJECT_ROOT→tmp(ui.config 属性替换,
R103 先例:AppTest 内 from-import 读到已替换值)。
"""

import ast
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_LAB = ROOT / "ui" / "pages" / "00_Training_Lab.py"
RUNNER = ROOT / "src" / "tracking" / "runner.py"
DOMAIN_EVAL = ROOT / "domains" / "medical_entity" / "evaluate.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_eval_button_block_wiring_and_placement():
    """接线钉:⚡ 主按钮(key=eval_btn_{run_id})调 launch_eval(run_id,
    output_dir);按钮块位置在 Chat 指路之后、eval_sets 门之前(不依赖
    wizard test 集存在);running 态按钮退场防双发;Activity 循环对
    medical_eval 行不渲染训练下一步面板(防 eval-of-eval 嵌套)。"""
    source = _source(PAGE_LAB)
    # 按钮块位置:Chat 指路 < 按钮 < eval_sets 门(R112 两钉锚区零破坏)
    chat_pos = source.find("想先直观感受效果")
    btn_pos = source.find('st.button("⚡ 生成评测文件"')
    gate_pos = source.find('if arts.eval_sets.get("test"):')
    assert chat_pos != -1, "Chat 指路锚必须在场"
    assert btn_pos != -1, "页内评测主按钮必须在场"
    assert gate_pos != -1, "eval_sets 门锚必须在场"
    assert chat_pos < btn_pos < gate_pos, "按钮块必须在 Chat 指路后、wizard test 门前"
    # 接线:launch_eval 以 (run_id, output_dir) 被调,主按钮形态
    assert "launch_eval(run_id, str(arts.output_dir))" in source
    assert 'key=f"eval_btn_{run_id}", type="primary"' in source
    # running 态按钮退场防双发:running 分支体内不得再有 st.button
    assert '_eval_status == "running"' in source
    running_block = source.split('_eval_status == "running"', 1)[1]
    running_block = running_block.split("elif", 1)[0]
    assert "st.button" not in running_block, "running 态不得渲染启动/重试按钮(防双发)"
    # Activity 循环门:eval 行(technique=medical_eval)不渲染训练下一步面板
    assert 'info.get("technique") != "medical_eval"' in source, (
        "00 页 Activity 循环必须排除 medical_eval 行(防 eval-of-eval 嵌套面板)"
    )


def test_eval_button_honest_state_copy():
    """诚实文案钉(四态):成功态两页点名+「无需导入」为真(load_eval_data
    每 rerun glob 取最新);失败态含「以页面为准」兜底(report.py:184-190
    MLflow 后写洞:退出码非 0 但 eval_detail 可能已落盘)与重试出路;
    running 态声明 30s 自动刷新(同池 fragment 门兑现);idle 态文案
    R114 起显式命名「医疗实体匹配」+「与你本次训练使用的数据无关」——
    r114-scout 裁决 obs-2 软化:finance/wizard 用户会把「领域自带测试集」
    误读成「我的领域」,显式命名同时服务两类受众;硬门落选(医疗训练
    不入 .run_meta.json 池,门=按钮对所有人隐身)。仍不得暗示吃用户
    Wizard 测试集(schema 不符)。"""
    source = _source(PAGE_LAB)
    # 四态分支骨架在场
    assert source.count("_eval_status ==") >= 3, (
        "评测四态分支(running/finished/failed/idle)必须在场"
    )
    # 成功态:两页点名 + 无需导入
    assert "评测完成——到「评测结果」「模型对比」页查看" in source
    assert "无需导入" in source
    # 失败态:以页面为准(report.py 后写洞兜底)+ 重试按钮同 key 回场
    assert "以页面为准" in source
    # running 态:30s 自动刷新声明(与 fragment 门同拍)
    assert "30 秒自动刷新" in source
    # idle 态(R114):显式命名评测任务 + 与训练数据无关的诚实锚。
    # 区间钉(r114-reviewer nit-3 采纳):锚定 idle 分支区间(页内评测标题
    # 到 eval_sets 门),不再全文级——防锚漂到其他分支后钉仍绿
    idle_start = source.find('st.markdown("**页内评测**')
    idle_end = source.find('if arts.eval_sets.get("test"):')
    assert idle_start != -1 and idle_end != -1 and idle_start < idle_end, (
        "idle 分支区间锚(页内评测标题 → eval_sets 门)必须在场"
    )
    idle_block = source[idle_start:idle_end]
    assert "医疗实体匹配" in idle_block, "idle 文案必须显式命名「医疗实体匹配」领域"
    assert "与你本次训练使用的数据无关" in idle_block, (
        "idle 文案必须声明评测用领域测试集、与用户本次训练数据无关"
    )
    assert "自带测试集" in idle_block  # 仍是领域自带集,不是用户 Wizard test.json
    assert "3,136" in idle_block and "500" in idle_block, (
        "规模数字与实测一致(基线全量 3,136+采样 500)"
    )


def test_stop_popover_copy_matches_row_technique():
    """obs-6 源码钉(R114 顺带):Stop popover 的确认文案按行类型分支——
    eval 行(technique=medical_eval)不得再说「训练进程/checkpoint」
    (评测进程无 checkpoint,错误名词诱导用户担心不存在的产物)。
    锚定 popover 区间:⏹ 停止 popover 起点与 🗑 删除 popover 起点之间,
    与 R113 Activity 门钉同型的区间钉。"""
    source = _source(PAGE_LAB)
    start = source.find('with st.popover("⏹ 停止"')
    end = source.find('st.popover("🗑 删除"')
    assert start != -1 and end != -1 and start < end, "Stop/Delete popover 锚必须在场"
    popover_block = source[start:end]
    # 分支在场:eval 行走专用文案
    assert 'info.get("technique") == "medical_eval"' in popover_block, (
        "Stop popover 必须按 medical_eval 分支文案"
    )
    assert "将终止该运行的评测进程" in popover_block, "eval 行文案必须说「评测进程」"
    assert "已写入的日志保留" in popover_block, "eval 行只承诺日志保留(无 checkpoint)"
    # 训练行原文案保留在另一分支;极性钉(r114-reviewer nit-3 采纳):
    # 条件 → eval 文案 → 训练文案的顺序锁死 if/else 极性,反转仍绿即红
    cond_pos = popover_block.find('info.get("technique") == "medical_eval"')
    eval_pos = popover_block.find("将终止该运行的评测进程")
    train_pos = popover_block.find("将终止该运行的训练进程")
    assert -1 < cond_pos < eval_pos < train_pos, (
        "分支极性:medical_eval 条件在先,eval 文案居 if 体,训练文案居 else 体"
    )


def test_runner_eval_flags_match_domain_cli_surface():
    """argparse 契约钉(R112 typer 契约钉平移,scout D-2③):runner.launch_eval
    构造的领域评测旗标必须 ⊆ domains/medical_entity/evaluate.py 真实
    argparse 选项面——UI 推荐命令与 CLI 契约的跨文件对齐不靠人肉,
    R112 --test-file(argparse 即死)同类事故在评测按钮上当场可抓。
    ast 提取 add_argument("--xxx") 首参常量,免重依赖导入。"""
    dom = _source(DOMAIN_EVAL)
    opts: set[str] = set()
    for node in ast.walk(ast.parse(dom)):
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
    assert opts, "domain evaluate.py 选项面提取不得为空(提取器失效即红)"
    runner_src = _source(RUNNER)
    m = re.search(r"def launch_eval\([\s\S]*?(?=\n    def |\nclass |\Z)", runner_src)
    assert m is not None, "launch_eval 方法必须在 runner.py 在场"
    flags = {f.strip('"') for f in re.findall(r'"--[a-z-]+"', m.group(0))}
    assert flags, "launch_eval 旗标提取不得为空(提取器失效即红)"
    assert flags <= opts, f"评测命令旗标越出领域 CLI 真实选项面: {sorted(flags - opts)}"


def test_eval_button_click_launches_eval(tmp_path, monkeypatch):
    """旅程钉:真实 AppTest 渲染 00 页(finished run + adapter 产物),
    下一步面板出现 ⚡ 按钮;点按 → TrainingRunner.launch_eval 以
    (源 run_id, 绝对 output_dir) 被调。方法级打桩(R104 教义的薄桩点:
    页面所有 from-import 拿到同一类对象),其余读路径全真实(preset
    .run_meta.json returncode:0 → finished → 面板渲染)。"""
    from streamlit.testing.v1 import AppTest

    import ui.config
    from src.tracking.runner import TrainingRunner

    out = tmp_path / "outputs" / "run-x"
    out.mkdir(parents=True)
    (out / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "adapter_model.safetensors").write_bytes(b"x")
    cfg_dir = tmp_path / "outputs" / "configs"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "run-x.yaml"
    cfg.write_text(
        "model:\n"
        "  name: Qwen/Qwen2.5-1.5B-Instruct\n"
        "training:\n"
        f"  output_dir: {out}\n"
        "data:\n"
        "  dataset_name: yahma/alpaca-cleaned\n"
        "logging:\n"
        "  use_mlflow: true\n",
        encoding="utf-8",
    )
    (tmp_path / "outputs" / ".run_meta.json").write_text(
        json.dumps(
            {
                "run-x": {
                    "technique": "sft",
                    "config_path": str(cfg),
                    "log_path": str(tmp_path / "outputs" / "logs" / "run-x.log"),
                    "pid": 123,
                    "start_time": 0.0,
                    "returncode": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    def _fake_launch(self, source_run_id: str, model_path: str, **_: object) -> str:
        calls.append((source_run_id, model_path))
        return f"eval-{source_run_id}"

    monkeypatch.setattr(TrainingRunner, "launch_eval", _fake_launch)
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)

    page = AppTest.from_file(str(PAGE_LAB), default_timeout=30)
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    btn = next((b for b in page.button if b.label == "⚡ 生成评测文件"), None)
    assert btn is not None, "finished run 的下一步面板必须渲染页内评测按钮"
    btn.click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert calls == [("run-x", str(out.resolve()))], (
        "点击必须以(源 run_id, 绝对 output_dir) 调 launch_eval,恰一次"
    )
