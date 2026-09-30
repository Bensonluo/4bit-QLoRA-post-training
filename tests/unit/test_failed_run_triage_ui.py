"""00 页失败 run 非专家急救包（R124）：红徽章之后不能是死胡同。

选点依据：下一步面板门 `status == "finished"` 把失败 run 排除——失败是
非专家最需要帮助的时刻，却只有红徽章 + 日志 expander，无「为什么失败/
怎么重试」指引。本文件钉两层：
① 纯函数层：_diagnose_failure 签名匹配（ast 沙箱提取，免整页 exec 的
   Streamlit 副作用——00 页模块级有 `with tab_activity:` 会真渲染整页）；
② 接线层：AppTest 造 failed run（returncode 1 + OOM 日志），急救包
   expander 内的 markdown 必须带出匹配诊断与重试指路。

对策口径与 CLAUDE.md OOM 恢复一致（降 max_length / 降 LoRA r / 换小模型），
签名表为通用训练失败设计，不绑任何域（评测行另有重试指路分支）。
"""

import ast
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_LAB = ROOT / "ui" / "pages" / "00_Training_Lab.py"


def _load_triage() -> tuple[list, object]:
    """ast 沙箱提取 _FAILURE_TRIAGE + _diagnose_failure（只 exec 这两个
    顶层定义，页面其余模块级代码不跑）。提取失败即红（结构变更须连测试改）。"""
    tree = ast.parse(PAGE_LAB.read_text(encoding="utf-8"))
    wanted: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_diagnose_failure":
            wanted.append(node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # _FAILURE_TRIAGE 是带注解赋值（AnnAssign），普通 Assign 分支抓不到
            if node.target.id == "_FAILURE_TRIAGE":
                wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_FAILURE_TRIAGE" for t in node.targets
        ):
            wanted.append(node)
    assert len(wanted) == 2, "页内 _FAILURE_TRIAGE / _diagnose_failure 必须都在场"
    ns: dict = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(PAGE_LAB), "exec"), ns)
    return ns["_FAILURE_TRIAGE"], ns["_diagnose_failure"]


def test_diagnose_matches_oom_signature_case_insensitive():
    """OOM 签名钉：真实 traceback 大小写（OutOfMemoryError / CUDA out of
    memory）小写化后必须命中「显存/内存不足」。"""
    _, diagnose = _load_triage()
    hits = diagnose("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert [d for d, _ in hits] == ["显存/内存不足"], hits
    hits2 = diagnose("torch.OutOfMemoryError: CUDA error OOM")
    assert hits2, "OutOfMemoryError 驼峰写法同样命中"


def test_diagnose_matches_data_and_multi_signatures_in_order():
    """数据路径签名 + 多命中按表序钉：一条日志同时命中 OOM 与依赖问题时，
    返回顺序必须与签名表一致（表序即产品优先序）。"""
    triage, diagnose = _load_triage()
    hits = diagnose("FileNotFoundError: outputs/wizard/x/train.json does not exist")
    assert [d for d, _ in hits] == ["文件/数据路径问题"], hits
    both = diagnose("ImportError: bitsandbytes; CUDA out of memory")
    assert [d for d, _ in both] == ["显存/内存不足", "依赖/环境问题"], (
        f"多命中必须按签名表序:{[d for d, _ in both]}"
    )
    assert len(triage) == 4, f"签名表四类(显存/数据/依赖/网络)不得静默增删:{len(triage)}"


def test_diagnose_clean_or_empty_log_returns_no_hits():
    """无命中钉：正常日志与空日志（无日志文件 → read_recent_logs 返回 ""）
    都必须返回空列表，走未命中分支的通用三步。"""
    _, diagnose = _load_triage()
    assert diagnose("") == []
    assert diagnose("epoch 1: train_loss=2.31, eval_loss=2.44\nsaved checkpoint") == []


def _stage_run(tmp_path: Path, returncode: int, log_text: str | None) -> None:
    """复刻 test_next_steps_eval_button_ui 的 staging 法：.run_meta.json +
    config yaml + 日志文件；returncode 0→finished、1→failed。"""
    out = tmp_path / "outputs" / "run-x"
    out.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "outputs" / "configs" / "run-x.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "model:\n"
        "  name: Qwen/Qwen2.5-1.5B-Instruct\n"
        "training:\n"
        f"  output_dir: {out}\n"
        "data:\n"
        "  dataset_name: yahma/alpaca-cleaned\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "outputs" / "logs" / "run-x.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_text is not None:
        log_path.write_text(log_text, encoding="utf-8")
    (tmp_path / "outputs" / ".run_meta.json").write_text(
        json.dumps(
            {
                "run-x": {
                    "technique": "sft",
                    "config_path": str(cfg),
                    "log_path": str(log_path),
                    "pid": 123,
                    "start_time": 0.0,
                    "returncode": returncode,
                }
            }
        ),
        encoding="utf-8",
    )


def _boot_lab(monkeypatch, tmp_path: Path):
    from streamlit.testing.v1 import AppTest

    import ui.config

    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    page = AppTest.from_file(str(PAGE_LAB), default_timeout=30)
    page.run()
    return page


def test_triage_block_renders_matched_hint_for_failed_run(tmp_path, monkeypatch):
    """接线钉（主旅程）：failed run + OOM 日志 → 急救包 markdown 带出
    「从最近日志匹配到可能原因」+ 匹配诊断 + 重试指路（配置页）。"""
    _stage_run(
        tmp_path,
        1,
        "Traceback (most recent call last):\n"
        "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB\n",
    )
    page = _boot_lab(monkeypatch, tmp_path)
    assert not page.exception, [e.message for e in page.exception]
    md = [m.value for m in page.markdown]
    assert any("从最近日志匹配到可能原因" in v for v in md), "命中态标题必须在场"
    assert any("显存/内存不足" in v and "LoRA r 16→8" in v for v in md), "匹配诊断与对策必须在场"
    assert any("回到「配置」标签页即可重新启动" in v for v in md), "重试指路必须在场"


def test_triage_absent_for_finished_run_and_expander_label_pinned(tmp_path, monkeypatch):
    """极性钉：finished run 不渲染急救包（急救包专属失败态）；expander
    标签「🩹 失败诊断与重试」源码钉（AppTest 树不暴露 expander 标签）。"""
    _stage_run(tmp_path, 0, "epoch 1 done")
    page = _boot_lab(monkeypatch, tmp_path)
    assert not page.exception, [e.message for e in page.exception]
    md = [m.value for m in page.markdown]
    assert not any("失败诊断与重试" in v or "匹配到可能原因" in v for v in md), (
        "finished run 不得渲染急救包"
    )
    source = PAGE_LAB.read_text(encoding="utf-8")
    assert 'st.expander("🩹 失败诊断与重试")' in source, "急救包 expander 标签钉"
    assert 'if status == "failed":' in source, "急救包必须以 failed 状态门控"
