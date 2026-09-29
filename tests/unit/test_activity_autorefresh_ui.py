"""Activity 标签自动刷新(R104):训练进行中,状态/日志/曲线随节拍自己走,
不要求非专家用户守着页面反复点 🔄 刷新。

选点依据(ux-scout 审计+官方文档核实):st.fragment 自 1.37.0 稳定,本机
1.57.0 实测无 parallel 参数;run_every 无数据驱动的停止条件,所以自动刷新
只能在全量重跑时条件应用(有活跃训练才挂节拍,空闲页不空转)。

覆盖边界:AppTest 不模拟 run_every 定时触发,自动刷新行为本身无法用
旅程测试验证——与 R103 spinner 同理,用源码扫描钉锁代码形态:
① Activity 全量渲染收进 _render_activity,fragment 边界必须包含 runner
   构造与 run 列表读取(否则 fragment 节拍内永远看不到新启动的 run);
② 条件应用形状 + run_every="30s"(与 ui/queries.py 的 30s TTL 缓存对齐,
   更短只会放大 TTL 过期时的整段重查卡顿);
③ 页内所有 st.rerun() 保持全应用 scope(Stop/Delete/刷新/合并完成的既有
   语义零漂移——scope="fragment" 会让它们只重跑局部);
④ 禁用 parallel=(本机 1.57.0 无此参数,写了即 TypeError);
⑤ pyproject 地板覆盖 fragment 稳定版本(>=1.37.0)。
外加一条合成应用的 AppTest 卫兵钉:证明「运行时条件应用 fragment」这一
形态在 1.57.0 下能被 AppTest 正常渲染与交互(issue #9242 的「不兼容」
说法已被本机实测推翻;若未来升级 Streamlit 破坏该兼容,此钉变红示警),
以及一条真页旅程钉:预置活 pid 的 .run_meta.json 让生产页面本体走真实
fragment 应用分支渲染(R104 审查 nit-1/2/4 采纳后的加固形态)。
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

ROOT = Path(__file__).resolve().parents[2]
PAGE_LAB = ROOT / "ui/pages/00_Training_Lab.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _activity_body(source: str) -> str | None:
    r"""提取 _render_activity 的缩进块(止于首个列 0 行):钉必须锚定块内。
    惰性 [\s\S]*? 全文扫描可被底部探测块的 TrainingRunner( 满足——
    「runner 构造提升出函数体」的绕过不受罚(nit-1)。"""
    match = re.search(
        r"^def _render_activity\(\) -> None:\n((?:    [^\n]*\n|\n)+)",
        source,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def test_activity_body_is_single_render_function():
    """Activity 全量渲染(含 runner 构造与 run 列表读取)收进单一函数:
    fragment 边界漏掉这两行的话,fragment 节拍内永远看不到新启动/新删除
    的 run——每 tick 必须重读 .run_meta.json(成本≈一次 JSON 读)。"""
    body = _activity_body(_source(PAGE_LAB))
    assert body is not None, "Activity 渲染体必须是独立函数 _render_activity"
    assert "runner = TrainingRunner(" in body, (
        "runner 构造必须在函数体内(fragment tick 重读 .run_meta.json 的前提)"
    )
    assert "list_all_runs()" in body, "run 列表读取必须在函数体内(fragment 边界内)"
    assert "for run_id in reversed(" in body, "逐 run 渲染循环必须在函数体内(fragment 边界内)"


def test_fragment_applied_conditionally_at_30s():
    """条件应用:仅当存在活跃训练时挂 run_every 节拍,空闲时裸调;
    间隔锁 "30s"——与 metric 查询的 30s TTL 缓存同拍,更短只会让 TTL
    过期时的整段重查卡顿更频繁。"""
    source = _source(PAGE_LAB)
    assert re.search(
        r"if [^\n]*list_active\(\):\n"
        r'\s+st\.fragment\(run_every="30s"\)\(_render_activity\)\(\)\n'
        r"\s+else:\n"
        r"\s+_render_activity\(\)",
        source,
    ), 'Activity 必须按 list_active() 探测条件应用 st.fragment(run_every="30s")'


def test_all_reruns_stay_app_scope():
    """Stop/Delete/🔄 刷新/合并完成的 st.rerun() 必须保持全应用 scope:
    它们要么改变 gating 探测结果(Stop 之后无活跃 run,fragment 应退场),
    要么重渲染 fragment 外的页面区域。"""
    source = _source(PAGE_LAB)
    assert "st.rerun(scope=" not in source, (
        "00 页不应出现 scope= 形式的 st.rerun(app scope 是默认且唯一合法语义)"
    )


def test_no_unsupported_fragment_options():
    """本机 Streamlit 1.57.0 的 st.fragment 无 parallel 参数(官方文档页
    描述的是更新版本)——写了即 TypeError。"""
    source = _source(PAGE_LAB)
    assert "parallel=" not in source, "st.fragment 不携带 parallel=(1.57.0 未实现该参数)"


def test_streamlit_floor_covers_fragment():
    """pyproject 地板必须 >= fragment 稳定版本 1.37.0(此前声明的 1.36.0
    不含稳定 st.fragment)。"""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"streamlit>=(?:1\.3[7-9]|1\.[4-9]|[2-9])", pyproject), (
        "pyproject 的 streamlit 地板须抬到 >=1.37.0(st.fragment 稳定版);"
        "亦接受 2.x+ 地板,防未来升级时此钉假红(nit-4)"
    )


def test_runtime_applied_fragment_renders_under_apptest(tmp_path):
    """合成应用卫兵钉:与生产同形的「运行时应用 fragment」(st.fragment(
    run_every=...)(fn)() 而非装饰器语法)在 AppTest 下必须能渲染、能交互。
    1.57.0 实测 #9242 不适用;此钉在 Streamlit 升级破坏兼容时变红示警。
    run_every 的定时触发本身不在 AppTest 能力内(见模块 docstring 边界)。"""
    from streamlit.testing.v1 import AppTest

    app = tmp_path / "fragment_app.py"
    app.write_text(
        "import streamlit as st\n"
        "\n"
        "\n"
        "def _panel():\n"
        '    st.write("activity-here")\n'
        '    if st.button("hit"):\n'
        '        st.session_state["hit"] = True\n'
        '    if st.session_state.get("hit"):\n'
        '        st.write("clicked")\n'
        "\n"
        "\n"
        'st.fragment(run_every="30s")(_panel)()\n',
        encoding="utf-8",
    )
    page = AppTest.from_file(str(app), default_timeout=20)
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    assert any("activity-here" in w.value for w in page.markdown), "fragment 函数体必须渲染进主区域"
    page.button[0].click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert any("clicked" in w.value for w in page.markdown), "fragment 内按钮交互必须保持可用"


def test_lab_page_boots_with_empty_activity(tmp_path, monkeypatch):
    """重构后的 00 页整页启动渲染旅程钉:空活动态(无任何 run)走裸调
    _render_activity() 分支——探测块、fragment 条件应用与渲染体在真实
    AppTest 渲染中必须无异常执行,空态文案如实出现。PROJECT_ROOT 经
    monkeypatch 替换(R103 05 页旅程钉先例:AppTest 内 from-import 会读
    到已被替换的 ui.config 属性)。"""
    from streamlit.testing.v1 import AppTest

    import ui.config

    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    page = AppTest.from_file(str(PAGE_LAB), default_timeout=30)
    page.run()
    assert not page.exception, [e.message for e in page.exception]
    assert any("No training runs yet" in i.value for i in page.info), (
        "空活动态必须如实渲染『No training runs yet』提示(裸调分支真实执行)"
    )


def test_lab_page_renders_fragment_branch_with_active_run(tmp_path, monkeypatch):
    """真页 fragment 分支旅程钉(nit-2 采纳):预置含活 pid 的 .run_meta.json,
    _probe.list_active() 为真 → 生产页面走真实条件应用分支 st.fragment(
    run_every="30s")(_render_activity)(),活跃 run 以 🟢 Running 呈现。
    合成卫兵钉只证「fragment 形态在 AppTest 下可渲染」,本钉把该证明落到
    生产页面本体(活 pid 无 returncode = UI 重启恢复路径,读路径全真实)。"""
    import json
    import subprocess
    import sys

    from streamlit.testing.v1 import AppTest

    import ui.config

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / ".run_meta.json").write_text(
        json.dumps(
            {
                "probe-live": {
                    "technique": "sft",
                    "pid": proc.pid,
                    "log_path": str(outputs / "logs" / "probe-live.log"),
                    "config_path": str(outputs / "configs" / "probe-live.yaml"),
                    "start_time": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    try:
        page = AppTest.from_file(str(PAGE_LAB), default_timeout=30)
        page.run()
        assert not page.exception, [e.message for e in page.exception]
        assert any("probe-live" in w.value for w in page.markdown), (
            "活跃 run 必须在真页 fragment 应用分支中渲染出来"
        )
        assert any("Running" in m.value for m in page.metric), (
            "活 pid(无 returncode)必须呈现 🟢 Running(UI 重启恢复路径)"
        )
    finally:
        proc.terminate()
        proc.wait()
