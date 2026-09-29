"""加载反馈一致性(R103):重本地操作必须给用户可见的进行中反馈。

Streamlit 的 spinner 是瞬态元素——脚本跑完就消失,AppTest 的最终元素树里
不可观测,所以这里用源码扫描钉(R97/R102 先例):锁住「重调用必须包在
with st.spinner(...) 里」的代码形态,防止未来重构把包裹剥掉回到裸跑。
外加一条 05 页演示按钮的真实旅程钉,证明包裹没有破坏控制流(st.error/
st.rerun 语义保持)。

选点依据(ux-scout 审计+主会话逐页核实):07 页已有 10 处 spinner 是惯例
基准;00-06 的重操作(数据集体检/生成训练集/MLflow 导入)与 07 页的 10 处
重按钮(方案准备/迭代准备/数据物化×2/全量验证×5/题集冻结——服务层是
tokenizer 加载、全量 token 预检、实体组分区、全记录 deepcopy)全部裸跑,
用户在这些秒级等待期间看到的是无响应的页面。

覆盖边界:只钉「用户主动触发的阻塞操作」(按钮点击 → 多秒等待)。页面
渲染路径的读取(01 fetch_metric_history 有 30s 缓存、03 load_eval_data)
不加 spinner——渲染路径加 spinner 会在每次无关重跑闪现,属 R104 自动
刷新的领地,不在本钉范围。
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

UI = Path(__file__).resolve().parents[2] / "ui"
PAGE_WIZARD = UI / "pages/05_Data_Wizard.py"
PAGE_LAB = UI / "pages/00_Training_Lab.py"
PAGE_EVAL = UI / "pages/02_Evaluation.py"
PAGE_INTAKE = UI / "pages/07_Data_Intake.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_wizard_generate_wrapped_in_spinner_naming_real_stages():
    """05 生成训练集(最重操作:候选构造→体检→切分→导出)必须有进行中反馈,
    标签如实点名真实阶段——不虚构进度条,不用无信息量的「处理中」。"""
    source = _source(PAGE_WIZARD)
    assert re.search(
        r'with st\.spinner\("正在生成训练集：候选构造 → 数据体检 → 防泄漏切分 → 导出…"\):\n'
        r"\s+report = WizardPipeline\(spec\)\.run\(table, out_dir\)",
        source,
    ), "生成训练集调用必须包在 st.spinner 内,标签逐字锁定(阶段列表点名真实管线)"


def test_wizard_import_paths_wrapped_in_spinner():
    """05 两条导入路径(上传/演示字节流 + 服务器路径读取)都包 spinner:
    大 Excel 解析可达数秒,裸跑时页面无响应。"""
    source = _source(PAGE_WIZARD)
    assert re.search(
        r'with st\.spinner\("正在读取表格…"\):\n\s+table = import_table\(path\)',
        source,
    ), "上传/演示路径的 import_table 必须包在 st.spinner 内"
    assert re.search(
        r'with st\.spinner\("正在读取表格…"\):\n\s+table = import_table\(server_path\.strip\(\)\)',
        source,
    ), "服务器路径读取的 import_table 必须包在 st.spinner 内"


def test_lab_dataset_preflights_wrapped_in_spinner():
    """00 两处数据集体检(手动「体检」按钮 + 提交时 SFT 强制预检)都包
    spinner:大 JSON 全量解析可达数秒,且提交路径此前完全无反馈。"""
    source = _source(PAGE_LAB)
    assert re.search(
        r'with st\.spinner\("正在体检数据集格式与样本…"\):\n'
        r"\s+pv_errors, pv_insp = check_dataset_for_sft\(pv_path\)",
        source,
    ), "手动体检按钮的 check_dataset_for_sft 必须包在 st.spinner 内"
    assert re.search(
        r'with st\.spinner\("正在体检数据集格式与样本…"\):\n'
        r"\s+ds_errors, ds_insp = check_dataset_for_sft\(dataset\)",
        source,
    ), "提交时 SFT 预检的 check_dataset_for_sft 必须包在 st.spinner 内"
    assert re.search(
        r'with st\.spinner\("正在读取样本预览…"\):\n'
        r"\s+pv_records, pv_err = load_preview_records\(",
        source,
    ), "体检通过后的样本预览读取与 check 同重量级,必须同包 spinner(r103-reviewer nit-1)"


def test_evaluation_mlflow_import_wrapped_in_spinner():
    """02 历史结果导入 MLflow(逐文件建 run,网络+DB 写入)包 spinner。同一
    「Scan & Import」操作在本页有三处入口(无域分支/无数据分支/页面底部),
    反馈必须三处同在——只包一处会让最常到达的底部入口反而裸跑。"""
    source = _source(PAGE_EVAL)
    assert re.search(
        r"with st\.spinner\([^)]*MLflow[^)]*\):\n\s+for domain_dir in DOMAINS_DIR\.iterdir\(\):",
        source,
    ), "MLflow 导入循环必须包在 st.spinner 内,标签点名 MLflow"
    assert (
        len(re.findall(r'with st\.spinner\("正在扫描并导入历史评测结果到 MLflow…"\):', source)) == 3
    ), "三处 Scan & Import 入口必须同标签同反馈"


def test_data_intake_heavy_buttons_wrapped_in_spinner():
    """07 核心旅程页 10 处重按钮全包 spinner(方案准备/迭代准备/数据物化×2/
    全量验证×5/题集冻结):服务层是 tokenizer 加载、全量 token 预检、实体组
    分区、全记录 deepcopy——生产规模下秒级等待,裸跑即页面无响应。快操作
    (confirm 类纯内存校验、start 类子进程 spawn)有意不包,包了是噪音。"""
    source = _source(PAGE_INTAKE)
    assert re.search(r"with st\.spinner\([^)]*\):\n\s+plans\.prepare\(plan_id", source), (
        "「确认推荐方案并准备训练」必须包 spinner"
    )
    assert re.search(r"with st\.spinner\([^)]*\):\n\s+iteration_service\.prepare\(", source), (
        "「按已确认范围准备下一轮训练」必须包 spinner"
    )
    assert (
        len(re.findall(r"with st\.spinner\([^)]*\):\n\s+service\.materialize_dataset\(", source))
        == 2
    ), "两处数据物化(迭代题集版/生成数据集版本)都必须包 spinner"
    assert (
        len(re.findall(r"with st\.spinner\([^)]*\):\n\s+service\.validate_full_sources\(", source))
        == 2
    ), "两处全量组合验证都必须包 spinner"
    assert (
        len(re.findall(r"with st\.spinner\([^)]*\):\n\s+service\.validate_full_data\(", source))
        == 3
    ), "三处全量验证(首次上传/演示全量/表单上传)都必须包 spinner"
    assert re.search(
        r"with st\.spinner\([^)]*\):\n\s+reference = suite_service\.freeze\(session\)", source
    ), "「固定当前开发与测试题集」必须包 spinner"
    assert len(re.findall(r"with st\.spinner\(", source)) >= 20, (
        "07 页 spinner 总数下限 20(10 既有 + 10 本轮)——只增不减"
    )


def test_wizard_demo_button_journey_still_loads_table(tmp_path, monkeypatch):
    """包裹后的 05 页控制流不回归:点演示数据按钮,真实 import_table 路径
    (含 spinner 包裹)执行后表格加载成功——st.error/st.rerun 语义保持。
    PROJECT_ROOT 经 monkeypatch 替换(07 页 data_page 夹具先例)。"""
    from streamlit.testing.v1 import AppTest

    import ui.config

    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)
    page = AppTest.from_file(str(PAGE_WIZARD), default_timeout=20)
    page.run()
    assert not page.exception
    demo = next(b for b in page.button if "医疗演示数据" in b.label)
    demo.click().run()
    assert not page.exception, [e.message for e in page.exception]
    assert any("已加载" in s.value for s in page.success), [s.value for s in page.success]


def test_wizard_entry_pointer_names_real_buttons():
    """05 空态指路句必须点名真实存在的按钮(r108-scout 发现):旧句点名
    「试试演示数据」,实际按钮是「🧪 医疗演示数据」「🏭 主数据演示数据」
    ——与 R106 轮 00:189 预设按钮失配同类(指路牌指向不存在的门,
    非专家照指路找按钮找不到)。"""
    source = _source(PAGE_WIZARD)
    assert "或点两个演示数据按钮之一" in source, "指路句必须点名真实按钮"
    assert "「试试演示数据」" not in source, "幽灵按钮名必须退场"


def test_wizard_handoff_send_side_wired():
    """05→00 页内交接·发送侧源码钉(R107 审查发现零覆盖):预填
    dataset_input + 写 wizard_handoff + switch_page 三件套缺一不可,
    少一样交接横幅/预填静默失效。"""
    source = _source(PAGE_WIZARD)
    assert 'st.session_state["dataset_input"] = str(train_path)' in source
    assert 'st.session_state["wizard_handoff"]' in source
    assert 'st.switch_page("pages/00_Training_Lab.py")' in source


def test_wizard_handoff_receive_side_wired():
    """交接·接收侧源码钉:00 页必须 pop 横幅数据并渲染已预填说明——
    发送侧三件套与接收侧 pop/渲染两端钉合,才是完整契约。"""
    source = _source(PAGE_LAB)
    assert 'st.session_state.pop("wizard_handoff", None)' in source
    assert "已从 **Data Wizard** 预填" in source
