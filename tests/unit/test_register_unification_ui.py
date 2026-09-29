"""语域统一(R105):UI 面向非专家使用者,页面文案与 05/07 标杆页对齐为中文。

选点依据(r105-scout 三部分普查+主会话逐页复核):01/03/04 整页英文对非
专家构成功能性不可用级伤害(03 恰是决策者直达页);02 单次交互内中英乒乓
(英文按钮→中文 spinner→英文成功句)造成信任损耗;01:199 删除是全库唯一
「无确认+无 UI 恢复入口」的破坏性操作(mlflow 软删除入 .trash)。

术语边界:LoRA/SFT/MLflow/Champion/Challenger/Top-1 等领域术语与数据
枚举(FINISHED 在 KPI/表格经显示映射、easy/medium/hard 及实体类型值)
不译——语域统一≠术语翻译。难度枚举跨页(02 adapter 组件/03)一致保留
英文,连同 ui/components/domain_adapters.py 组件文案登记 R106 统一。

覆盖边界:语域是源码字符串层属性,用源码扫描钉(R97/R102/R103 先例)——
逐字锁关键句 + 旧英文串按「UI 调用形态」退场断言(锁 st.metric("Total
Runs" 而非裸词,避免误伤 MLflow 列名/代码注释);01 删除确认另加控制流
形状钉(确认按钮在 popover 内且先于 delete_run)。02 三入口一致性是 R103
登记-①闭环;spinner 标签逐字不动(test_loading_feedback_ui.py 既有钉,
此处冗余重锁防本轮误改)。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

UI = Path(__file__).resolve().parents[2] / "ui"
PAGE_EXP = UI / "pages/01_Experiments.py"
PAGE_CMP = UI / "pages/03_Model_Comparison.py"
PAGE_REG = UI / "pages/04_Registry.py"
PAGE_HOME = UI / "app.py"
PAGE_EVAL = UI / "pages/02_Evaluation.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_page_titles_and_headings_localized():
    """页面标题/browser 标签中文化(07 页「目标与数据 — TuneSmith」先例):
    非专家用户第一眼看到的页面名不能是英文——页面名是导航的地基。"""
    assert 'st.title("📊 实验记录")' in _source(PAGE_EXP)
    assert 'page_title="实验记录"' in _source(PAGE_EXP)
    assert 'st.title("⚖️ 模型对比")' in _source(PAGE_CMP)
    assert 'page_title="模型对比"' in _source(PAGE_CMP)
    assert 'st.title("🏛️ 模型注册表")' in _source(PAGE_REG)
    assert 'page_title="模型注册表"' in _source(PAGE_REG)
    hero = _source(PAGE_HOME)
    assert "先说业务目标，再看数据" in hero, "app 首页既有中文 hero 区不得漂移"


def test_experiments_kpi_table_and_filters_localized():
    """01 KPI 卡/筛选区/表头/详情区中文化,MLflow 状态枚举在显示层映射为
    中文(FINISHED→已完成);列名 key(tags.mlflow.runName 等)不动——那是
    数据契约,动的只有显示文案。"""
    source = _source(PAGE_EXP)
    for label in ("运行总数", "已完成", "运行中", "失败"):
        assert f'st.metric("{label}"' in source, f"KPI 标签必须中文化:{label}"
    assert '"FINISHED": "已完成"' in source, "状态枚举必须有显示层中文映射"
    for header in (
        '"运行名"',
        '"开始时间"',
        '"轮数"',
        '"学习率"',
        '"训练损失"',
        '"验证损失"',
    ):
        assert header in source, f"表头必须中文化:{header}"
    assert 'st.subheader("对比运行")' in source
    assert 'st.subheader("运行详情")' in source
    # 旧英文串按调用形态退场(裸词会误伤列名/注释)
    assert 'st.metric("Total Runs"' not in source
    assert "All Runs (" not in source
    assert "Select 2–5 runs" not in source
    assert 'st.subheader("Compare Runs")' not in source
    assert 'st.subheader("Run Details")' not in source


def test_experiments_delete_requires_confirmation():
    """01 删除确认(00:596 popover 范式移植):mlflow.delete_run 是软删除
    (入 .trash,UI 无恢复入口),全库唯一「无确认+事实不可逆」误触面,
    且紧邻高频筛选控件——必须 popover 二次确认,caption 如实说明软删除
    性质(不夸大「永久删除」,也不隐瞒「界面不再显示」)。"""
    source = _source(PAGE_EXP)
    assert 'st.popover("🗑 删除此运行"' in source, "删除入口必须是 popover"
    popover_block = source.split('st.popover("🗑 删除此运行"', 1)[1]
    confirm_pos = popover_block.find('st.button("确认删除"')
    delete_pos = popover_block.find("mlflow.delete_run")
    assert confirm_pos != -1 and delete_pos != -1 and confirm_pos < delete_pos, (
        "确认按钮必须在 popover 内且先于 delete_run 执行"
    )
    assert ".trash" in popover_block, "caption 必须如实说明软删除去向(.trash)"
    # 防未来绕过:全文件仅此一处 delete_run 调用(块外裸删逃不过 popover 钉),
    # 且删除后必须清 30s TTL 缓存——否则 caption「将从界面移除」被陈旧缓存证伪
    # (审查 should-fix-1:fetch_runs 是 @st.cache_data(ttl=30),04 页 mutation 后
    # .clear() 是既有 idiom)。
    assert source.count("mlflow.delete_run") == 1, "delete_run 只能出现在确认流程内"
    assert "fetch_runs.clear()" in popover_block, "删除后必须清 fetch_runs 缓存"
    assert "🗑 Delete This Run" not in source, "裸删按钮必须退场"


def test_evaluation_import_entries_unified():
    """02 三入口(无域分支/无数据分支/页面底部)四种文案各统一为一种中文
    说法(R103 登记-①闭环):按钮/成功句/空态句/错误句各 ×3 计数闭合;
    spinner 标签逐字保持(R103 既有钉,冗余重锁防本轮误改)。"""
    source = _source(PAGE_EVAL)
    assert source.count('st.button("扫描并导入 MLflow"') == 3, "三入口按钮必须同文案"
    assert source.count("已导入 {imported} 个评测结果文件到 MLflow。") == 3
    assert source.count("未发现可导入的评测结果文件。") == 3
    assert source.count("导入 {json_file.name} 失败：{e}") == 3
    assert source.count('with st.spinner("正在扫描并导入历史评测结果到 MLflow…"):') == 3, (
        "spinner 标签逐字不动(R103 钉)"
    )
    assert "Scan & Import" not in source, "旧英文按钮文案必须退场"


def test_model_comparison_localized():
    """03 是决策者直达页(执行摘要/成本估算),必须整页中文——英文的决策
    页对非专家决策者是零信息。指标名中文化但 MRR 术语保留。"""
    source = _source(PAGE_CMP)
    assert 'st.subheader("执行摘要")' in source
    assert 'st.subheader("成本估算")' in source
    assert 'st.subheader("指标对比")' in source
    for text in ("总体准确率", "平均置信度", "平均延迟", "100 万样本耗时", "本地部署"):
        assert text in source, f"03 关键文案必须中文化:{text}"
    assert 'st.subheader("Executive Summary")' not in source
    assert 'st.subheader("Cost Estimation")' not in source
    assert '"MRR"' in source, "MRR 是领域术语,保留不译"


def test_registry_localized():
    """04 中文化但 MLflow 生命周期术语保留:champion/challenger 别名与
    stage 枚举(Staging/Production/Archived)是 API 契约值,动的只有
    控件文案(设置/移除/应用)。"""
    source = _source(PAGE_REG)
    assert 'st.subheader("别名（Aliases）")' in source
    assert 'st.button("🏷 设置"' in source
    assert 'st.button("🗑 移除"' in source
    assert 'st.button("应用"' in source
    assert "阶段迁移（旧版）" in source
    assert "champion" in source and "challenger" in source, "别名术语保留不译"
    assert "本页需要 mlflow" in source, "mlflow 缺失错误句必须中文化(审查 should-fix-2)"
    assert "Install mlflow to use this page" not in source
    assert 'st.button("🏷 Set")' not in source
    assert 'st.button("🗑 Remove")' not in source
    assert 'st.button("Apply"' not in source


def test_app_home_localized():
    """app 首页 hero 以下对齐 hero 中文(状态条产品名/设备枚举保留):
    快捷操作/统计卡/最近动态是首页最高频视线区。"""
    source = _source(PAGE_HOME)
    assert 'st.subheader("快捷操作")' in source
    assert 'st.subheader("最近动态")' in source
    for label in ("运行总数", "已完成", "运行中", "失败", "进行中的任务"):
        assert f'st.metric("{label}"' in source, f"首页统计卡必须中文化:{label}"
    for text in ("🏋️ 发起训练", "📊 查看实验", "🎯 评测结果", "⚖️ 对比模型"):
        assert text in source, f"快捷操作按钮必须中文化:{text}"
    assert "暂无训练记录" in source, "空态提示必须中文化"
    assert 'st.subheader("Quick Actions")' not in source
    assert 'st.subheader("Recent Activity")' not in source
    assert 'st.metric("Total Runs"' not in source


def test_localized_pages_boot_without_exception():
    """整页启动旅程钉:五文件文案改造后必须仍能真实渲染(AppTest 无异常)——
    源码扫描钉锁字符串,锁不住 NameError/缩进错/参数名错配这类编辑事故。"""
    from streamlit.testing.v1 import AppTest

    for page in (PAGE_EXP, PAGE_EVAL, PAGE_CMP, PAGE_REG, PAGE_HOME):
        app = AppTest.from_file(str(page), default_timeout=30)
        app.run()
        assert not app.exception, (
            f"{page.name} 本地化后必须正常启动:{[e.message for e in app.exception]}"
        )
