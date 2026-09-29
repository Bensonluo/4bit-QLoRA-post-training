"""语域收尾(R106):00 页英文骨架/Stop 确认族/domain_adapters 文案+难度枚举统一。

选点依据(R105 轮报登记+主会话等待窗口一手侦察):00 页是中英混排最重页
(79/453 已中文,而预设按钮三件套/配置表单六 section 头/三滑杆/Activity 骨架
全英文),01:39 已用「训练实验室(Training Lab)」括注指路、00:189 中文句指名
英文按钮「Start Training」——指路牌与目的地语言不一致。00:593 ⏹ Stop 与
07:1526「停止本轮后台执行」均为裸按钮直发,中断在途训练/agent 执行无确认。
domain_adapters.py 组件文案约 20 串英文在 02/03 页内渲染,难度/实体类型枚举
跨三页(02 adapter/03/数据值)显示层不统一,R105 有意保留登记本轮统一。

术语边界(承接 R105):LoRA/Dropout/PID/SFT/DPO/GRPO/MLflow/MRR/HF 等术语与
硬件名(Apple Silicon (MPS)/NVIDIA (CUDA)/CPU)与数据枚举值(easy/medium/
hard、drug/hospital——只动显示层 format_func/标签映射,不动数据值与筛选
逻辑)不译。00 页 Status 枚举(running/finished/failed)显示映射与 01 页
_STATUS_ZH 对齐。「训练实验室(Training Lab)」括注范式:首次出现给中文名,
英文括注可留作对齐文档术语。

钉型(R97-R105 先例):逐字锁关键句+retirement 按 UI 调用形态锁+Stop 控制流
钉(确认按钮先于 stop 调用,防 popover 外裸发)+计数闭合(stop_training/
execution_service.stop 全文件唯一)。R104 空态钉「No training runs yet」
(test_activity_autorefresh_ui.py:152)随文案同步维护——钉随文案走的合法
维护,轮报披露。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

UI = Path(__file__).resolve().parents[2] / "ui"
PAGE_LAB = UI / "pages/00_Training_Lab.py"
PAGE_INTAKE = UI / "pages/07_Data_Intake.py"
PAGE_CHAT = UI / "pages/06_Chat.py"
PAGE_CMP = UI / "pages/03_Model_Comparison.py"
ADAPTERS = UI / "components/domain_adapters.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_lab_header_and_presets_localized():
    """00 页标题/browser 标签/预设按钮三件套中文化:预设按钮是 00:189 中文
    指路句点名的操作对象,英文按钮让指路失效。"""
    source = _source(PAGE_LAB)
    assert 'page_title="训练实验室"' in source
    assert 'st.title("🏋️ 训练实验室")' in source
    for label in ("⚡ 快速测试", "🔥 标准", "🚀 完整运行"):
        assert f'st.button("{label}"' in source, f"预设按钮必须中文化:{label}"
    assert "小模型" in source and "完整模型" in source, "预设按钮 help 必须中文化"
    # 旧英文按调用形态退场
    assert 'st.title("🏋️ Training Lab"' not in source
    assert 'st.button("⚡ Quick Test"' not in source
    assert 'st.button("🔥 Standard"' not in source
    assert 'st.button("🚀 Full Run"' not in source
    assert "Start Training" not in source, "表单提交按钮与指路句必须统一中文"


def test_lab_config_sections_localized():
    """配置表单六 section 头/控件标签/滑杆中文化;LoRA/Dropout 等术语按括注
    范式保留(「LoRA 秩(r)」「Dropout 比例」);底座模型沿用 CLAUDE.md 底座
    词汇;全精度提示句如实(4-bit 仅 CUDA)。"""
    source = _source(PAGE_LAB)
    for sub in ("技术", "模型与数据", "训练参数", "模型注册表", "量化", "配置预览"):
        assert f'st.subheader("{sub}")' in source, f"section 头必须中文化:{sub}"
    for label in ("运行平台", "底座模型", "验证集比例", "LoRA 秩（r）", "Dropout 比例"):
        assert f'"{label}"' in source, f"控件标签必须中文化:{label}"
    assert "全精度 LoRA" in source and "仅在 NVIDIA CUDA" in source, "全精度提示必须中文化"
    # 旧英文退场
    for retired in (
        'st.subheader("Technique")',
        'st.subheader("Model & Data")',
        'st.subheader("Training")',
        'st.subheader("Registry")',
        'st.subheader("Quantization")',
        'st.subheader("Config Preview")',
    ):
        assert retired not in source
    assert "Full Precision LoRA — 4-bit requires NVIDIA CUDA" not in source


def test_lab_activity_localized():
    """Activity 骨架(训练动态/刷新/状态/空态/删除 popover)中文化;Status
    枚举显示映射与 01 页 _STATUS_ZH 对齐(running→运行中)。"""
    source = _source(PAGE_LAB)
    assert 'st.subheader("训练动态")' in source
    assert 'st.button("🔄 刷新"' in source
    assert 'st.metric("状态"' in source
    assert '"running": "运行中"' in source, "Status 枚举必须有中文显示映射"
    assert "暂无训练运行记录" in source, "空态必须中文化"
    assert "最近日志" in source, "日志 expander 必须中文化"
    assert "Technique: {info.get(" not in source, "运行 caption 必须中文化"
    # 旧英文退场
    for retired in (
        'st.subheader("Training Activity")',
        'st.button("🔄 Refresh"',
        'st.metric("Status"',
        "No training runs yet",
        'st.popover("🗑 Delete"',
        "Confirm delete",
    ):
        assert retired not in source


def test_lab_stop_requires_confirmation():
    """00:593 ⏹ Stop 从裸按钮改 popover 确认:中断在途训练(杀进程)是高代价
    误触面。控制流钉:确认按钮在 popover 内且先于 stop_training;全文件唯一
    stop_training 调用(防 popover 外裸发)。"""
    source = _source(PAGE_LAB)
    assert 'st.popover("⏹ 停止"' in source, "停止入口必须是 popover"
    block = source.split('st.popover("⏹ 停止"', 1)[1]
    confirm_pos = block.find('st.button("确认停止"')
    stop_pos = block.find("runner.stop_training")
    assert confirm_pos != -1 and stop_pos != -1 and confirm_pos < stop_pos, (
        "确认按钮必须在 popover 内且先于 stop_training"
    )
    assert source.count("runner.stop_training") == 1, "stop_training 只能出现在确认流程内"
    # 00 页删除 popover 同轮中文化(caption 如实:本地运行记录,产物保留)
    del_block = source.split('st.popover("🗑 删除"', 1)[1]
    assert 'st.button("确认删除"' in del_block, "删除确认按钮必须中文化"
    assert "config" in del_block.lower() or "配置" in del_block, "删除 caption 必须说明保留范围"


def test_lab_stop_gated_to_running_status():
    """停止面板只对运行中的 run 渲染(r106-reviewer should-fix-1 采纳):
    HEAD 原为 `if status == "running" and st.button("⏹ Stop", ...)`,
    popover 化时门被丢弃——stop_training 对已结束进程是静默 no-op,无条件
    渲染的 popover 会让 caption「将终止该运行的训练进程」对已完成/失败的
    run 说谎。门必须紧贴 popover(≤4 行),防未来再动 popover 时无声丢门。"""
    lines = _source(PAGE_LAB).splitlines()
    stop_idx = next(i for i, line in enumerate(lines) if 'st.popover("⏹ 停止"' in line)
    gates = [
        i for i, line in enumerate(lines[:stop_idx]) if line.strip() == 'if status == "running":'
    ]
    assert gates, '停止 popover 必须由 status == "running" 门控(非运行 run 不渲染)'
    assert stop_idx - max(gates) <= 4, "running 门必须紧贴停止 popover"


def test_intake_stop_requires_confirmation():
    """07:1526「停止本轮后台执行」同族处理:中断后台 agent 执行丢弃在途
    工作。确认按钮先于 execution_service.stop,全文件唯一。"""
    source = _source(PAGE_INTAKE)
    assert 'st.popover("停止本轮后台执行"' in source
    block = source.split('st.popover("停止本轮后台执行"', 1)[1]
    confirm_pos = block.find('st.button("确认停止"')
    stop_pos = block.find("execution_service.stop")
    assert confirm_pos != -1 and stop_pos != -1 and confirm_pos < stop_pos
    assert source.count("execution_service.stop(") == 1


def test_domain_adapter_copy_localized():
    """domain_adapters.py 组件文案中文化(02 页内渲染):display_name 影响
    02/03 域选择器;难度/实体类型枚举显示层统一映射(数据值不动——筛选
    options 仍是英文枚举,format_func 只动显示)。render_error_analysis 的
    内层 Error Analysis subheader 与 02 页 expander 标题重复,移除(声明性
    结构简化:唯一消费方是 02 的「🔍 错误分析」expander)。"""
    source = _source(ADAPTERS)
    assert 'display_name = "医疗实体匹配"' in source
    # 难度/实体类型显示映射 + 公开 helper(03 页共用)
    assert '"easy": "简单"' in source and '"hard": "困难"' in source
    assert "def difficulty_label(" in source
    assert "def entity_type_label(" in source
    assert '"drug": "药品"' in source and '"hospital": "医院"' in source
    for sub in ("按难度分组的准确率", "按实体类型分组的准确率", "延迟 vs 准确率"):
        assert f'st.subheader("{sub}")' in source
    assert "暂无评测数据。" in source
    assert "个错误样本" in source
    for text in ("按难度筛选", "按实体类型筛选", "查询", "正确答案", "置信度"):
        assert text in source, f"错误分析文案必须中文化:{text}"
    # 筛选值保持英文枚举 + format_func 显示层
    assert '"All"] + ["easy", "medium", "hard"]' in source, "筛选选项值必须是原始枚举"
    assert "format_func=" in source, "筛选显示必须走 format_func 映射"
    # 旧英文退场(含重复 subheader 移除)
    for retired in (
        'display_name = "Medical Entity Matching"',
        'st.subheader("Accuracy by Difficulty")',
        'st.subheader("Accuracy by Entity Type")',
        'st.subheader("Latency vs Accuracy")',
        "No evaluation data available.",
        "Filter by difficulty",
        "Filter by entity type",
        "Ground truth",
        "Difficulty: {err.get",
        'st.subheader("Error Analysis")',
    ):
        assert retired not in source, f"旧英文必须退场:{retired}"


def test_comparison_difficulty_labels_unified():
    """03 页难度/实体类型标签与 adapter 同源(共用 helper),diff.title()/
    etype.title() 显示退场——跨页显示统一是 R105 登记的本轮目标。"""
    source = _source(PAGE_CMP)
    assert "difficulty_label" in source, "03 必须复用 adapter 的显示映射"
    assert "entity_type_label" in source
    assert "diff.title()" not in source
    assert "etype.title()" not in source


def test_chat_sliders_localized():
    """06 页两滑杆中文化(低收益随轮):采样参数对非专家用标准中文术语。"""
    source = _source(PAGE_CHAT)
    assert "最大生成 token 数" in source
    assert "采样温度" in source
    assert 'st.slider("Max new tokens"' not in source
    assert 'st.slider("Temperature"' not in source
