"""Data Wizard — 引导式数据准备：原始表格 → 质检过的训练集。

面向不懂 ML 的工程师：上传 CSV/Excel/JSONL → 确认列映射 → 一键生成 →
七项数据体检 → 预览/下载。所有逻辑复用 src/data/wizard（CLI 同源），
本页只做展示与交互。等价 CLI: python scripts/data_wizard.py --input ...
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Data Wizard", page_icon="🧙", layout="wide")
st.title("🧙 数据向导 · Data Wizard")
st.caption("原始表格 → 训练集：字段映射 · 候选构造 · 数据体检 · 防泄漏切分，全流程引导。")
# 双数据门路牌（R133）：有表直转是本页定位；从业务目标出发的引导式全流程在数据
# 入口页——00/05 此前对它零提及（审计证实），两扇数据门互不指路。
st.caption(
    "有原始表格、想直接转成训练集——你找对了。想从业务目标出发、由产品引导完成"
    "分析→核对→训练→评测？去数据入口。"
)
if st.button("🚪 去数据入口"):
    st.switch_page("pages/07_Data_Intake.py")

try:
    import pandas as pd

    from src.data.wizard.importers import RawTable, import_table, suggest_mapping
    from src.data.wizard.pipeline import WizardPipeline
    from src.data.wizard.spec import FieldMapping, WizardError, WizardSpec
    from src.data.wizard.templates import available_templates, get_template
    from ui.config import PROJECT_ROOT
except ImportError as exc:
    st.error(f"缺少依赖，无法使用本页: {exc}。可运行 `pip install -e .[ui]` 安装。")
    st.stop()

# ── 演示数据：让用户不用准备文件也能走完全流程 ──────────────────────

DEMO_CSV = """标准名,别名,编码
阿莫西林胶囊,阿莫西林,Z1
阿莫西林颗粒,阿莫西林干混悬,Z2
布洛芬片,芬必得,Z3
布洛芬缓释胶囊,芬必得缓释,Z4
对乙酰氨基酚片,泰诺林,Z5
氨咖黄敏胶囊,感冒灵,Z6
罗红霉素片,严迪,Z7
头孢克肟分散片,世福素,Z8
头孢克洛缓释片,希刻劳,Z9
左氧氟沙星片,可乐必妥,Z10
"""

# 通用演示：供应商名归一化（虚构数据）——generic-first 的默认门面
DEMO_CSV_GENERIC = """标准名,别名,编码
蓝星智能科技（深圳）有限公司,蓝星科技,SUP-0001
蓝星智能科技（深圳）有限公司,蓝星智能,SUP-0001
蓝星精密制造（苏州）有限公司,蓝星精密,SUP-0002
蓝星精密制造（苏州）有限公司,蓝星苏州精密,SUP-0002
云岭数据技术（杭州）有限公司,云岭 DT,SUP-0003
云岭数据技术（杭州）有限公司,云岭数据,SUP-0003
长风物流（上海）有限公司,长风货运,SUP-0004
长风物流（上海）有限公司,长风物流上海,SUP-0004
华信电气（北京）有限公司,华信电气,SUP-0005
华信电气（北京）有限公司,华信北京,SUP-0005
"""

DEMO_CSV_MD = """标准名,别名,编码,类型,规格
保和堂(昌平区光明路店),保和堂大药房,P000001,机构,
益民堂(海淀区中关村店),益民堂药房,P000002,机构,
仁和药房(朝阳区望京店),仁和,P000003,机构,
同济堂(浦东新区张江店),同济堂药房,P000004,机构,
阿莫西林胶囊,阿莫仙,Z15020414,产品,0.25g*24片/盒
阿莫西林颗粒,阿莫西林干混悬,Z15020415,产品,0.125g*12袋/盒
布洛芬缓释胶囊,芬必得缓释,Z15020416,产品,0.3g*20粒/盒
草果四味汤散,草果四味,Z15020417,产品,10g/袋
"""


def _load_bytes(name: str, data: bytes, template_hint: str) -> None:
    """上传/演示字节流 → 落临时文件 → 走真实 import_table 代码路径。"""
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="tunesmith_wizard_"))
        path = tmp_dir / name
        path.write_bytes(data)
        with st.spinner("正在读取表格…"):
            table = import_table(path)
    except WizardError as exc:
        st.error(str(exc))
        return
    st.session_state["wizard_table"] = table
    st.session_state["wizard_source"] = name
    st.session_state["wizard_template"] = template_hint  # 演示数据自动切换对应模板
    st.session_state.pop("wizard_report", None)  # 新数据让旧报告失效
    st.session_state.pop("wizard_preview", None)  # 新数据让旧样本预览失效（R126）
    st.rerun()


# ═══ 第 1 步 · 导入数据 ═══════════════════════════════════════════

st.subheader("① 导入数据")

up_col, path_col, demo_col = st.columns([3, 2, 1])
with up_col:
    upload = st.file_uploader(
        "上传表格文件",
        type=["csv", "xlsx", "jsonl"],
        help="CSV / Excel / JSONL，需要一列「标准名」和一列「查询/别名」。Excel 需要 openpyxl。",
    )
    if upload is not None:
        _load_bytes(
            upload.name,
            upload.getvalue(),
            template_hint=st.session_state.get("wizard_template", "entity_matching"),
        )
with path_col:
    server_path = st.text_input(
        "或输入服务器上的文件路径",
        placeholder="/path/to/drugs.csv",
        help="大文件建议放服务器后填路径，不走浏览器上传。",
    )
    if st.button("读取该路径", disabled=not server_path.strip()):
        try:
            with st.spinner("正在读取表格…"):
                table = import_table(server_path.strip())
        except WizardError as exc:
            st.error(str(exc))
        else:
            st.session_state["wizard_table"] = table
            st.session_state["wizard_source"] = Path(server_path.strip()).name
            st.session_state.pop("wizard_report", None)
            st.session_state.pop("wizard_preview", None)  # 同上演示/上传路径（R126）
            st.rerun()
with demo_col:
    if st.button("🗂️ 通用演示（供应商名）"):
        _load_bytes(
            "demo_suppliers.csv", DEMO_CSV_GENERIC.encode("utf-8"), template_hint="entity_matching"
        )
    if st.button("💊 案例：药品名（医疗）"):
        _load_bytes("demo_drugs.csv", DEMO_CSV.encode("utf-8"), template_hint="medical_entity")
    if st.button("🏭 案例：主数据"):
        _load_bytes(
            "demo_master_data.csv", DEMO_CSV_MD.encode("utf-8"), template_hint="master_data"
        )

table: RawTable | None = st.session_state.get("wizard_table")
if table is None:
    st.info(
        "👆 上传文件、填路径、或点演示数据按钮（🗂️ 通用演示为默认旅程；💊/🏭 为已验证案例）"
        "开始。向导不会写任何文件，直到第④步点生成。"
    )
    st.stop()

st.success(
    f"已加载 **{st.session_state.get('wizard_source', '?')}** — {len(table.rows)} 行 × {len(table.columns)} 列"
)
with st.expander("预览前 20 行"):
    st.dataframe(pd.DataFrame(table.rows, columns=table.columns), use_container_width=True)

st.divider()

# ═══ 第 2 步 · 列映射 ═════════════════════════════════════════════

st.subheader("② 确认列映射")

try:
    suggested = suggest_mapping(table.columns)
except WizardError:
    suggested = FieldMapping(standard_name=table.columns[0] if table.columns else "")

template_names = available_templates()
default_template = "entity_matching" if "entity_matching" in template_names else template_names[0]
template_name = st.selectbox(
    "垂类模板",
    template_names,
    index=template_names.index(default_template),
    key="wizard_template",
    help="entity_matching=通用 Alpaca 选择题（默认）；医疗/主数据=已验证案例模板。",
)
template = get_template(template_name)
with st.expander("模板说明"):
    st.text(template.describe())

st.markdown("**列 → 角色**（已按列名自动识别，识别错了直接改下拉框）")
roles = [
    ("standard_name", "标准名（必填）", suggested.standard_name),
    ("query", "查询 / 别名", suggested.query),
    ("code", "标准编码", suggested.code),
    ("variants", "变体（一格多个，按 、;，| 切）", suggested.variants),
    ("entity_type", "实体类型", suggested.entity_type),
    ("spec", "规格（产品任务）", suggested.spec),
]
NONE_OPTION = "（不使用）"
selected: dict[str, str | None] = {}
cols = st.columns(len(roles))
for col, (role, label, default) in zip(cols, roles):
    options = [NONE_OPTION, *table.columns]
    default_idx = options.index(default) if default in options else 0
    picked = col.selectbox(label, options, index=default_idx, key=f"role_{role}")
    selected[role] = None if picked == NONE_OPTION else picked

current_mapping = FieldMapping(
    standard_name=selected["standard_name"] or "",  # type: ignore[arg-type]
    query=selected["query"],
    code=selected["code"],
    variants=selected["variants"],
    entity_type=selected["entity_type"],
    spec=selected["spec"],
)
mapping_errors = current_mapping.validate(table.columns)

st.divider()

# ═══ 第 3 步 · 生成参数 ═══════════════════════════════════════════

st.subheader("③ 生成参数")

p1, p2, p3, p4, p5 = st.columns(5)
n_candidates = p1.number_input(
    "每个样本的候选数", 2, 20, 8, help="太少学不到判别，太多浪费上下文。"
)
r_train = p2.number_input("train 比例", 0.05, 0.95, 0.8, 0.05)
r_val = p3.number_input("val 比例", 0.05, 0.95, 0.1, 0.05)
r_test = p4.number_input("test 比例", 0.05, 0.95, 0.1, 0.05)
seed = p5.number_input("随机种子", 0, 9999, 42, help="同种子 = 同产出，可复现。")
noise = st.checkbox(
    "噪音增强（错别字鲁棒性）",
    value=False,
    help="每个样本追加一条带错别字/漏字的查询副本：标准答案与候选不变，难度按扰动后重估。"
    "模拟真实输入噪声；val/test 也会各自带上扰动样本。",
)
dedup = st.checkbox("自动去重（相同查询+标准名只保留一条）", value=True)

ratios_sum = round(r_train + r_val + r_test, 2)
ratios_ok = abs(ratios_sum - 1.0) <= 0.01
if not ratios_ok:
    st.warning(f"三个比例之和应为 1.0，当前 {ratios_sum}。")
ready = not mapping_errors and ratios_ok
if mapping_errors:
    st.warning("；".join(mapping_errors))

# ── 样本预览（R126）：生成前用真实数据验证映射 ──────────────────────
# 映射错了（非专家最常见错误）旧世界要等第④步生成完才发现，甚至带着错
# 数据去训练。第③步末就地试生成：前 3 行真实数据 + 当前映射/参数 →
# 2 条样本长什么样。按需计算（按钮触发 + 指纹缓存），大表不在每次
# rerun 全量 build；指纹一变旧预览自然失效。
preview_fp = (
    table.source,
    template_name,
    selected["standard_name"],
    selected["query"],
    selected["code"],
    selected["variants"],
    selected["entity_type"],
    selected["spec"],
    int(n_candidates),
    int(seed),
)
if st.button(
    "👀 试生成前 2 条样本",
    disabled=not ready,
    help="用前 3 行真实数据 + 当前映射/参数试跑模板——生成前确认样本长什么样、映射没接错列。",
):
    try:
        _slice = RawTable(source=table.source, columns=table.columns, rows=table.rows[:3])
        _spec = WizardSpec(
            mapping=current_mapping,
            template=template_name,
            split_ratios=(0.8, 0.1, 0.1),
            n_candidates=int(n_candidates),
            dedup=True,
            seed=int(seed),
        )
        with st.spinner("正在试生成…"):
            _built = template.build_samples(_slice, current_mapping, _spec)
        st.session_state["wizard_preview"] = {
            "key": preview_fp,
            "records": [template.format_record(s) for s in _built.samples[:2]],
            "dropped": _built.dropped,
        }
    except Exception as exc:  # 预览失败不阻断主流程，如实报错即可
        st.error(f"试生成失败：{exc}")
        st.session_state.pop("wizard_preview", None)

_pv = st.session_state.get("wizard_preview")
if _pv is not None and _pv["key"] != preview_fp:
    _pv = None  # 映射/参数已变，旧预览不再代表当前配置
if _pv is not None:
    st.caption("预览基于前 3 行真实数据（负例池小，候选数可能少于设置值）——完整生成以第④步为准。")
    for _i, _rec in enumerate(_pv["records"], 1):
        with st.expander(f"样本 {_i}（试生成）"):
            if "messages" in _rec:
                for msg in _rec["messages"]:
                    label = {"system": "🧭 system", "user": "👤 user"}.get(
                        msg["role"], "🤖 assistant"
                    )
                    st.markdown(f"**{label}**")
                    st.code(str(msg["content"]), language=None)
            else:
                st.markdown(f"**instruction**\n\n{_rec['instruction']}")
                st.markdown(f"**input**\n\n```\n{_rec['input']}\n```")
                st.markdown(f"**output**\n\n```\n{_rec['output']}\n```")
    if not _pv["records"]:
        if _pv["dropped"]:
            st.warning(
                "前 3 行全部被跳过："
                + "；".join(f"第 {d.row} 行 {d.reason}" for d in _pv["dropped"])
                + "——检查列映射后重试。"
            )
        else:
            st.info("前 3 行没有生成任何样本——检查列映射后重试。")

st.divider()

# ═══ 第 4 步 · 生成 + 数据体检 ════════════════════════════════════

st.subheader("④ 生成与数据体检")
gen_col, out_col = st.columns([1, 2])
out_dir_text = out_col.text_input(
    "导出目录（相对项目根）",
    value=f"outputs/wizard/{Path(st.session_state.get('wizard_source', 'data')).stem}",
    help="生成 train/val/test.json + wizard_report.json，供后续训练直接使用。",
)
if gen_col.button("🚀 生成训练集", type="primary", disabled=not ready):
    try:
        spec = WizardSpec(
            mapping=current_mapping,
            template=template_name,
            split_ratios=(float(r_train), float(r_val), float(r_test)),
            n_candidates=int(n_candidates),
            noise_augment=bool(noise),
            dedup=bool(dedup),
            seed=int(seed),
        )
        out_dir = (PROJECT_ROOT / out_dir_text).resolve()
        # 标签如实点名流水线的真实阶段（WizardPipeline 无阶段回调，不虚构分段进度）
        with st.spinner("正在生成训练集：候选构造 → 数据体检 → 防泄漏切分 → 导出…"):
            report = WizardPipeline(spec).run(table, out_dir)
    except WizardError as exc:
        st.error(f"生成失败：{exc}")
    else:
        st.session_state["wizard_report"] = report
        st.session_state["wizard_out_dir"] = str(out_dir)

report = st.session_state.get("wizard_report")
if report is None:
    st.stop()

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("原始行", report.total_rows)
m2.metric("生成样本", report.built_samples)
m3.metric("去重移除", report.dedup_removed)
m4.metric("跳过行", len(report.dropped_rows))
m5.metric("train", report.split_counts.get("train", 0))
m6.metric(
    "val / test", f"{report.split_counts.get('val', 0)} / {report.split_counts.get('test', 0)}"
)
if report.augmented:
    st.caption(f"🧬 噪音增强已追加 {report.augmented} 条带错别字的查询副本（标签不变，难度重估）。")

st.markdown("**数据体检**（✗ = 阻断性问题，必须修复才能用于训练；⚠ = 有风险，建议关注）")
_COLOR = {"error": "#F87171", "warning": "#FBBF24", "info": "#94A3B8"}
for c in report.checks:
    icon = "✓" if c.passed else ("✗" if c.severity == "error" else "⚠")
    color = "#4ADE80" if c.passed else _COLOR.get(c.severity, "#94A3B8")
    st.markdown(
        f"<div style='padding:0.35rem 0;border-bottom:1px solid #334155;'>"
        f"<span style='color:{color};font-weight:700'>{icon}</span> "
        f"<span style='color:#CBD5E1'>{c.message}</span></div>",
        unsafe_allow_html=True,
    )

# 难度分布（info 级，帮助理解评测覆盖面）；difficulty 是 split→level→count，先聚合
diff_totals: dict[str, int] = {}
for per_split in (report.export.difficulty if report.export else {}).values():
    for level, n in per_split.items():
        diff_totals[level] = diff_totals.get(level, 0) + n
if diff_totals:
    total_diff = sum(diff_totals.values()) or 1
    st.markdown("**难度分布**")
    for level, label in (("easy", "easy 简单"), ("medium", "medium 中等"), ("hard", "hard 困难")):
        n = diff_totals.get(level, 0)
        st.progress(n / total_diff, text=f"{label}: {n}（{n / total_diff:.0%}）")

st.divider()

# ═══ 第 5 步 · 预览与下载 ════════════════════════════════════════

st.subheader("⑤ 预览与下载")
if report.export is None:
    st.error("存在阻断性错误，未导出任何文件。请按上方 ✗ 项修复后重新生成。")
    st.stop()

out_dir = Path(st.session_state.get("wizard_out_dir", ""))
st.success(f"已导出到 `{out_dir}`")
split_choice = st.radio("选择 split 预览", ["train", "val", "test"], horizontal=True)
try:
    records = json.loads((out_dir / f"{split_choice}.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    st.error(f"读取导出文件失败: {exc}")
    st.stop()
for i, rec in enumerate(records[:3]):
    with st.expander(f"样本 {i + 1}（{split_choice}）"):
        if "messages" in rec:
            # messages chat 格式（主数据模板）：逐角色展示
            _ROLE_ICON = {"system": "🧭 system", "user": "👤 user", "assistant": "🤖 assistant"}
            for msg in rec["messages"]:
                label = _ROLE_ICON.get(msg["role"], msg["role"])
                st.markdown(f"**{label}**")
                st.code(msg["content"], language=None)
        else:
            st.markdown(f"**instruction**\n\n{rec['instruction']}")
            st.markdown(f"**input**\n\n```\n{rec['input']}\n```")
            st.markdown(f"**output**\n\n```\n{rec['output']}\n```")
            st.caption(f"metadata: {rec['metadata']}")
if len(records) > 3:
    st.caption(f"…共 {len(records)} 条")

dl1, dl2, dl3, dl4 = st.columns(4)
for col, name in zip((dl1, dl2, dl3, dl4), ("train", "val", "test", "wizard_report")):
    f = out_dir / f"{name}.json"
    if f.exists():
        col.download_button(f"⬇️ {name}.json", f.read_bytes(), file_name=f"{name}.json")

if template_name == "master_data":
    st.info(
        f"训练集就绪（messages 双任务格式，主数据训练脚本兼容）。下一步在终端运行：\n\n"
        f"`python domains/master_data/scripts/train.py --train-file {out_dir / 'train.json'} --epochs 1`\n\n"
        f"机构与产品样本已混排在同一训练集，system prompt 会告诉模型当前是哪个任务。"
    )
else:
    train_path = out_dir / "train.json"
    if st.button("🏋️ 送去 Training Lab 训练", type="primary", use_container_width=True):
        # 页内交接：预填 Training Lab 的数据集字段后跳转，全程不落终端
        st.session_state["dataset_input"] = str(train_path)
        st.session_state["wizard_handoff"] = {
            "path": str(train_path),
            "samples": report.split_counts.get("train", 0),
        }
        st.switch_page("pages/00_Training_Lab.py")
    st.caption(
        f"或走 CLI：`python scripts/train_sft.py -d {train_path}`"
        f"（训练时 Validation Split 会从 train.json 内部再切验证集，向导导出的 val/test 供评测用）。"
    )
