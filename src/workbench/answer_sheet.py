"""待补答案清单:把「由了解业务的人填写」变成可交接的文件(介入点 10 编码)。

缺答案行此前只是页面上的一段提示——填写人拿不到只含待补行的清单,
预览编号也无法直接定位原文件。这里把行筛选、填写表与交接规则收敛为
单一来源:页面下载按钮与 CLI answer-sheet 命令渲染同一份行与规则,
needs_labels 状态判定也复用同一个行筛选。
"""

from __future__ import annotations

import csv
import io

from src.workbench.intake_models import IntakeSession, PreviewRow, SampleSource
from src.workbench.temporal_split import TemporalSplitPolicy


def missing_answer_rows(
    rows: list[PreviewRow], policy: TemporalSplitPolicy | None = None
) -> list[PreviewRow]:
    """筛选当前需要人工补答案的行(单一来源)。

    needs_label 且不是时间方案的未成熟标签——未成熟行是「标签窗口未结束」
    的既定保留,不是漏填,不应混进交给填写人的清单。next_action 的
    needs_labels 判定、页面提示与 CLI 清单共用这一个筛选。
    """
    # 调用时再导入:测试按 temporal_split 模块属性替换 is_pending_label 时,
    # 这里的引用随调用解析(顶层 from-import 会把旧引用绑进本模块)。
    from src.workbench.temporal_split import is_pending_label

    return [
        row for row in rows if row.status == "needs_label" and not is_pending_label(row, policy)
    ]


def answer_sheet_source(session: IntakeSession, *, full: bool = False) -> SampleSource | None:
    """Return a direct source only when preview IDs still identify its rows."""
    view = session.full_data if full else session
    if view is None or view.composition_report is not None:
        return None
    if view.sources:
        return view.sources.get("main")
    # Legacy adapted snapshots may not retain their original source metadata.
    return None if view.adapter_report is not None else view.source


def answer_sheet_csv(
    rows: list[PreviewRow], field_names: list[str], *, source: SampleSource | None = None
) -> bytes:
    """填写表导出为 CSV(带 BOM,Excel 直开);交给了解业务的人线下填写。

    行ID是预览编号而非原文件编号。直接来源提供记录起始行及工作表，
    组合来源或缺失来源不猜测位置。表内不包含已有答案。
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "行ID",
            "题目输入（模型将看到的内容）",
            *[f"待填答案（{name}）" for name in field_names],
            "原始记录位置",
        ]
    )
    source_rows = {row.row_id: row for row in source.rows} if source else {}
    if source and len(source_rows) != len(source.rows):
        source_rows = {}
    for row in rows:
        original = source_rows.get(row.row_id)
        location = "未记录或不可核验，请查看来源血缘"
        if (
            source is not None
            and original is not None
            and all(
                key in row.original and row.original[key] == value
                for key, value in original.values.items()
            )
        ):
            sheet = f" · 工作表 {source.sheet}" if source.sheet else ""
            location = f"文件 {source.name}{sheet} · 第 {original.line} 行（记录起始行）"
        writer.writerow([row.row_id, row.input, *[""] * len(field_names), location])
    return buffer.getvalue().encode("utf-8-sig")


def answer_sheet_lines(count: int) -> list[str]:
    """交接与填写规则(单一来源):页面提示与 CLI stderr 输出同一份行。

    这三行是介入点 10 的判断编码:怎么交接(只填答案列、手动回填原文件)、
    填写依据(业务事实,不凭猜测——答案直接成为监督信号)、输入不足时
    怎么办(留空并说明,是任务定义问题,不是编答案的理由)。
    """
    return [
        f"共 {count} 条待补：交给填写人后只填「待填答案」列，行ID 与题目输入列保持原样。行ID 是当前预览编号；依据原始记录位置手动回填预览时的原文件，组合来源先核对来源血缘。文件已修改或重排时请重新预览；清单不能直接上传回填。",
        "填写依据是业务事实（查业务系统或档案都可以）；不要凭猜测填——答案会直接成为模型的监督信号，错一条教错一条。",
        "填写人认为题目输入不足以判断答案时，把该行留空并在回传时说明——输入信息不足是任务定义问题（与盲标核验、可学性探针的同类提示同方向），比编一个答案更有价值。",
    ]
