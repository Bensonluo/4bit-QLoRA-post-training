"""待补答案清单:把「由了解业务的人填写」变成可交接的文件(介入点 10 编码)。

缺答案行此前只是页面上的一段提示——填写人拿不到只含待补行的清单,
「保留原编号」也只是口头要求。这里把行筛选、填写表与交接规则收敛为
单一来源:页面下载按钮与 CLI answer-sheet 命令渲染同一份行与规则,
needs_labels 状态判定也复用同一个行筛选。
"""

from __future__ import annotations

import csv
import io

from src.workbench.intake_models import PreviewRow
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


def answer_sheet_csv(rows: list[PreviewRow], field_names: list[str]) -> bytes:
    """填写表导出为 CSV(带 BOM,Excel 直开);交给了解业务的人线下填写。

    列固定为 行ID、题目输入、每字段一个待填列:填写人只补答案列,行ID 与
    题目输入列原样保留,回传后按行ID 对号回填。表内不包含任何数据答案——
    缺答案行本就没有答案可填,清单结构上不可能泄露监督标签。
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["行ID", "题目输入（模型将看到的内容）", *[f"待填答案（{name}）" for name in field_names]]
    )
    for row in rows:
        writer.writerow([row.row_id, row.input, *[""] * len(field_names)])
    return buffer.getvalue().encode("utf-8-sig")


def answer_sheet_lines(count: int) -> list[str]:
    """交接与填写规则(单一来源):页面提示与 CLI stderr 输出同一份行。

    这三行是介入点 10 的判断编码:怎么交接(只填答案列、行ID 对号回填)、
    填写依据(业务事实,不凭猜测——答案直接成为监督信号)、输入不足时
    怎么办(留空并说明,是任务定义问题,不是编答案的理由)。
    """
    return [
        f"共 {count} 条待补：交给填写人后只填「待填答案」列，行ID 与题目输入列保持原样——回传后按行ID 对号回填。",
        "填写依据是业务事实（查业务系统或档案都可以）；不要凭猜测填——答案会直接成为模型的监督信号，错一条教错一条。",
        "填写人认为题目输入不足以判断答案时，把该行留空并在回传时说明——输入信息不足是任务定义问题（与盲标核验、可学性探针的同类提示同方向），比编一个答案更有价值。",
    ]
