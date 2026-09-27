"""Built-in scenario matrix specs: the seed distribution for coverage measurement.

期望结局记录当前真相:unexpected_* 是产品缺陷;「通过但带已知局限」的场景
用 expect_note 记录待办(泄漏预警已上线;高基数标签拦截是已知待办)。
"""

from src.workbench.scenario_matrix import ScenarioSpec

_CLEAN_SAMPLE = ("编号,客户描述,类别\n001,杯子破损,质量\n002,物流未更新,物流\n").encode()
_CLEAN_FULL = (
    "编号,客户描述,类别\n"
    "001,杯子破损,质量\n002,物流未更新,物流\n003,屏幕碎裂,质量\n004,快递丢失,物流\n"
    "005,开不了机,质量\n006,地址填错,物流\n007,异味,质量\n008,延迟送达,物流\n"
    "009,无法充电,质量\n010,包装破损,物流\n"
).encode()

# GBK 编码样例:utf-8-sig 解码失败,入口自动回退 gb18030(GBK 超集)可正确解码。
_GBK_SAMPLE = "编号,描述,类别\n001,杯子破损,质量\n002,物流未更新,物流\n".encode("gbk")
_GBK_FULL = (
    "编号,描述,类别\n"
    "001,杯子破损,质量\n002,物流未更新,物流\n003,屏幕碎裂,质量\n004,快递丢失,物流\n"
    "005,开不了机,质量\n006,地址填错,物流\n007,异味,质量\n008,延迟送达,物流\n"
    "009,无法充电,质量\n010,包装破损,物流\n"
).encode("gbk")

# 宽表:28 列业务噪声全被用户排除,只留编号(分组)+客户描述(输入)+类别(答案)。
_WIDE_JUNK_COLUMNS = tuple(f"附加信息{i:02d}" for i in range(1, 29))
_WIDE_HEADER = ",".join(("编号", "客户描述", "类别", *_WIDE_JUNK_COLUMNS))
_WIDE_SAMPLE = (
    _WIDE_HEADER + "\n"
    "001,杯子破损,质量," + ",".join(["噪声01"] * 28) + "\n"
    "002,物流未更新,物流," + ",".join(["噪声02"] * 28) + "\n"
).encode()
_WIDE_FULL = (
    _WIDE_HEADER + "\n"
    "001,杯子破损,质量," + ",".join(["备注甲"] * 28) + "\n"
    "002,物流未更新,物流,"
    + ",".join(["备注乙"] * 28)
    + "\n"
    + "".join(
        f"{i:03d},问题{i},{'质量' if i % 2 else '物流'}," + ",".join([f"字段值{i % 7}"] * 28) + "\n"
        for i in range(3, 11)
    )
).encode()

# 混合类型列:工单号同时出现纯数字样与文本样编号,当前路径不做类型区分。
_MIXED_IDS = (
    "10023",
    "AB-1002",
    "10024",
    "AB-1003",
    "10025",
    "10026",
    "AB-1004",
    "10027",
    "10028",
    "AB-1005",
)
_MIXED_SAMPLE = (
    "编号,工单号,客户描述,类别\n001,10023,杯子破损,质量\n002,AB-1002,物流未更新,物流\n".encode()
)
_MIXED_FULL = (
    "编号,工单号,客户描述,类别\n"
    + "".join(
        f"{i:03d},{_MIXED_IDS[i - 1]},问题{i},{'质量' if i % 2 else '物流'}\n" for i in range(1, 11)
    )
).encode()

# Excel「Unicode 文本」导出:UTF-16(带 BOM)+ Tab 分隔,中文 Windows Excel 用户的常见产物。
_UTF16_ROWS = "编号\t客户描述\t类别\n001\t杯子破损\t质量\n002\t物流未更新\t物流\n"
_UTF16_SAMPLE = _UTF16_ROWS.encode("utf-16")
_UTF16_FULL = (
    "编号\t客户描述\t类别\n"
    + "".join(f"{i:03d}\t问题{i}\t{'质量' if i % 2 else '物流'}\n" for i in range(1, 11))
).encode("utf-16")

# 空答案行混入:样例两条都有标签,全量第 005/008 行答案为空(导出缺字段/漏标注)。
_EMPTY_LABEL_FULL = (
    "编号,客户描述,类别\n"
    "001,杯子破损,质量\n002,物流未更新,物流\n003,屏幕碎裂,质量\n004,快递丢失,物流\n"
    "005,开不了机,\n006,地址填错,物流\n007,异味,质量\n008,延迟送达,\n"
    "009,无法充电,质量\n010,包装破损,物流\n"
).encode()

# 列名前后空格:外部系统导出的表头带不可见的前后空格,用户按业务口径写「编号/类别」。
_SPACED_HEADER = " 编号 ,客户描述, 类别"
_SPACED_SAMPLE = (_SPACED_HEADER + "\n001,杯子破损,质量\n002,物流未更新,物流\n").encode()
_SPACED_FULL = (
    _SPACED_HEADER + "\n"
    "001,杯子破损,质量\n002,物流未更新,物流\n003,屏幕碎裂,质量\n004,快递丢失,物流\n"
    "005,开不了机,质量\n006,地址填错,物流\n007,异味,质量\n008,延迟送达,物流\n"
    "009,无法充电,质量\n010,包装破损,物流\n"
).encode()

# 超长单行:单个输入单元格里粘贴了数万字符的运行日志(含逗号,按 CSV 规范加引号),
# 单行数十 KB。日志单元格由 csv 模块按规范写出:带引号单元格里的逗号不是分隔符。
_LONG_CELL_PREFIX = (
    "2026-09-27 10:23:01 INFO 收到客户端请求,开始处理订单流程,读取配置项共 42 项;"
    "2026-09-27 10:23:02 WARN 缓存未命中,回源查询耗时 187ms;"
)

# JSONL 超长行:每行一个 JSON 对象,单条记录的输入字段带数万字符日志,
# 单行数十 KB(JSONL 路径逐行 json.loads,无按行截断)。
_JSONL_LONG_CELL = (
    "2026-09-27 10:23:01 INFO 收到客户端请求,处理订单流程,读取配置项共 42 项,回源查询耗时 187ms;"
)


def _jsonl_long_line_text() -> str:
    import json as _json

    lines = [_json.dumps({"编号": "001", "描述": "杯子破损", "类别": "质量"}, ensure_ascii=False)]
    lines.append(
        _json.dumps({"编号": "002", "描述": "物流未更新", "类别": "物流"}, ensure_ascii=False)
    )
    for i in range(3, 11):
        lines.append(
            _json.dumps(
                {
                    "编号": f"{i:03d}",
                    "描述": _JSONL_LONG_CELL * (400 + i * 40),
                    "类别": "质量" if i % 2 else "物流",
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) + "\n"


_JSONL_SAMPLE = (
    "\n".join(
        (
            '{"编号": "001", "描述": "杯子破损", "类别": "质量"}',
            '{"编号": "002", "描述": "物流未更新", "类别": "物流"}',
        )
    )
    + "\n"
).encode()
_JSONL_FULL = _jsonl_long_line_text().encode()

# 重复表头:导出工具把表头行追加在数据中部(常见拼接产物),表头行会变成一条普通数据。
_DUP_HEADER_FULL = (
    "编号,客户描述,类别\n"
    "001,杯子破损,质量\n002,物流未更新,物流\n003,屏幕碎裂,质量\n004,快递丢失,物流\n"
    "编号,客户描述,类别\n"
    "005,开不了机,质量\n006,地址填错,物流\n007,异味,质量\n008,延迟送达,物流\n"
    "009,无法充电,质量\n010,包装破损,物流\n"
).encode()

# 全角数字:编号与描述里出现全角数字(０１２３),零密钥路径不做全角/半角归一。
_FW_FULL = (
    "编号,客户描述,类别\n"
    "００１,订单１２３号商品杯子破损,质量\n００２,物流未更新,物流\n００３,屏幕碎裂第２次,质量\n"
    "００４,快递丢失,物流\n００５,开不了机,质量\n００６,地址填错,物流\n"
    "００７,异味,质量\n００８,延迟送达,物流\n００９,无法充电,质量\n０１０,包装破损,物流\n"
).encode()
_FW_SAMPLE = (
    "编号,客户描述,类别\n００１,订单１２３号商品杯子破损,质量\n００２,物流未更新,物流\n".encode()
)


# 只有 1 行数据的样例:用户拿一条记录试用产品;表头+1 行共 2 行,该行本身有标签,
# 拦截纯粹因为「一条样例不足以完成配对核验」,与缺标签/脏数据无关。
_ONE_ROW_SAMPLE = "编号,客户描述,类别\n001,杯子破损,质量\n".encode()

# 目标列全空:表头有「类别」但所有值为空(导出漏了标签值列,只保留了列名)。
_ALL_EMPTY_TARGET_SAMPLE = "编号,客户描述,类别\n001,杯子破损,\n002,物流未更新,\n".encode()
_ALL_EMPTY_TARGET_FULL = (
    "编号,客户描述,类别\n" + "".join(f"{i:03d},问题{i},\n" for i in range(1, 11))
).encode()

# 无 BOM 的 UTF-16:Excel「Unicode 文本」导出通常带 BOM,经其他工具转存/再导出可能丢 BOM;
# 交错的 NUL/换行字节让 utf-8-sig 与 gb18030 兜底都解不开(测试钉住这一点,防夹具漂移)。
_UTF16_NOBOM_SAMPLE = _UTF16_ROWS.encode("utf-16-le")
_UTF16_NOBOM_FULL = (
    "编号\t客户描述\t类别\n"
    + "".join(f"{i:03d}\t问题{i}\t{'质量' if i % 2 else '物流'}\n" for i in range(1, 11))
).encode("utf-16-le")

# 答案列单一取值:样例与全量所有行都是同一类别(某一时期只产生单一类别工单,
# 或标注者把所有行标成了同一类)——样例取满 10 行,让不均衡检出有机会说话。
_CONSTANT_TARGET_SAMPLE = (
    "编号,客户描述,类别\n" + "".join(f"{i:03d},问题{i},质量\n" for i in range(1, 11))
).encode()
_CONSTANT_TARGET_FULL = _CONSTANT_TARGET_SAMPLE


def _long_line_text() -> str:
    import csv as _csv
    import io as _io

    buffer = _io.StringIO()
    writer = _csv.writer(buffer, lineterminator="\n")
    writer.writerow(("编号", "问题描述", "类别"))
    writer.writerow(("001", _LONG_CELL_PREFIX * 520, "质量"))
    writer.writerow(("002", "物流未更新", "物流"))
    for i in range(3, 11):
        writer.writerow(
            (f"{i:03d}", _LONG_CELL_PREFIX * (300 + i * 20), "质量" if i % 2 else "物流")
        )
    return buffer.getvalue()


# 样例只取表头 + 前两条(单元格内无换行,按行切分安全);全量含全部十条超长行。
_LONG_LINE_LINES = _long_line_text().splitlines(keepends=True)
_LONG_LINE_SAMPLE = "".join(_LONG_LINE_LINES[:3]).encode()
_LONG_LINE_FULL = "".join(_LONG_LINE_LINES).encode()


def builtin_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            scenario_id="clean-classification",
            goal="根据客户首次描述判断售后类别",
            sample=_CLEAN_SAMPLE,
            sample_name="工单.csv",
            full=_CLEAN_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            tags=("classification",),
        ),
        ScenarioSpec(
            scenario_id="dirty-missing-label-column",
            goal="根据客户首次描述判断售后类别",
            sample=_CLEAN_SAMPLE,
            sample_name="工单.csv",
            full="编号,客户描述\n001,杯子破损\n002,物流未更新\n".encode(),
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:validate_full",
            expect_note="全量缺答案列必须拦在全量验证",
            tags=("dirty-data",),
        ),
        ScenarioSpec(
            scenario_id="ambiguous-labels-user-mismatch",
            goal="根据客户首次描述判断售后类别",
            sample=_CLEAN_SAMPLE,
            sample_name="工单.csv",
            full=_CLEAN_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:blind_verification",
            user_answers=_contradictory,
            expect_note="用户盲标与标签矛盾必须拦在盲标关",
            tags=("ambiguous",),
        ),
        ScenarioSpec(
            scenario_id="too-few-groups",
            goal="根据客户首次描述判断售后类别",
            sample=_CLEAN_SAMPLE,
            sample_name="工单.csv",
            full="编号,客户描述,类别\n001,杯子破损,质量\n002,物流未更新,物流\n".encode(),
            target_column="类别",
            group_columns=("客户描述",),
            expect="blocked_at:materialize",
            expect_note="独立分组不足 3 个不能形成三分区",
            tags=("small-sample",),
        ),
        ScenarioSpec(
            scenario_id="forecast-shaped-goal",
            goal="根据披露文本预测未来20个交易日价格是否上涨",
            sample=("event_id,披露文本,方向\ne1,收入上升,上涨\ne2,收入持平,未上涨\n").encode(),
            sample_name="events.csv",
            full=(
                "event_id,披露文本,方向\n"
                "e1,收入上升,上涨\ne2,收入持平,未上涨\ne3,收入下降,未上涨\n"
                "e4,收入大增,上涨\ne5,收入持平,未上涨\ne6,收入上升,上涨\n"
            ).encode(),
            target_column="方向",
            group_columns=("event_id",),
            expect="passes",
            expect_note=(
                "通过但带泄漏预警(预测型目标在基础分析收到时间分区警告);"
                "零密钥路径尚不能建立时间方案,配置 Agent 后应走时间契约——已知局限,非缺陷"
            ),
            tags=("forecast", "known-limitation"),
        ),
        ScenarioSpec(
            scenario_id="high-cardinality-target",
            goal="根据描述生成对应工单编号",
            sample=("描述,工单编号\n杯子破损,GD-001\n物流未更新,GD-002\n").encode(),
            sample_name="tickets.csv",
            full=(
                "描述,工单编号\n"
                "杯子破损,GD-001\n物流未更新,GD-002\n屏幕碎裂,GD-003\n快递丢失,GD-004\n"
                "开不了机,GD-005\n地址填错,GD-006\n异味,GD-007\n延迟送达,GD-008\n"
            ).encode(),
            target_column="工单编号",
            group_columns=(),
            expect="passes",
            expect_note=(
                "通过且带「可能选错答案列」预警(高基数目标早期警告已上线,M5/task-001);"
                "抽取类任务的高基数是合法的,预警不阻断"
            ),
            tags=("high-cardinality", "known-gap"),
        ),
        ScenarioSpec(
            scenario_id="dirty-label-variants",
            goal="根据客户首次描述判断售后类别",
            sample=("编号,客户描述,类别\n001,杯子破损,质量\n002,物流未更新,物流。\n").encode(),
            sample_name="工单.csv",
            full=(
                "编号,客户描述,类别\n"
                "001,杯子破损,质量\n002,物流未更新,物流。\n003,屏幕碎裂,质量。\n"
                "004,快递丢失,物流\n005,开不了机,质量\n006,地址填错,物流。\n"
                "007,异味,质量\n008,延迟送达,物流\n009,无法充电,质量。\n010,包装破损,物流\n"
            ).encode(),
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note="通过且带「标签多种写法」预警(变体检出已上线);变体在训练中会被当不同答案,预警建议归一",
            tags=("dirty-data", "label-variants"),
        ),
        ScenarioSpec(
            scenario_id="long-text-inputs",
            goal="根据客户投诉详情判断严重程度",
            sample=(
                "编号,投诉详情,严重程度\n001," + "非常冗长的投诉描述。" * 80 + ",高\n"
                "002," + "等待多日仍未送达。" * 80 + ",低\n"
            ).encode(),
            sample_name="投诉.csv",
            full=(
                "编号,投诉详情,严重程度\n"
                + "".join(
                    f"{i:03d}," + "投诉内容细节。" * (60 + i % 40) + f",{'高' if i % 2 else '低'}\n"
                    for i in range(1, 11)
                )
            ).encode(),
            target_column="严重程度",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "数据层通过;长文本的截断风险由训练前检查(真实 tokenizer)负责,不在矩阵覆盖内——边界如实记录"
            ),
            tags=("long-text", "boundary-note"),
        ),
        ScenarioSpec(
            scenario_id="duplicate-inputs",
            goal="根据客户首次描述判断售后类别",
            sample=(
                "编号,客户描述,类别\n001,杯子破损,质量\n001,杯子破损,质量\n002,物流未更新,物流\n"
            ).encode(),
            sample_name="工单.csv",
            full=(
                "编号,客户描述,类别\n"
                + "".join(
                    f"{'001' if i % 2 else f'{i:03d}'},{'杯子破损' if i % 2 else f'问题{i}'},质量\n"
                    for i in range(1, 13)
                )
            ).encode(),
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note="通过且带「输入完全重复」预警(重复检出已上线);重复行答案一致,仅样本量虚高",
            tags=("dirty-data", "duplicates"),
        ),
        ScenarioSpec(
            scenario_id="same-input-conflicting-labels",
            goal="根据客户首次描述判断售后类别",
            sample="编号,客户描述,类别\n001,杯子破损,质量\n002,物流未更新,物流\n".encode(),
            sample_name="工单.csv",
            full=(
                "编号,客户描述,类别\n"
                "001,杯子破损,质量\n001,杯子破损,物流\n002,物流未更新,物流\n"
                "003,屏幕碎裂,质量\n004,快递丢失,物流\n005,开不了机,质量\n"
                "006,地址填错,物流\n007,异味,质量\n008,延迟送达,物流\n"
            ).encode(),
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:validate_full",
            expect_note="同一输入对应不同答案(矛盾标签)必须被拦:模型无法同时满足两个答案",
            tags=("ambiguous", "conflicting-labels"),
        ),
        ScenarioSpec(
            scenario_id="multi-source-boundary",
            goal="用工单表和审核表联合判断售后类别",
            sample="描述,类别\n杯子破损,质量\n物流未更新,物流\n".encode(),
            sample_name="工单.csv",
            full=(
                "描述,类别\n"
                "杯子破损,质量\n物流未更新,物流\n屏幕碎裂,质量\n快递丢失,物流\n"
                "开不了机,质量\n地址填错,物流\n异味,质量\n延迟送达,物流\n"
                "无法充电,质量\n包装破损,物流\n"
            ).encode(),
            target_column="类别",
            group_columns=(),
            expect="passes",
            expect_note=(
                "边界如实记录:零密钥路径仅分析主资料;多源组合需配置 Agent——"
                "目标提及多表时基础分析不做组合,按已知边界通过"
            ),
            tags=("boundary", "multi-source"),
        ),
        ScenarioSpec(
            scenario_id="severe-class-imbalance",
            goal="根据客户首次描述判断是否需要人工复核",
            sample=("编号,描述,需复核\n001,普通问题,否\n002,投诉升级,是\n").encode(),
            sample_name="工单.csv",
            full=(
                "编号,描述,需复核\n"
                + "".join(
                    f"{i:03d},问题{i},{'是' if i % 10 == 0 else '否'}\n" for i in range(1, 21)
                )
            ).encode(),
            target_column="需复核",
            group_columns=("编号",),
            expect="passes",
            expect_note="通过且带「分布严重不均衡」预警(多数类 90%:全猜多数类即 90% 准确率,准确率会骗人)",
            tags=("imbalance",),
        ),
        ScenarioSpec(
            scenario_id="gbk-encoded-upload",
            goal="根据客户首次描述判断售后类别",
            sample=_GBK_SAMPLE,
            sample_name="工单.csv",
            full=_GBK_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "实测结局:create 不被拦——utf-8-sig 解码失败后入口自动回退 gb18030"
                "(GBK 的超集)正确解码,全旅程无乱码;入口暂无需手动选择编码,"
                "需在入口选择编码的是 gb18030 也解不开的冷门编码——边界如实记录"
            ),
            tags=("encoding", "gbk", "boundary-note"),
        ),
        ScenarioSpec(
            scenario_id="wide-table",
            goal="根据客户首次描述判断售后类别",
            sample=_WIDE_SAMPLE,
            sample_name="宽表.csv",
            full=_WIDE_FULL,
            target_column="类别",
            group_columns=("编号",),
            excluded_columns=_WIDE_JUNK_COLUMNS,
            expect="passes",
            expect_note=(
                "31 列宽表:用户在入口排除 28 个无关列,只留编号(分组)+客户描述(输入)+类别(答案);"
                "宽表不淹死基础分析——列角色由用户选择决定,被排除列仅作记录"
            ),
            tags=("wide-table", "column-selection"),
        ),
        ScenarioSpec(
            scenario_id="mixed-type-column",
            goal="根据客户首次描述判断售后类别",
            sample=_MIXED_SAMPLE,
            sample_name="工单.csv",
            full=_MIXED_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "边界如实记录:工单号列混杂数字样(10023)与文本样(AB-1002)编号,"
                "当前零密钥路径不做类型区分,一律按文本(strip 后原样)进入输入;"
                "如需数字语义须先在数据侧规整——已知边界,非缺陷"
            ),
            tags=("mixed-type", "boundary-note"),
        ),
        ScenarioSpec(
            scenario_id="chinese-punctuation-variants",
            goal="根据客户首次描述判断售后类别",
            sample=("编号,客户描述,类别\n001,杯子破损,质量\n002,物流未更新,物流\n").encode(),
            sample_name="工单.csv",
            full=(
                "编号,客户描述,类别\n"
                "001,杯子破损。,质量\n002,物流未更新,物流\n003,屏幕碎裂！,质量\n"
                "004,快递丢失?,物流\n005,开不了机。,质量\n006,地址填错，地址错了,物流\n"
                "007,异味,质量\n008,延迟送达!,物流\n009,无法充电……,质量\n010,包装破损,物流\n"
            ).encode(),
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "边界如实记录:零密钥路径不做标点归一,全角/半角标点与句尾标点差异"
                "原样进入训练;基础分析的变体检出只针对答案列,输入侧标点差异由"
                "真实预览由用户核对。语义是否受影响由用户判断。"
            ),
            tags=("boundary", "punctuation"),
        ),
        ScenarioSpec(
            scenario_id="utf16-excel-export",
            goal="根据客户首次描述判断售后类别",
            sample=_UTF16_SAMPLE,
            sample_name="工单.csv",
            full=_UTF16_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "Excel「Unicode 文本」导出(UTF-16 带 BOM + Tab 分隔)自动识别:"
                "带 BOM 的 UTF-16 按证据解码(FF FE/FE FF 开头不可能是合法 UTF-8 或 GBK),"
                "Tab 分隔由嗅探器识别;无 BOM 的 UTF-16 仍被明确拒绝——边界如实记录"
            ),
            tags=("encoding", "utf16", "excel"),
        ),
        ScenarioSpec(
            scenario_id="ultra-long-single-line",
            goal="根据客户粘贴的运行日志判断问题类别",
            sample=_LONG_LINE_SAMPLE,
            sample_name="日志工单.csv",
            full=_LONG_LINE_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "边界如实记录:单个输入单元格约数万字符(单行数十 KB)不被数据层拦截,"
                "数据层只验证结构与答案保留;是否截断由训练前预检用真实 tokenizer 测量,"
                "语义是否受影响由用户在真实预览核对"
            ),
            tags=("long-text", "boundary-note"),
        ),
        ScenarioSpec(
            scenario_id="empty-label-rows-in-full",
            goal="根据客户首次描述判断售后类别",
            sample=_CLEAN_SAMPLE,
            sample_name="工单.csv",
            full=_EMPTY_LABEL_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:validate_full",
            expect_note=(
                "实测结局:样例有标签、全量混入 2 条空答案行(005/008)——既不被静默跳过,"
                "也不带病通过,而是被全量验证硬拦:blocking 问题「全量存在缺少监督答案的记录…」"
                "点名行号;用户须补标签或删行后重新验证"
            ),
            tags=("dirty-data", "empty-labels"),
        ),
        ScenarioSpec(
            scenario_id="spaced-header-names",
            goal="根据客户首次描述判断售后类别",
            sample=_SPACED_SAMPLE,
            sample_name="工单.csv",
            full=_SPACED_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:baseline_analysis",
            expect_note=(
                "实测结局:表头「 编号 / 类别」带前后空格,入口读表不剥空格,"
                "零密钥路径的答案列按精确名匹配——「类别」匹配不上,在基础分析即被拦;"
                "报错把带空格的真实列名原样列出(可用:[' 编号 ','客户描述',' 类别']),"
                "用户能看见差异。边界:若改选确切带空格列名可全程通过,空格随之进入"
                "指令与标签——如实记录,不做自动剥空格"
            ),
            tags=("dirty-data", "header-hygiene"),
        ),
        ScenarioSpec(
            scenario_id="jsonl-long-line",
            goal="根据客户粘贴的运行日志判断问题类别",
            sample=_JSONL_SAMPLE,
            sample_name="日志工单.jsonl",
            full=_JSONL_FULL,
            full_name="full.jsonl",
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "边界如实记录:JSONL 单行数十 KB(单条记录的输入字段带数万字符日志)"
                "逐行 json.loads 不被数据层拦截,数据层只验证结构与答案保留;"
                "是否截断由训练前预检用真实 tokenizer 测量,语义是否受影响由用户在真实预览核对"
            ),
            tags=("long-text", "jsonl", "boundary-note"),
        ),
        ScenarioSpec(
            scenario_id="duplicate-header-rows-in-full",
            goal="根据客户首次描述判断售后类别",
            sample=_CLEAN_SAMPLE,
            sample_name="工单.csv",
            full=_DUP_HEADER_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:validate_full",
            expect_note=(
                "实测结局:导出拼接产生的重复表头行被全量验证硬拦(blocking 问题"
                "「与表头完全相同」点名行号);曾为已知缺口(静默成为训练样本),已清偿——"
                "用户删除表头行后可重新验证;没有自动删行"
            ),
            tags=("dirty-data", "header-hygiene"),
        ),
        ScenarioSpec(
            scenario_id="full-width-digits",
            goal="根据客户首次描述判断售后类别",
            sample=_FW_SAMPLE,
            sample_name="工单.csv",
            full=_FW_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="passes",
            expect_note=(
                "边界如实记录:编号与描述中的全角数字(０１２３)不做全角/半角归一,"
                "按原样文本进入输入与分组;与标点变体同理,语义是否受影响由用户判断,"
                "需要数字语义时先在数据侧规整——已知边界,非缺陷"
            ),
            tags=("boundary", "full-width"),
        ),
        ScenarioSpec(
            scenario_id="single-row-sample",
            goal="根据客户首次描述判断售后类别",
            sample=_ONE_ROW_SAMPLE,
            sample_name="工单.csv",
            full=_CLEAN_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:contrast_check",
            expect_note=(
                "实测结局:样例只有 1 行数据(2 行含表头)——create 与基础分析都不拦"
                "(分析如实观察「答案列非空取值 1/1 行,共 1 类」),旅程在对比核验被拦:"
                "「对比核验需要至少两条答案不同的已标注行。」即使全量数据正常,"
                "一条样例也不足以让用户完成配对核验;用户须至少提供 2 条答案不同的已标注样例"
            ),
            tags=("small-sample", "negative-scenario"),
        ),
        ScenarioSpec(
            scenario_id="all-empty-target-column",
            goal="根据客户首次描述判断售后类别",
            sample=_ALL_EMPTY_TARGET_SAMPLE,
            sample_name="工单.csv",
            full=_ALL_EMPTY_TARGET_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:contrast_check",
            expect_note=(
                "实测结局:答案列表头存在但所有值为空(样例与全量皆空)——基础分析不拦,"
                "finding 如实观察「答案列非空取值 0/2 行,共 0 类」,预览逐行标 needs_label;"
                "旅程在对比核验被拦:「对比核验需要至少两条答案不同的已标注行。」"
                "不静默跳过、不自动补值;用户须先补标签再重走旅程"
            ),
            tags=("dirty-data", "empty-target"),
        ),
        ScenarioSpec(
            scenario_id="utf16-no-bom-rejected",
            goal="根据客户首次描述判断售后类别",
            sample=_UTF16_NOBOM_SAMPLE,
            sample_name="工单.csv",
            full=_UTF16_NOBOM_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:create",
            expect_note=(
                "实测结局:无 BOM 的 UTF-16 字节流(「Unicode 文本」导出经其他工具转存丢 BOM)"
                "utf-8-sig 解不开,gb18030 兜底也解不开(前导字节后跟交错的 NUL/换行字节),"
                "create 即明确拒绝:「无法解码文件，请明确指定编码；原始数据未修改。」;"
                "恢复路径已实测:入口显式指定编码(如 utf-16-le)即可正确解码并继续旅程;"
                "与带 BOM 的 utf16-excel-export(按 BOM 证据自动识别)构成编码家族的完整边界"
            ),
            tags=("encoding", "utf16", "excel"),
        ),
        ScenarioSpec(
            scenario_id="constant-target",
            goal="根据客户首次描述判断售后类别",
            sample=_CONSTANT_TARGET_SAMPLE,
            sample_name="工单.csv",
            full=_CONSTANT_TARGET_FULL,
            target_column="类别",
            group_columns=("编号",),
            expect="blocked_at:contrast_check",
            expect_note=(
                "实测结局:答案列只有一种取值——create 与基础分析都不拦,基础分析如实发出"
                "「分布严重不均衡:「质量」占 100%(10 行),全猜这一类就有 100% 准确率」的"
                "非阻断预警;旅程在对比核验被拦:「对比核验需要至少两条答案不同的已标注行。」"
                "单一类别连配对核验都组不成,模型没有可学习的区分边界;"
                "用户须让数据覆盖至少两个类别,或确认答案列本身选错"
            ),
            tags=("dirty-data", "single-class"),
        ),
    ]


def _contradictory(session):
    answers = {row.row_id: row.target for row in session.full_data.preview.rows}
    first = sorted(answers)[0]
    answers[first] = "和所有标签都不同的答案"
    return answers
