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
                "已知待办:高基数目标(每行唯一)目前不被任何关卡拦截——"
                "学习逐行唯一标签几乎必然失败,应在旅程更早处给出警告/拦截(M5 候选)"
            ),
            tags=("high-cardinality", "known-gap"),
        ),
    ]


def _contradictory(session):
    answers = {row.row_id: row.target for row in session.full_data.preview.rows}
    first = sorted(answers)[0]
    answers[first] = "和所有标签都不同的答案"
    return answers
