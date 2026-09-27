"""Event-to-market labels from explicit local prices and a supplied session calendar."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import Field, model_validator

from src.workbench.intake_models import Contract, SampleSource


class ForecastLabelsStep(Contract):
    operation: Literal["forecast_labels"] = "forecast_labels"
    price_source: str = Field(min_length=1)
    calendar_source: str = Field(min_length=1)
    event_id_column: str = Field(min_length=1)
    symbol_column: str = Field(min_length=1)
    available_at_column: str = Field(min_length=1)
    price_symbol_column: str = Field(min_length=1)
    price_close_at_column: str = Field(min_length=1)
    price_value_column: str = Field(min_length=1)
    calendar_close_at_column: str = Field(min_length=1)
    price_basis: Literal["split_adjusted_close", "total_return_adjusted_close"] = Field(
        description="须依据行情来源明确价格调整口径，不能只按列名猜测或混用。"
    )
    observation_end: str = Field(
        description="带时区的资料观察截止时间；之后的价格不得用于填造已知标签。"
    )
    horizon_sessions: int = Field(default=20, strict=True, ge=1)
    prefix: str = Field(default="forecast_", min_length=1)

    @model_validator(mode="after")
    def valid_contract(self):
        instant(self.observation_end)
        if self.price_source == self.calendar_source:
            raise ValueError("行情与交易日历必须是分别可核查的数据来源。")
        return self


def instant(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("时间必须是包含时区的 ISO 文本。")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("时间不是有效 ISO 时间戳；不能用报告期或日期替代公开时间。") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("时间必须带时区，不能猜测当地时间或美股收盘时间。")
    return parsed.astimezone(timezone.utc)


def _price(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("价格必须是明确的正数。")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("价格不是有效数字。") from None
    if not result.is_finite() or result <= 0:
        raise ValueError("价格必须是有限正数，不能用零或空值填补缺行情。")
    return result


def output_columns(step: ForecastLabelsStep) -> list[str]:
    return [
        step.prefix + name
        for name in (
            "prediction_at",
            "label_end_at",
            "reference_price",
            "target_price",
            "realized_return",
            "direction",
            "status",
        )
    ]


def build_forecast_labels(
    events: list[dict], prices: SampleSource, calendar: SampleSource, step: ForecastLabelsStep
) -> dict:
    """Preserve every event, identify missing evidence, and never infer market sessions."""
    observation = instant(step.observation_end)
    columns = output_columns(step)
    result = {
        "rows": [{**event, **dict.fromkeys(columns)} for event in events],
        "issues": [],
        "origins": {index: [] for index in range(len(events))},
        "columns": columns,
    }

    def issue(code, message, *, index=None, refs=None, severity="blocking"):
        result["issues"].append(
            {
                "code": code,
                "message": message,
                "event_index": index,
                "refs": refs or [],
                "severity": severity,
            }
        )

    def ref(source, row):
        return {"source_digest": source.digest, "row_id": row.row_id}

    if any(set(columns) & set(event) for event in events):
        raise ValueError("预测标签输出字段与原始字段冲突，请修改prefix。")
    required_prices = {
        step.price_symbol_column,
        step.price_close_at_column,
        step.price_value_column,
    }
    if (
        not required_prices <= set(prices.columns)
        or step.calendar_close_at_column not in calendar.columns
    ):
        raise ValueError("行情或交易日历缺少已声明的字段。")
    schedule = {}
    for row in calendar.rows:
        try:
            close = instant(row.values.get(step.calendar_close_at_column))
            if close in schedule:
                raise ValueError("交易日历含重复收盘时点，不能重复计算交易日。")
            schedule[close] = row
        except ValueError as exc:
            issue("invalid_calendar", str(exc), refs=[ref(calendar, row)])
    if not schedule:
        issue("empty_calendar", "交易日历为空，不能把行情行数或自然日数当成交易日。")
    quotes = {}
    for row in prices.rows:
        try:
            symbol = row.values.get(step.price_symbol_column)
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("行情证券标识必须是非空文本，不能猜测历史ticker映射。")
            close = instant(row.values.get(step.price_close_at_column))
            if close > observation:
                continue  # A future quote in a supplied file cannot mature a future label.
            price = _price(row.values.get(step.price_value_column))
            if (symbol, close) in quotes:
                raise ValueError("同证券同收盘时点有重复行情，需明确来源或修订版本。")
            if close not in schedule:
                raise ValueError("行情收盘时点不在提供的交易日历内，请核对时区与交易所。")
            quotes[(symbol, close)] = (price, row)
        except ValueError as exc:
            issue("invalid_price", str(exc), refs=[ref(prices, row)])
    if any(item["severity"] == "blocking" for item in result["issues"]):
        for row in result["rows"]:
            row[step.prefix + "status"] = "invalid_sources"
        return result

    closes = sorted(schedule)
    event_identity = {}
    for index, event in enumerate(events):
        output = result["rows"][index]
        output[step.prefix + "status"] = "invalid_event"
        try:
            event_id, symbol = event.get(step.event_id_column), event.get(step.symbol_column)
            if any(not isinstance(value, str) or not value.strip() for value in (event_id, symbol)):
                raise ValueError("每条披露须有明确事件标识和证券标识。")
            available = instant(event.get(step.available_at_column))
            identity = (symbol, available)
            if event_id in event_identity and event_identity[event_id] != identity:
                raise ValueError("同一事件标识对应不同证券或公开时间，请核对事件分组。")
            event_identity[event_id] = identity
            start = bisect_right(closes, available)
            if start == 0:
                raise ValueError("交易日历缺少公开时间之前的覆盖，无法证明选中的是之后首个收盘。")
            if start + step.horizon_sessions >= len(closes):
                raise ValueError("交易日历覆盖不足，无法确定完整预测窗口；不能按自然日推算。")
            prediction, label_end = closes[start], closes[start + step.horizon_sessions]
            output[step.prefix + "prediction_at"] = prediction.isoformat()
            output[step.prefix + "label_end_at"] = label_end.isoformat()
            result["origins"][index].extend(
                ref(calendar, schedule[close])
                for close in closes[start - 1 : start + step.horizon_sessions + 1]
            )
            if label_end > observation:
                output[step.prefix + "status"] = "label_not_observed"
                issue(
                    "label_not_observed",
                    "预测窗口尚未结束，保留原事件等待标签，不填成未上涨。",
                    index=index,
                    severity="review",
                )
                continue
            window = closes[start : start + step.horizon_sessions + 1]
            missing = [close.isoformat() for close in window if (symbol, close) not in quotes]
            if missing:
                raise ValueError(
                    f"预测窗口缺少{len(missing)}条行情（首个{missing[0]}）；不跨过缺失交易日计算。"
                )
            for close in window:
                result["origins"][index].append(ref(prices, quotes[(symbol, close)][1]))
            baseline, final = quotes[(symbol, prediction)][0], quotes[(symbol, label_end)][0]
            output.update(
                {
                    step.prefix + "reference_price": str(baseline),
                    step.prefix + "target_price": str(final),
                    step.prefix + "realized_return": str((final / baseline - 1).normalize()),
                    step.prefix + "direction": "上涨" if final > baseline else "未上涨",
                    step.prefix + "status": "ready",
                }
            )
        except ValueError as exc:
            issue("invalid_forecast_event", str(exc), index=index)
    issue(
        "forecast_contract_review",
        f"按提供的完整交易日历，在资料可用后的首个收盘至其后{step.horizon_sessions}个交易日计算标签；"
        f"行情口径声明为{step.price_basis}。请核对日历覆盖、行情调整口径、证券历史身份和公开时间。"
        "价格源的口径属于需核对的声明，本程序不凭价格列名证明其调整正确。",
        severity="review",
    )
    return result
