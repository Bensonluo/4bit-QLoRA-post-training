"""Per-run cost accounts: time, hardware, estimated electricity, optional API comparison.

Every number is either measured (duration) or an explicitly labeled estimate
(device power draw, electricity price). Nothing here claims business ROI; it
answers the north-star question "可承担的成本" with numbers the user can check
and assumptions they can override.
"""

from __future__ import annotations

from typing import Any

# 典型整卡/整机功耗估计(瓦),明示为估计值,可被调用方覆盖。
DEFAULT_DEVICE_WATTS = {"cuda": 300, "mps": 80, "cpu": 60}
# 居民电价默认值(元/千瓦时),仅作参考估计。
DEFAULT_ELECTRICITY_PRICE = 0.6


def summarize_run_cost(
    record: dict,
    *,
    device: str = "mps",
    device_watts: int | None = None,
    electricity_price: float = DEFAULT_ELECTRICITY_PRICE,
    api_price_per_million_tokens: float | None = None,
    expected_monthly_queries: int | None = None,
    tokens_per_query: int = 512,
) -> dict[str, Any]:
    """Build an honest cost account from one training run record."""
    started = record.get("started_at")
    finished = record.get("finished_at")
    duration_hours: float | None = None
    if started and finished:
        from datetime import datetime

        try:
            begin = datetime.fromisoformat(started)
            end = datetime.fromisoformat(finished)
            duration_hours = max((end - begin).total_seconds() / 3600.0, 0.0)
        except ValueError:
            duration_hours = None
    watts = device_watts if device_watts is not None else DEFAULT_DEVICE_WATTS.get(device, 60)
    electricity_cost = (
        round(duration_hours * watts / 1000 * electricity_price, 2)
        if duration_hours is not None
        else None
    )
    account: dict[str, Any] = {
        "kind": "run_cost",
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "duration_hours": round(duration_hours, 3) if duration_hours is not None else None,
        "device": device,
        "device_watts_estimated": watts,
        "electricity_price_assumed": electricity_price,
        "electricity_cost_estimated": electricity_cost,
        "hardware_note": (
            "功耗为该设备类别的估计值,不是实测;电费按居民电价估计。"
            "硬件购置成本不含在内(使用你已有的电脑)。"
        ),
    }
    if (
        api_price_per_million_tokens is not None
        and expected_monthly_queries is not None
        and expected_monthly_queries > 0
        and tokens_per_query > 0
    ):
        monthly_tokens_m = expected_monthly_queries * tokens_per_query / 1_000_000
        account["api_comparison"] = {
            "expected_monthly_queries": expected_monthly_queries,
            "tokens_per_query_assumed": tokens_per_query,
            "api_monthly_cost_estimated": round(monthly_tokens_m * api_price_per_million_tokens, 2),
            "note": (
                "API 成本按你给的单价与调用量估计;本地侧是一次性训练电费,推理在本地另有算力占用。"
                "两者口径不同,仅供量级比较,不是精确账单。"
            ),
        }
    return account


def cost_lines(account: dict[str, Any]) -> list[str]:
    """Plain-language cost summary for a non-expert."""
    lines: list[str] = []
    if account.get("duration_hours") is not None:
        hours = account["duration_hours"]
        shown = f"{hours:.1f} 小时" if hours >= 1 else f"{hours * 60:.0f} 分钟"
        lines.append(f"这次训练实际运行了 {shown}(设备:{account['device']})。")
    else:
        lines.append("这次运行缺少起止时间,算不出时长;成本账不完整。")
    if account.get("electricity_cost_estimated") is not None:
        lines.append(
            f"按 {account['device_watts_estimated']}W 估计功耗与 "
            f"{account['electricity_price_assumed']} 元/度电价,电费约 "
            f"{account['electricity_cost_estimated']} 元——这是估计值,不是电表读数。"
        )
    comparison = account.get("api_comparison")
    if comparison:
        lines.append(
            f"如果改用 API 完成同样的 {comparison['expected_monthly_queries']:,} 次月调用量,"
            f"按 {comparison['tokens_per_query_assumed']} token/次估计,月成本约 "
            f"{comparison['api_monthly_cost_estimated']} 元;本地训练是一次性电费,推理另计。"
        )
        lines.append(comparison.get("note", ""))
    lines.append(account.get("hardware_note", ""))
    return [line for line in lines if line]
