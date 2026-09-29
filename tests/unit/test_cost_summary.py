"""成本账:时长实测、功耗/电价明示为估计、API 对比口径如实。"""

from src.workbench.cost_summary import cost_lines, summarize_run_cost


def test_duration_and_electricity_are_estimated_honestly():
    account = summarize_run_cost(
        {
            "run_id": "wb-x",
            "status": "succeeded",
            "started_at": "2026-09-27T10:00:00+00:00",
            "finished_at": "2026-09-27T11:30:00+00:00",
        },
        device="mps",
        device_watts=80,
        electricity_price=0.6,
    )
    assert account["duration_hours"] == 1.5
    assert account["electricity_cost_estimated"] == round(1.5 * 80 / 1000 * 0.6, 2)
    assert "估计值" in account["hardware_note"]
    lines = "\n".join(cost_lines(account))
    assert "1.5 小时" in lines and "估计" in lines and "不是电表读数" in lines


def test_missing_timing_is_stated_not_guessed():
    account = summarize_run_cost({"run_id": "wb-y", "status": "failed"})
    assert account["duration_hours"] is None
    assert "算不出时长" in "\n".join(cost_lines(account))


def test_api_comparison_uses_user_assumptions():
    account = summarize_run_cost(
        {
            "run_id": "wb-z",
            "status": "succeeded",
            "started_at": "2026-09-27T10:00:00+00:00",
            "finished_at": "2026-09-27T12:00:00+00:00",
        },
        device="mps",
        api_price_per_million_tokens=8.0,
        expected_monthly_queries=1_000_000,
        tokens_per_query=512,
    )
    comparison = account["api_comparison"]
    assert comparison["api_monthly_cost_estimated"] == round(
        512 * 8.0, 2
    )  # 100万次×512token=5.12亿token
    lines = "\n".join(cost_lines(account))
    assert "1,000,000 次月调用量" in lines
    assert "口径不同" in lines or "量级比较" in lines
