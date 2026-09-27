"""Time policies and forecast source requirements are visible before materialization."""

import pytest

pytest.importorskip("streamlit")

from tests.unit.test_data_intake_ui import button, data_page  # noqa: F401
from tests.unit.test_forecast_full_sources import forecast_session
from tests.unit.test_temporal_intake_entrypoints import temporal_session


def test_time_policy_hides_random_controls_and_displays_actual_exclusions(data_page):  # noqa: F811
    service, _, page = data_page
    session = temporal_session(service)
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("2024-02-01T00:00:00Z" in item.value for item in page.markdown)
    assert any("随机比例与种子不生效" in item.value for item in page.info)
    assert any("标签窗口尚未成熟" in item.value for item in page.code)
    assert not any(
        item.label in {"验证集比例", "独立测试集比例", "可复现分区种子"}
        for item in page.number_input
    )
    button(page, "生成数据集版本").click().run()
    assert not page.exception
    assert any("纳入 3 条" in item.value and "保留 2 条" in item.value for item in page.info)
    exclusion_table = next(
        table.value for table in page.dataframe if "排除原因" in table.value.columns
    )
    assert set(exclusion_table["排除原因"]) == {
        "观察截止时标签尚未成熟",
        "训练标签窗口跨越验证起点",
    }
    assert service.load(session.session_id).dataset.statistics["row_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_forecast_full_validation_lists_calendar_and_prices_and_reuses_only_full_sources(data_page):  # noqa: F811
    service, _, page = data_page
    session = forecast_session(service, scope="full")
    page.run()
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    assert any("calendar、main、prices" in item.value for item in page.markdown)
    button(page, "验证已提供的全部全量资料").click().run()
    assert not page.exception
    updated = service.load(session.session_id)
    assert set(updated.full_data.sources) == {"main", "calendar", "prices"}
    assert updated.full_data.preview.rows[0].target == "上涨"
