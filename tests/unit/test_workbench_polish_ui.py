"""Verify reading order, recorded comparison values and usable display exits."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from tests.unit.test_home_journey_progress_ui import boot, task

ROOT = Path(__file__).resolve().parents[2]


def test_input_answer_verification_precedes_supporting_materials(monkeypatch, tmp_path):
    service, session = task(tmp_path)
    page = boot(monkeypatch, tmp_path)
    next(item for item in page.button if item.label == "继续任务").click().run()
    assert not page.exception
    labels = [item.label for item in page.main.expander]
    assert labels.index("真实转换预览") < labels.index("数据来源与读取说明")
    assert labels.index("真实转换预览") < labels.index("业务理解与方案修订")
    preview = next(item for item in page.expander if item.label == "真实转换预览")
    assert preview.proto.expanded
    assert any(item.value == "模型实际输入" for item in preview.caption)
    assert any(item.value == "监督答案" for item in preview.caption)
    nodes = list(page.main)
    confirm_index = next(
        i
        for i, item in enumerate(nodes)
        if item.type == "button" and item.label == "确认当前转换含义"
    )
    support_index = next(
        i
        for i, item in enumerate(nodes)
        if item.type == "expander" and item.label == "数据来源与读取说明"
    )
    assert confirm_index < support_index
    assert service.load(session.session_id).revision == session.revision


def comparison(monkeypatch, tmp_path, records):
    import ui.components.domain_adapters as adapters
    from ui import config, runtime

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    monkeypatch.setattr(config, "DOMAINS_DIR", tmp_path)
    monkeypatch.setattr(adapters, "load_eval_data", lambda domain: records)
    return AppTest.from_file(
        str(ROOT / "ui/views/03_Model_Comparison.py"), default_timeout=30
    ).run()


def test_comparison_keeps_zero_and_missing_values_distinct(monkeypatch, tmp_path):
    page = comparison(
        monkeypatch,
        tmp_path,
        [
            {"model": "基座", "overall_accuracy": 0.0},
            {"model": "微调", "overall_accuracy": 0.5, "throughput_per_sec": 100},
        ],
    )
    assert not page.exception
    metrics = page.dataframe[0].value.set_index("指标")
    a, b = [column for column in metrics.columns if column.startswith(("A · ", "B · "))]
    assert metrics.loc["总体准确率", a] == "0.0%"
    assert metrics.loc["总体准确率", "差值（B − A）"] == "+50.0 个百分点"
    assert metrics.loc["平均延迟 (ms)", a] == "—"
    cost = next(
        item.value for item in page.dataframe if "100 万样本耗时" in item.value["指标"].values
    )
    cost = cost.set_index("指标")
    assert cost.loc["100 万样本耗时", a] == "—"
    assert cost.loc["100 万样本耗时", b] == "2.8 小时"
    assert "部署方式" not in cost.index


def test_comparison_includes_groups_present_only_in_either_record(monkeypatch, tmp_path):
    page = comparison(
        monkeypatch,
        tmp_path,
        [
            {"model": "甲", "accuracy_by_type": {"device": 0.6}},
            {"model": "乙", "accuracy_by_type": {"supplier": 0.8}},
        ],
    )
    assert not page.exception
    groups = next(item.value for item in page.dataframe if "分组" in item.value.columns).set_index(
        "分组"
    )
    assert set(groups.index) == {"实体类型 · device", "实体类型 · supplier"}
    assert groups.loc["实体类型 · device", "B · 乙"] == "—"
    assert groups.loc["实体类型 · supplier", "A · 甲"] == "—"


def test_distinct_records_with_same_model_name_remain_comparable(monkeypatch, tmp_path):
    page = comparison(
        monkeypatch,
        tmp_path,
        [
            {
                "model": "same-model",
                "overall_accuracy": 0.2,
                "per_sample": [{"correct": False, "query": "a"}],
            },
            {
                "model": "same-model",
                "overall_accuracy": 0.7,
                "per_sample": [{"correct": False, "query": "b"}],
            },
        ],
    )
    assert not page.exception
    table = page.dataframe[0].value
    assert "A · same-model（记录 1）" in table.columns
    assert "B · same-model（记录 2）" in table.columns
    assert len([item for item in page.expander if "个错误样本" in item.label]) == 2
    a = next(item for item in page.selectbox if item.label == "模型 A")
    a.set_value(1).run()
    assert not page.exception
    b = next(item for item in page.selectbox if item.label == "模型 B")
    assert b.value == 0


@pytest.mark.parametrize(
    "page_path",
    [
        "01_Experiments.py",
        "02_Evaluation.py",
        "03_Model_Comparison.py",
        "04_Registry.py",
        "05_Data_Wizard.py",
        "06_Chat.py",
    ],
)
def test_display_pages_offer_an_available_task_without_execution(monkeypatch, tmp_path, page_path):
    import ui.components.domain_adapters as adapters
    from ui import queries

    service, session = task(tmp_path)
    page = boot(monkeypatch, tmp_path, display=True)
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])
    monkeypatch.setattr(queries, "fetch_model_versions", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(adapters, "load_eval_data", lambda domain: [])
    page.switch_page(f"views/{page_path}").run()
    assert not page.exception
    navigation_label = (
        "查看完整同题对照"
        if page_path == "04_Registry.py"
        else "打开目标与数据"
        if page_path in {"01_Experiments.py", "02_Evaluation.py", "03_Model_Comparison.py"}
        else "浏览示例任务"
    )
    assert any(item.label == navigation_label and not item.disabled for item in page.button)
    assert not any("先训练一个模型" in item.value for item in page.info)
    if page_path == "06_Chat.py":
        assert not page.get("chat_input")
        assert not any(item.label == "⚡ 加载模型" for item in page.button)
    next(item for item in page.button if item.label == navigation_label).click().run()
    assert not page.exception
    expected_title = "示例与历史" if page_path == "04_Registry.py" else session.goal
    assert any(item.value == expected_title for item in page.title)
    assert service.load(session.session_id).revision == session.revision
