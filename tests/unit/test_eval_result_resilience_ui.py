"""Malformed but parseable evaluation files do not hide usable evidence."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("02_Evaluation.py", "03_Model_Comparison.py")
OLD = "eval_detail_20261026.json"
NEW = "eval_detail_20261027.json"


def page_for(monkeypatch, tmp_path, filename, *, display=False):
    import mlflow

    from ui import runtime
    from ui.components import archive_navigation, domain_adapters

    monkeypatch.setattr(runtime, "DISPLAY_ONLY", display)
    monkeypatch.setattr(domain_adapters, "DOMAINS_DIR", tmp_path)
    monkeypatch.setattr(archive_navigation, "render_archive_source", lambda **kwargs: False)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *args: None)
    monkeypatch.setattr(mlflow, "get_experiment_by_name", lambda *args: None)
    page = AppTest.from_file(str(ROOT / "ui/views" / filename), default_timeout=30).run()
    assert not page.exception, [item.message for item in page.exception]
    return page


def write_results(tmp_path, *, latest, old=None):
    root = tmp_path / "entity_matching/data/results"
    root.mkdir(parents=True)
    (root / NEW).write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
    if old is not None:
        (root / OLD).write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    return root


GOOD = [{"model": "模型甲", "overall_accuracy": 0.8}, {"model": "模型乙", "overall_accuracy": 0.6}]
BAD = [
    None,
    {"model": "对象而非数组"},
    [None],
    ["坏记录"],
    [{}],
    [{"model": []}],
    [{"model": "坏", "accuracy_by_type": []}],
    [{"model": "坏", "accuracy_by_type": {"drug": {"不应是": "对象"}}}],
    [{"model": "坏", "accuracy_by_difficulty": {"easy": float("nan")}}],
    [{"model": "坏", "accuracy_by_difficulty": "invalid"}],
    [{"model": "坏", "per_sample": {"query": "错"}}],
    [{"model": "坏", "per_sample": [None]}],
]


@pytest.mark.parametrize("filename", PAGES)
@pytest.mark.parametrize("bad", BAD)
def test_unusable_newest_file_falls_back_to_identified_real_older_file(
    monkeypatch, tmp_path, filename, bad
):
    root = write_results(tmp_path, latest=bad, old=GOOD)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    page = page_for(monkeypatch, tmp_path, filename)
    warnings = "\n".join(item.value for item in page.warning)
    assert NEW in warnings and OLD in warnings and "较新" in warnings
    assert str(tmp_path) not in warnings
    if filename == "02_Evaluation.py":
        assert [item.label for item in page.metric] == ["模型甲", "模型乙"]
    else:
        assert any(item.value == "指标对比" for item in page.subheader)
        assert any("模型甲" in item.options for item in page.selectbox)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize("filename", PAGES)
@pytest.mark.parametrize("display", [False, True])
def test_all_unreadable_files_do_not_claim_there_were_no_records(
    monkeypatch, tmp_path, filename, display
):
    root = write_results(tmp_path, latest=None)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    page = page_for(monkeypatch, tmp_path, filename, display=display)
    messages = "\n".join(item.value for item in [*page.warning, *page.info, *page.error])
    assert NEW in messages and "全部跳过" in messages
    assert any(text in messages for text in ("无法判断", "无法确认"))
    assert "暂无评测数据" not in messages and "没有任何模型的评测结果" not in messages
    assert str(tmp_path) not in messages
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize("filename", PAGES)
def test_valid_empty_latest_result_is_not_replaced_by_an_older_result(
    monkeypatch, tmp_path, filename
):
    write_results(tmp_path, latest=[], old=GOOD)
    page = page_for(monkeypatch, tmp_path, filename)
    assert not any("损坏" in item.value for item in page.warning)
    assert not page.metric
    assert any(
        "暂无评测数据" in item.value or "没有任何模型的评测结果" in item.value
        for item in page.warning
    )


@pytest.mark.parametrize("filename", PAGES)
def test_legitimate_unknowns_and_duplicate_names_remain_browsable(monkeypatch, tmp_path, filename):
    rows = [
        {
            "model": "同名模型",
            "overall_accuracy": 0.8,
            "accuracy_by_type": None,
            "accuracy_by_difficulty": None,
            "per_sample": [{"query": "首条", "correct": False, "entity_type": None}],
        },
        {
            "model": "同名模型",
            "overall_accuracy": 0.6,
            "accuracy_by_type": {"drug": 0.6},
            "accuracy_by_difficulty": {"easy": 0.6},
            "per_sample": [{"query": "次条", "correct": False, "entity_type": "drug"}],
        },
    ]
    write_results(tmp_path, latest=rows)
    page = page_for(monkeypatch, tmp_path, filename)
    assert not any("损坏" in item.value for item in page.warning)
    if filename == "02_Evaluation.py":
        assert [item.label for item in page.metric] == ["同名模型（记录 1）", "同名模型（记录 2）"]
        assert len([item for item in page.selectbox if item.label == "按实体类型筛选"]) == 2
    else:
        assert any(item.value == "指标对比" for item in page.subheader)
        assert len([item for item in page.selectbox if item.label == "按实体类型筛选"]) == 2


def test_same_name_records_keep_independent_error_filters(monkeypatch, tmp_path):
    rows = [
        {"model": "同名", "per_sample": [{"query": "a", "correct": False, "entity_type": "drug"}]},
        {"model": "同名", "per_sample": [{"query": "b", "correct": False, "entity_type": "drug"}]},
    ]
    write_results(tmp_path, latest=rows)
    page = page_for(monkeypatch, tmp_path, "02_Evaluation.py")
    first = page.selectbox(key="err_同名（记录 1）_0_entity")
    second = page.selectbox(key="err_同名（记录 2）_1_entity")
    first.select("drug").run()
    assert not page.exception
    assert first.value == "drug" and second.value == "All"


def test_switching_comparison_model_does_not_reuse_previous_error_filter(monkeypatch, tmp_path):
    rows = [
        {
            "model": name,
            "per_sample": [{"query": name, "correct": False, "entity_type": "drug"}],
        }
        for name in ("甲", "乙", "丙")
    ]
    write_results(tmp_path, latest=rows)
    page = page_for(monkeypatch, tmp_path, "03_Model_Comparison.py")
    page.selectbox(key="err_A · 甲_0_entity").select("drug").run()
    assert not page.exception
    assert page.selectbox(key="err_A · 甲_0_entity").value == "drug"
    next(item for item in page.selectbox if item.label == "模型 A").set_value(2).run()
    assert not page.exception, [item.message for item in page.exception]
    assert page.selectbox(key="err_A · 丙_0_entity").value == "All"


def test_model_name_matching_generated_label_keeps_every_chart_series(monkeypatch, tmp_path):
    from ui.components import domain_adapters

    rows = [
        {"model": "模型一", "accuracy_by_type": {"drug": 0.1}},
        {"model": "模型一", "accuracy_by_type": {"drug": 0.2}},
        {"model": "模型一（记录 1）", "accuracy_by_type": {"drug": 0.3}},
    ]
    write_results(tmp_path, latest=rows)
    calls = []
    original = domain_adapters.make_grouped_bar

    def observe(groups, models, values, *args, **kwargs):
        calls.append((list(models), dict(values)))
        return original(groups, models, values, *args, **kwargs)

    monkeypatch.setattr(domain_adapters, "make_grouped_bar", observe)
    page = page_for(monkeypatch, tmp_path, "02_Evaluation.py")
    assert len(page.metric) == 3
    labels = [item.label for item in page.metric]
    assert labels == ["模型一（记录 1）", "模型一（记录 2）", "模型一（记录 1）（记录 3）"]
    assert len(set(labels)) == 3
    models, values = calls[-1]
    assert models == labels
    assert [values[label] for label in labels] == [[0.1], [0.2], [0.3]]
