"""The first action starts a business task, with existing experiment records still available."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def boot(monkeypatch, tmp_path, runs=None):
    from ui import config, queries, runtime

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        queries, "fetch_runs", lambda *a, **k: pd.DataFrame() if runs is None else runs
    )
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", False)
    return AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()


def test_empty_workspace_leads_with_business_goal(monkeypatch, tmp_path):
    page = boot(monkeypatch, tmp_path)
    assert not page.exception
    assert any("业务模型工作台" in t.value for t in page.title)
    assert any(b.label == "创建业务任务" for b in page.button)
    # Public historical scores are visible even when the user's local store is empty.
    assert any("公开历史验证" in item.value for item in page.caption)
    assert not any(item.label in {"运行总数", "运行中"} for item in page.metric)
    assert all(fmt in " ".join(m.value for m in page.markdown) for fmt in ("CSV", "Excel", "JSONL"))


def test_create_task_opens_goal_and_data_without_an_agent_requirement(monkeypatch, tmp_path):
    page = boot(monkeypatch, tmp_path)
    next(b for b in page.button if b.label == "创建业务任务").click().run()
    assert not page.exception
    assert any(t.value == "先说业务目标，再看数据" for t in page.title)
    assert any(x.label == "希望模型完成什么业务工作？" for x in page.text_area)
    assert any(b.label == "读取数据并开始" for b in page.button)


def test_recorded_evidence_is_distinct_from_business_acceptance(monkeypatch, tmp_path):
    runs = pd.DataFrame(
        [
            {
                "run_id": "recorded",
                "status": "FINISHED",
                "tags.mlflow.runName": "local-validation",
                "tags.evidence.source": "public-evidence.json",
                "metrics.loss": 0.12,
            }
        ]
    )
    page = boot(monkeypatch, tmp_path, runs)
    assert not page.exception
    table = next(item.value for item in page.dataframe if "来源" in item.value.columns)
    assert table.iloc[0]["来源"] == "公开历史验证"
    assert table.iloc[0]["状态"] == "已完成"
    assert not any("业务达标" in str(x) for x in table.iloc[0].values)
