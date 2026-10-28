"""The workspace identifies each recent run and shows its recorded evidence."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def test_recent_runs_disambiguate_same_names_and_fall_back_to_available_loss(monkeypatch):
    from ui import queries
    from ui.components import workspace

    runs = pd.DataFrame(
        [
            {
                "run_id": "12345678aa000000",
                "tags.mlflow.runName": "same",
                "status": "FINISHED",
                "start_time": datetime(2026, 10, 22, 1, 2, tzinfo=timezone.utc),
                "metrics.train_loss": 0.0,
                "metrics.loss": 0.9,
            },
            {
                "run_id": "12345678bb000000",
                "tags.mlflow.runName": "same",
                "status": "FINISHED",
                "start_time": datetime(2026, 10, 22, 2, 3, tzinfo=timezone.utc),
                "metrics.train_loss": float("nan"),
                "metrics.loss": 0.125,
            },
            {
                "run_id": None,
                "tags.mlflow.runName": None,
                "status": "FAILED",
                "start_time": None,
                "metrics.train_loss": float("inf"),
                "metrics.loss": float("nan"),
            },
            # Prefix collisions outside the three visible rows still matter on the experiments page.
            {"run_id": "12345678ac000000", "tags.mlflow.runName": "other"},
        ]
    )
    monkeypatch.setattr(queries, "fetch_runs", lambda *_: runs)
    monkeypatch.setattr(workspace, "load_tasks", lambda: ([], 0))
    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=30).run()
    assert not page.exception
    table = next(item.value for item in page.dataframe if "运行 ID" in item.value.columns)
    assert table["记录"].tolist() == ["same", "same", "未命名运行"]
    assert table["运行 ID"].tolist() == ["12345678aa", "12345678b", "未记录"]
    assert table["训练损失"].tolist() == ["0", "0.125", "—"]
    assert table["开始时间"].tolist() == [
        "2026-10-22 01:02:00（UTC）",
        "2026-10-22 02:03:00（UTC）",
        "未记录",
    ]
    assert any(item.label == "查看全部实验记录" for item in page.get("page_link"))


def test_start_time_does_not_invent_a_timezone():
    from ui.components.workspace import _start_time

    assert _start_time(datetime(2026, 10, 22, 1, 2)) == "2026-10-22 01:02:00（时区未记录）"
    assert _start_time("not a timestamp") == "未记录"


def test_short_ids_remain_distinct_when_one_id_is_prefix_of_another():
    from ui.components.workspace import _short_run_ids

    ids = ["abcdefgh", "abcdefghx12", "abcde"]
    assert _short_run_ids(pd.DataFrame({"run_id": ids})) == {
        "abcdefgh": "abcdefgh",
        "abcdefghx12": "abcdefghx",
        "abcde": "abcde",
    }
