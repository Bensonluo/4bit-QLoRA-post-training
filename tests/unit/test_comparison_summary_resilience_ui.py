"""A damaged optional summary must not prevent comparing usable evaluations."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("state", ["invalid", "unreadable", "missing", "valid"])
def test_summary_state_preserves_model_comparison(monkeypatch, tmp_path, state):
    from ui import config, queries
    from ui.components import domain_adapters

    records = [
        {"model": "基础模型", "overall_accuracy": 0.5},
        {"model": "微调模型", "overall_accuracy": 0.75},
    ]
    results = tmp_path / "entity_matching/data/results"
    results.mkdir(parents=True)
    summary = results / "executive_summary_20261028.md"
    if state == "invalid":
        summary.write_bytes(b"\xff\xfeinvalid")
    elif state in {"unreadable", "valid"}:
        summary.write_text("健康历史摘要", encoding="utf-8")
    if state == "unreadable":
        read_text = Path.read_text

        def deny_summary(path, *args, **kwargs):
            if path == summary:
                raise PermissionError("no access")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", deny_summary)
    monkeypatch.setattr(config, "DOMAINS_DIR", tmp_path)
    monkeypatch.setattr(domain_adapters, "load_eval_data", lambda *_: records)
    monkeypatch.setattr(queries, "fetch_runs", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(queries, "fetch_experiments", lambda *a, **k: [])

    page = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=45).run()
    page.switch_page("views/03_Model_Comparison.py").run()

    assert not page.exception, [item.message for item in page.exception]
    if state in {"invalid", "unreadable"}:
        assert any("领域历史摘要暂不可读" in item.value for item in page.warning)
        assert not any("未找到执行摘要文件" in item.value for item in page.info)
    elif state == "missing":
        assert any("未找到执行摘要文件" in item.value for item in page.info)
        assert not any("领域历史摘要暂不可读" in item.value for item in page.warning)
    else:
        assert any(item.value == "健康历史摘要" for item in page.markdown)
    table = next(item.value for item in page.dataframe if "差值（B − A）" in item.value.columns)
    assert table.iloc[0]["差值（B − A）"] == "+25.0 个百分点"
