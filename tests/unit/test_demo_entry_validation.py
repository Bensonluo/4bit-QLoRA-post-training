"""Every demo creation entrypoint validates the same unchanged sample bytes."""

import hashlib
import json

import pytest
from streamlit.testing.v1 import AppTest

from scripts import data_intake
from src.workbench.demo_task import (
    DEMO_SAMPLE_BUTTON,
    DEMO_VARIANTS,
    demo_create_payload,
    demo_full,
    demo_sample,
    demo_variant_sample,
)
from src.workbench.intake_service import IntakeService
from tests.unit.test_demo_cli import _run
from tests.unit.test_example_resilience_ui import example_path, example_root
from tests.unit.test_intake_display_readonly_ui import ROOT, clean, configure

VARIANTS = [
    (None, "售后工单分类", DEMO_SAMPLE_BUTTON, "客户描述,类别\n杯子破损,\n"),
    (
        "generation",
        "客服答复生成",
        DEMO_VARIANTS["generation"]["button"],
        "客户问题,标准答复\n退货怎么处理,\n",
    ),
    (
        "extraction",
        "工单编号抽取",
        DEMO_VARIANTS["extraction"]["button"],
        "客户描述,工单号\n查工单,\n",
    ),
]


@pytest.mark.parametrize("variant,title,label,invalid", VARIANTS)
def test_cli_rejects_bad_sample_but_other_variant_still_creates_original_task(
    monkeypatch, capsys, tmp_path, variant, title, label, invalid
):
    root = example_root(tmp_path)
    source = example_path(root, variant)
    raw = invalid.encode()
    source.write_bytes(raw)
    monkeypatch.setattr(data_intake, "PROJECT_ROOT", root)
    store = tmp_path / "intake"
    command = ["create", "--demo"] + (["--demo-variant", variant] if variant else [])
    assert _run(monkeypatch, capsys, store, *command) == 2
    captured = capsys.readouterr()
    assert "示例输入或期望答案为空" in captured.err
    assert IntakeService(store).list_sessions() == []
    assert source.read_bytes() == raw
    good_variant = "generation" if variant != "generation" else "extraction"
    good_raw = example_path(root, good_variant).read_bytes()
    assert _run(monkeypatch, capsys, store, "create", "--demo", "--demo-variant", good_variant) == 0
    identity = json.loads(capsys.readouterr().out)["session_id"]
    assert IntakeService(store).load(identity).source.digest == hashlib.sha256(good_raw).hexdigest()


@pytest.mark.parametrize("variant,title,label,invalid", VARIANTS)
@pytest.mark.parametrize("missing", [False, True], ids=["invalid", "missing"])
def test_intake_entry_isolates_missing_or_invalid_sample_and_keeps_other_actions(
    monkeypatch, tmp_path, variant, title, label, invalid, missing
):
    root = example_root(tmp_path)
    source = example_path(root, variant)
    if missing:
        source.unlink()
    else:
        source.write_text(invalid, encoding="utf-8")
    configure(monkeypatch, root, display=False)
    page = AppTest.from_file(str(ROOT / "ui/views/07_Data_Intake.py"), default_timeout=45).run()
    clean(page)
    assert not any(item.label == label for item in page.button)
    notice = "\n".join(item.value for item in [*page.warning, *page.caption])
    assert title in notice and ("暂不可用" if missing else "不可读或不完整") in notice
    assert str(root) not in notice
    good_variant = "generation" if variant != "generation" else "extraction"
    good_raw = example_path(root, good_variant).read_bytes()
    page.button(key=f"demo_{good_variant}").click().run()
    clean(page)
    sessions = IntakeService(root / "outputs/workbench/intake").list_sessions()
    assert len(sessions) == 1
    assert sessions[0].source.digest == hashlib.sha256(good_raw).hexdigest()
    assert sessions[0].analysis is None


def test_sample_reader_and_companion_api_keep_original_raw_behavior(tmp_path):
    root = example_root(tmp_path)
    raw = b"client,answer\nbroken,\n"
    example_path(root, None).write_bytes(raw)
    assert demo_sample(root)[1] == raw
    assert demo_variant_sample(root, "generation") is not None
    assert demo_full(root) is None
    with pytest.raises(ValueError):
        demo_create_payload(root)


def test_real_user_csv_remains_usable_with_unfilled_answers(monkeypatch, capsys, tmp_path):
    raw = "客户描述,类别\n杯子破损,\n".encode()
    source = tmp_path / "own-data.csv"
    source.write_bytes(raw)
    store = tmp_path / "intake"
    assert (
        _run(monkeypatch, capsys, store, "create", "--input", str(source), "--goal", "用户目标")
        == 0
    )
    identity = json.loads(capsys.readouterr().out)["session_id"]
    session = IntakeService(store).load(identity)
    assert session.source.digest == hashlib.sha256(raw).hexdigest()
    assert session.source.rows[0].values["类别"] == ""


@pytest.mark.parametrize("variant", ["generation", "extraction"])
def test_cli_missing_variant_names_real_file(monkeypatch, capsys, tmp_path, variant):
    root = example_root(tmp_path)
    source = example_path(root, variant)
    source.unlink()
    monkeypatch.setattr(data_intake, "PROJECT_ROOT", root)
    assert (
        _run(
            monkeypatch, capsys, tmp_path / "intake", "create", "--demo", "--demo-variant", variant
        )
        == 2
    )
    error = capsys.readouterr().err
    assert source.name in error
    assert str(root) not in error
