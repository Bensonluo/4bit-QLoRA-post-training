"""对比核验 CLI:三道语义安全关卡中曾唯一缺 CLI 的那道,agent 路径补齐。

stdout 纯 JSON、stderr 人读;判定与连胜口径(二连对)如实亮出;confirm 的
stderr 如实报告当前配对状态——软门禁语义,不阻断但四种状态都不静默。
"""

import json
import subprocess
import sys
from pathlib import Path

from src.workbench.intake_service import IntakeService
from tests.unit.test_data_intake import CSV, analysis

CLI = Path(__file__).resolve().parents[2] / "scripts/data_intake.py"


def _store(tmp_path):
    service = IntakeService(tmp_path / "intake")
    session = service.create("根据客户首次描述预测类别", "工单.csv", CSV)
    session = service.apply_analysis(session, analysis())
    return service, session


def invoke(service, *args):
    return subprocess.run(
        [sys.executable, str(CLI), "--store", str(service.root), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def _targets(service, session_id):
    return {row.row_id: row.target for row in service.load(session_id).preview.rows}


def test_contrast_cli_mismatch_is_reported_and_confirm_hint_is_honest(tmp_path):
    """配错留档不阻断确认(软门禁),但判定行与 confirm 提示都不静默。"""
    service, session = _store(tmp_path)
    started = invoke(service, "contrast-check", session.session_id, "--revision", session.revision)
    assert started.returncode == 0, started.stderr
    pending = json.loads(started.stdout)
    assert pending["check_id"] and len(pending["items"]) == 2
    assert len(set(pending["options"])) == 2
    assert "contrast-check-submit" in pending["submit_hint"]
    # stderr 人读面:题目原文、打乱候选与作答说明,不含正确答案标记。
    assert "请把每个答案配到正确的输入上" in started.stderr
    assert started.stderr.count("[r") == 2
    assert "已打乱" in started.stderr

    targets = _targets(service, session.session_id)
    first, second = (item["row_id"] for item in pending["items"])
    swapped = {first: targets[second], second: targets[first]}  # 全错调换 = 0/2
    submitted = invoke(
        service,
        "contrast-check-submit",
        session.session_id,
        "--check-id",
        pending["check_id"],
        "--answer",
        f"{first}={swapped[first]}",
        "--answer",
        f"{second}={swapped[second]}",
    )
    assert submitted.returncode == 0, submitted.stderr
    result = json.loads(submitted.stdout)
    assert result["verdict"] == "mismatch"
    assert result["matched"] == 0 and result["total"] == 2
    assert "判定：mismatch（0/2 配对正确）" in submitted.stderr
    assert "盲点头" in submitted.stderr, "配错的 verdict_note 必须点名盲点头风险"

    # confirm 不因配错被阻断(软门禁语义),但 stderr 如实报告最近配对错误。
    confirmed = invoke(service, "confirm", session.session_id, "--revision", session.revision)
    assert confirmed.returncode == 0, confirmed.stderr
    assert "对比核验：最近一次配对错误" in confirmed.stderr
    assert "请重新查看预览" in confirmed.stderr


def test_contrast_cli_two_verified_rounds_reach_second_round_standard(tmp_path):
    """连胜口径如实:一轮对还差一轮、二连对达标,confirm 提示同步变化。"""
    service, session = _store(tmp_path)
    targets = _targets(service, session.session_id)
    outputs = []
    for _ in range(2):
        started = invoke(
            service, "contrast-check", session.session_id, "--revision", session.revision
        )
        assert started.returncode == 0, started.stderr
        pending = json.loads(started.stdout)
        answers = [f"{item['row_id']}={targets[item['row_id']]}" for item in pending["items"]]
        args = ["contrast-check-submit", session.session_id, "--check-id", pending["check_id"]]
        for answer in answers:
            args += ["--answer", answer]
        outputs.append(invoke(service, *args))
    first, second = outputs
    assert first.returncode == 0 and second.returncode == 0, second.stderr
    assert json.loads(first.stdout)["verdict"] == "verified"
    assert json.loads(second.stdout)["verdict"] == "verified"
    assert "已连续 1 轮" in first.stderr
    assert "还需再连续配对正确一轮（二连对）才算真正看清" in first.stderr
    assert "二连对达标" in second.stderr

    confirmed = invoke(service, "confirm", session.session_id, "--revision", session.revision)
    assert confirmed.returncode == 0, confirmed.stderr
    assert "对比核验：已连续 2 轮配对正确（二连对达标）" in confirmed.stderr


def test_contrast_cli_confirm_without_check_suggests_contrast_check_first(tmp_path):
    """未核验直接确认不阻断,但提示先做配对核验——建议而非门禁。"""
    service, session = _store(tmp_path)
    confirmed = invoke(service, "confirm", session.session_id, "--revision", session.revision)
    assert confirmed.returncode == 0, confirmed.stderr
    assert "尚未做过配对核验" in confirmed.stderr
    assert "contrast-check" in confirmed.stderr


def test_contrast_cli_rejects_stale_revision_and_resubmission(tmp_path):
    """过期 revision 拒绝抽题;已提交结论的核验不能重复提交;方案变化后失效。"""
    service, session = _store(tmp_path)
    stale = invoke(
        service, "contrast-check", session.session_id, "--revision", session.revision + 1
    )
    assert stale.returncode == 2
    assert "任务已更新" in stale.stderr

    started = invoke(service, "contrast-check", session.session_id, "--revision", session.revision)
    pending = json.loads(started.stdout)
    targets = _targets(service, session.session_id)
    answers = [f"{item['row_id']}={targets[item['row_id']]}" for item in pending["items"]]
    args = ["contrast-check-submit", session.session_id, "--check-id", pending["check_id"]]
    for answer in answers:
        args += ["--answer", answer]
    submitted = invoke(service, *args)
    assert submitted.returncode == 0, submitted.stderr
    resubmitted = invoke(service, *args)
    assert resubmitted.returncode == 2
    assert "已提交过结论" in resubmitted.stderr

    # 预览/方案变化(binding 变化)后,再抽的一轮提交时如实报失效。
    started = invoke(service, "contrast-check", session.session_id, "--revision", session.revision)
    pending = json.loads(started.stdout)
    changed = analysis()
    changed.recipe.instruction = "调整后的判断指令。"
    service.apply_analysis(session, changed)
    args = ["contrast-check-submit", session.session_id, "--check-id", pending["check_id"]]
    for item in pending["items"]:
        args += ["--answer", f"{item['row_id']}={targets[item['row_id']]}"]
    expired = invoke(service, *args)
    assert expired.returncode == 2
    assert "已变化" in expired.stderr and "失效" in expired.stderr
