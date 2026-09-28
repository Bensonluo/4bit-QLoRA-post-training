"""任务规约投影：collect 只读汇编 / summarize 人话 / CLI 双流 / 页面规约卡。

设计文档 ADR-1（只读投影）的落地测试，钉住三条诚实边界：
- 草稿评分规则不进入投影，只汇编已确认的事实；
- 未冻结验收如实显示「未冻结」，不假装存在标准；
- 收尾行固定声明「只汇编已确认事实，不代表模型效果达标」。
"""

# ruff: noqa: F811

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("datasets")

from src.workbench.task_spec_projection import collect_task_spec, summarize_task_spec
from tests.unit.test_business_evaluation import task  # noqa: F401


def _roots(tmp_path: Path) -> dict[str, Path]:
    return {
        name: tmp_path / name
        for name in ("intake", "scoring", "acceptance", "evaluations", "iterations", "training")
    }


@pytest.fixture()
def patched_lists(monkeypatch):
    """空列表桩：签名与生产一致（session_id 关键字），只服务本测试的汇编断言。"""
    from src.workbench.acceptance import AcceptanceService
    from src.workbench.iterations import IterationService

    captured = {}

    def acceptances(self, session_id=None):
        captured["acceptance_session"] = session_id
        return []

    def iterations(self, session_id=None):
        captured["iteration_session"] = session_id
        return []

    monkeypatch.setattr(AcceptanceService, "list_acceptances", acceptances)
    monkeypatch.setattr(IterationService, "list_iterations", iterations)
    return captured


def test_collect_assembles_five_elements_and_excludes_drafts(task, tmp_path, patched_lists):
    """已确认评分进入投影、草稿只计数；无验收如实显示未冻结。"""
    from src.workbench.business_scoring import ScoringService
    from tests.unit.test_business_scoring import recipe_for

    _, session = task
    roots = _roots(tmp_path)
    scoring = ScoringService(roots["scoring"])
    context = scoring.context(session, "完整答案必须与标准答案精确相同")
    recipe = recipe_for(context["development_cases"][0])
    confirmed = scoring.draft(session, recipe)
    scoring.confirm(confirmed["scoring_id"], session)
    scoring.draft(session, recipe)  # 第二份保持草稿，不得进入投影。

    spec = collect_task_spec(
        session.session_id,
        roots["intake"],
        roots["scoring"],
        roots["acceptance"],
        roots["evaluations"],
        roots["iterations"],
        roots["training"],
    )
    assert spec["session_id"] == session.session_id
    assert spec["revision"] == session.revision
    # 业务目标来自 session + analysis.task。
    assert spec["goal"]["goal"] == "根据客户首次描述判断问题类型"
    assert spec["goal"]["usage_input"]
    assert spec["goal"]["success_criteria"]
    # 答案语义来自已确认 recipe。
    assert spec["answer_semantics"]["instruction"]
    assert spec["answer_semantics"]["targets"][0]["label"] == "类别"
    assert spec["answer_semantics"]["supervision_source"]
    # 评分口径：1 份已确认 + 1 份草稿只计数。
    assert len(spec["scoring"]["confirmed"]) == 1
    assert spec["scoring"]["confirmed"][0]["business_standard"] == (
        "完整答案必须与标准答案精确相同"
    )
    assert spec["scoring"]["confirmed"][0]["example_count"] == len(recipe.examples)
    assert spec["scoring"]["draft_count"] == 1
    # 验收未冻结、非时间任务、无改进轮——如实显示，不假装。
    assert spec["acceptance"] == {"state": "未冻结", "records": []}
    assert spec["temporal_split"] is None
    assert spec["latest_iteration"] is None
    # 投影按 session 过滤读取。
    assert patched_lists["acceptance_session"] == session.session_id
    assert patched_lists["iteration_session"] == session.session_id


def test_collect_shows_frozen_acceptance_latest_iteration_and_stale_verification(
    task, tmp_path, monkeypatch
):
    from src.workbench.acceptance import AcceptanceService
    from src.workbench.iterations import IterationService

    _, session = task
    roots = _roots(tmp_path)
    acceptance_record = {
        "acceptance_id": "ac-00000000000000000000000000000000",
        "status": "completed",
        "protocol": {"scorer": "classification_exact"},
        "criteria": {
            "metric": "exact_match",
            "minimum_score": 0.8,
            "minimum_cases": 10,
            "business_standard": "按业务标准逐字比对",
        },
        "result": {"decision": "passed"},
    }
    iteration_record = {
        "iteration_id": "it-00000000000000000000000000000000",
        "hypothesis": "增加困难样例提高长尾类别召回",
        "expected_outcome": "开发集对照提升",
        "status": "proposed",
    }
    monkeypatch.setattr(
        AcceptanceService,
        "list_acceptances",
        lambda self, session_id=None: [acceptance_record],
    )
    monkeypatch.setattr(
        IterationService,
        "list_iterations",
        lambda self, session_id=None: [iteration_record],
    )
    spec = collect_task_spec(
        session.session_id,
        roots["intake"],
        roots["scoring"],
        roots["acceptance"],
        roots["evaluations"],
        roots["iterations"],
        roots["training"],
    )
    assert spec["acceptance"]["state"] == "已冻结"
    assert spec["acceptance"]["records"][0]["criteria"]["metric"] == "exact_match"
    assert spec["acceptance"]["records"][0]["result_decision"] == "passed"
    assert spec["latest_iteration"]["hypothesis"] == "增加困难样例提高长尾类别召回"
    assert spec["latest_iteration"]["status"] == "proposed"
    # 盲标核验缺失是如实状态，不是错误。
    assert spec["answer_semantics"]["label_verification"] is None


def test_summarize_pins_honest_boundaries_with_confirmed_scoring(task, tmp_path, patched_lists):
    from src.workbench.business_scoring import ScoringService
    from tests.unit.test_business_scoring import recipe_for

    _, session = task
    roots = _roots(tmp_path)
    scoring = ScoringService(roots["scoring"])
    context = scoring.context(session, "完整答案必须与标准答案精确相同")
    record = scoring.draft(session, recipe_for(context["development_cases"][0]))
    scoring.confirm(record["scoring_id"], session)
    spec = collect_task_spec(
        session.session_id,
        roots["intake"],
        roots["scoring"],
        roots["acceptance"],
        roots["evaluations"],
        roots["iterations"],
        roots["training"],
    )
    lines = summarize_task_spec(spec)
    assert lines[0].startswith(f"任务 {session.session_id[:12]}（revision {session.revision}）")
    assert "由既有确认记录只读汇编，不新增状态。" in lines[0]
    assert f"业务目标：{spec['goal']['goal']}" in lines
    assert any(line.startswith("答案语义：text 输出「类别」") for line in lines)
    assert "评分口径：已确认自定义规则（完整答案必须与标准答案精确相同，通过阈值 1.0）。" in lines
    assert "验收标准：未冻结——当前没有可对照的独立验收条款。" in lines
    assert "时间约束：无（非时间预测任务）。" in lines
    assert "最新改进轮：尚无。" in lines
    assert lines[-1] == "以上是训练启动前的对齐视图：只汇编已确认事实，不代表模型效果达标。"


def test_summarize_reports_stale_verification_and_frozen_acceptance(task, tmp_path, monkeypatch):
    from src.workbench.acceptance import AcceptanceService
    from src.workbench.intake_service import IntakeService
    from src.workbench.iterations import IterationService

    _, session = task
    roots = _roots(tmp_path)

    stale = session.model_copy(deep=True)
    stale.label_verification = {
        "stale": True,
        "previous_verdict": "verified",
        "previous_created_at": "t",
    }
    monkeypatch.setattr(
        IntakeService,
        "load",
        lambda self, session_id: stale,
    )
    monkeypatch.setattr(
        AcceptanceService,
        "list_acceptances",
        lambda self, session_id=None: [
            {
                "acceptance_id": "ac-x",
                "status": "completed",
                "protocol": {"scorer": "classification_exact"},
                "criteria": {
                    "metric": "exact_match",
                    "minimum_score": 0.8,
                    "minimum_cases": 10,
                    "business_standard": "标准",
                },
                "result": {"decision": "passed"},
            }
        ],
    )
    monkeypatch.setattr(
        IterationService,
        "list_iterations",
        lambda self, session_id=None: [
            {
                "iteration_id": "it-x",
                "hypothesis": "假设",
                "expected_outcome": "预期",
                "status": "proposed",
            }
        ],
    )
    spec = collect_task_spec(
        session.session_id,
        roots["intake"],
        roots["scoring"],
        roots["acceptance"],
        roots["evaluations"],
        roots["iterations"],
        roots["training"],
    )
    lines = summarize_task_spec(spec)
    assert "盲标核验：已失效（此前结论 verified），数据或方案更新后需重新核验。" in lines
    assert "验收标准：已冻结（exact_match 最低 0.8、最少 10 例）；最近一次结果 passed。" in lines
    assert "最新改进轮：假设（proposed）。" in lines
    assert "评分口径：默认严格匹配（暂无已确认的自定义规则）。" in lines


def test_cli_task_spec_show_streams_json_stdout_and_human_stderr(
    task, tmp_path, monkeypatch, capsys, patched_lists
):
    """双流契约：stdout 纯 JSON（脚本可解析），stderr 人话与 summarize_task_spec 同源。"""
    from scripts import data_intake

    _, session = task
    roots = _roots(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_intake.py",
            "--store",
            str(roots["intake"]),
            "--scoring-root",
            str(roots["scoring"]),
            "--acceptance-root",
            str(roots["acceptance"]),
            "--evaluation-root",
            str(roots["evaluations"]),
            "--iteration-root",
            str(roots["iterations"]),
            "--training-root",
            str(roots["training"]),
            "task-spec-show",
            session.session_id,
        ],
    )
    assert data_intake.main() == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["session_id"] == session.session_id
    assert payload["acceptance"]["state"] == "未冻结"
    assert "的任务规约：由既有确认记录只读汇编" in err
    assert "不代表模型效果达标" in err


def test_ui_task_spec_card_matches_cli_wording(tmp_path, monkeypatch):
    """页面规约卡在训练启动前渲染，与 CLI 同一条人话（单一来源词汇）。"""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import ui.config
    from src.workbench.intake_service import IntakeService
    from tests.unit.test_full_data import FULL, approved

    PAGE = Path(__file__).resolve().parents[2] / "ui/pages/07_Data_Intake.py"
    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)
    monkeypatch.setattr(ui.config, "PROJECT_ROOT", tmp_path)

    service = IntakeService(tmp_path / "outputs/workbench/intake")
    session = approved(service)
    session = service.validate_full_data(session.session_id, session.revision, "full.csv", FULL)
    session = service.confirm_full_data(session.session_id, session.revision)
    session = service.materialize_dataset(session.session_id, session.revision)

    page = AppTest.from_file(str(PAGE), default_timeout=20)
    page.run()
    assert not page.exception
    page.selectbox(key="intake_select").select(session.session_id).run()
    assert not page.exception
    texts = [block.value for block in page.markdown]
    labels = [expander.label for expander in page.expander]
    assert any("任务规约投影（训练启动前对齐" in label for label in labels)
    assert any("的任务规约：由既有确认记录只读汇编，不新增状态。" in text for text in texts)
    assert any("验收标准：未冻结——当前没有可对照的独立验收条款。" in text for text in texts)
    assert any(
        "以上是训练启动前的对齐视图：只汇编已确认事实，不代表模型效果达标。" in text
        for text in texts
    )


def test_spec_anchor_lines_match_the_element_slice_of_summarize(task, tmp_path, patched_lists):
    """spec_anchor_lines 单一来源:四要素行与 summarize 输出中段逐行相等;空输入为空。"""
    from src.workbench.business_scoring import ScoringService
    from src.workbench.task_spec_projection import spec_anchor_lines
    from tests.unit.test_business_scoring import recipe_for

    _, session = task
    roots = _roots(tmp_path)
    scoring = ScoringService(roots["scoring"])
    context = scoring.context(session, "完整答案必须与标准答案精确相同")
    record = scoring.draft(session, recipe_for(context["development_cases"][0]))
    scoring.confirm(record["scoring_id"], session)
    spec = collect_task_spec(
        session.session_id,
        roots["intake"],
        roots["scoring"],
        roots["acceptance"],
        roots["evaluations"],
        roots["iterations"],
        roots["training"],
    )
    # summarize 输出 = 头行 + 四要素行 + 验收行 + 改进轮行 + 收尾行;
    # 四要素行与单一来源逐行相等(spec_anchor_lines 与 summarize 共用同一段渲染)。
    assert spec_anchor_lines(spec) == summarize_task_spec(spec)[1:-3]
    # 空输入(旧验收记录缺 task_spec 键时读到的空 dict / None):如实返回空清单。
    assert spec_anchor_lines({}) == []
    assert spec_anchor_lines(None) == []
