"""Agent plans use real local tokenizer facts and prepare only after confirmation."""

import json
from unittest.mock import patch

import pytest

from src.workbench.training_plans import TrainingPlanService
from tests.unit.test_workbench_training_runs import environment  # noqa: F401


@pytest.fixture
def plans(environment, tmp_path):  # noqa: F811
    _, session, runs, model = environment
    service = TrainingPlanService(tmp_path / "plans", runs.root)
    service.training = runs
    return service, session, model


def _proposal(model, **changes):
    return {
        "status": "ready",
        "model_path": str(model),
        "max_length": 32,
        "training_options": {"num_epochs": 1, "batch_size": 1},
        "lora_options": {"r": 2, "lora_alpha": 4},
        "model_options": {"quantization_bits": None, "torch_dtype": "float32"},
        "rationale": ["按已确认分类标签和实际长度选择本地 SFT/LoRA。"],
        "limitations": ["预检没有测量训练峰值内存或业务效果。"],
        "business_questions": [],
        **changes,
    }


def _save(plans, **changes):
    service, session, model = plans
    context = service.context(session, [str(model)])
    # 生产形状的轨迹（src/agent/training.py）：逐工具调用记录，含失败项。
    trace = [
        {"tool": "training_context", "ok": True},
        {"tool": "probe_dataset", "ok": True},
    ]
    return service.save(session, _proposal(model, **changes), context, trace)


def test_actual_context_and_probe_do_not_prepare_a_run(plans, tmp_path):
    service, session, model = plans
    facts = service.context(session, [str(model), str(tmp_path / "missing")])
    assert facts["business"]["goal"] == session.goal
    assert facts["statistics"]["total_rows"] == 9
    assert facts["platform"]["device"] in ("cpu", "cuda", "mps")
    assert facts["models"][0]["config"]["model_type"] == "gpt2"
    assert facts["models"][0]["tokenizer"]["pad_token_id"] == 1
    assert facts["models"][1]["status"] == "unsupported"
    probe = service.probe(session, str(model), 32)
    assert probe["status"] == "passed"
    assert probe["preflight"]["supervision_strategy"] == "full_prompt_padding_masked"
    assert not list(service.training.root.glob("wb-*"))
    lost = service.probe(session, str(model), 2)
    assert lost["status"] == "blocked"
    assert any(issue.get("code") == "answer_lost" for issue in lost["issues"])
    assert service.probe(session, str(model), 65)["status"] == "blocked"


def test_save_reopen_confirm_prepare_and_repeat_reuses_one_actual_run(plans):
    service, session, model = plans
    plan = _save(plans)
    assert not list(service.training.root.glob("wb-*"))
    reopened = TrainingPlanService(service.root, service.training.root)
    assert reopened.get(plan["plan_id"])["model_identity"] == plan["model_identity"]
    assert len(reopened.list_plans(session.session_id)) == 1
    assert not reopened.list_plans("different-session")
    result = service.prepare(plan["plan_id"], session)
    assert result["training_run"]["status"] == "prepared"
    # 方案确认时把方案记录的 trace 固化进运行记录（run 目录自包含证据包）。
    assert result["training_run"]["plan_trace"] == plan["trace"]
    run_on_disk = json.loads(
        (service.training.root / result["run_id"] / "run.json").read_text(encoding="utf-8")
    )
    assert run_on_disk["plan_trace"] == plan["trace"]
    config = result["training_run"]["config"]
    assert config["data"]["train_file"] == session.dataset.paths["train"]
    assert config["data"]["validation_split"] == 0
    assert session.dataset.paths["test"] not in config["data"].values()
    assert service.prepare(plan["plan_id"], session)["run_id"] == result["run_id"]
    assert len(list(service.training.root.glob("wb-*"))) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"business_questions": ["标签来源尚未确定"]},
        {"max_length": 2},
        {"max_length": 100},
        {"max_length": True},
        {"training_options": {"output_dir": "/tmp/override"}},
        {"training_options": {"learning_rate": -1}},
        {"lora_options": {"r": True}},
        {"lora_options": {"task_type": "SEQ_CLS"}},
        {"model_options": {"trust_remote_code": True}},
        {"model_options": {"use_flash_attention": True}},
    ],
)
def test_invalid_or_unresolved_ready_plan_is_rejected(plans, changes):
    with pytest.raises((ValueError, TypeError)):
        _save(plans, **changes)
    assert not list(plans[0].training.root.glob("wb-*"))


def test_preparation_rejects_revised_dataset_and_same_size_model_replacement(plans):
    service, session, model = plans
    plan = _save(plans)
    revised = session.model_copy(deep=True)
    revised.revision += 1
    with pytest.raises(ValueError, match="已更新"):
        service.prepare(plan["plan_id"], revised)
    weights = next(model.glob("*.safetensors"))
    original = weights.read_bytes()
    weights.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="模型内容已变化"):
        service.prepare(plan["plan_id"], session)
    assert not list(service.training.root.glob("wb-*"))


def test_stale_context_changed_model_and_uninspected_candidate_rejected(plans):
    service, session, model = plans
    context = service.context(session, [str(model)])
    other = session.model_copy(deep=True)
    other.goal = "a changed business goal"
    with pytest.raises(ValueError, match="上下文已过期"):
        service.save(other, _proposal(model), context, [])
    with pytest.raises(ValueError, match="候选"):
        service.save(session, _proposal(model / "another"), context, [])
    config = json.loads((model / "config.json").read_text())
    config["new_fact"] = "changed"
    (model / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="模型内容已变化"):
        service.save(session, _proposal(model), context, [])


@pytest.mark.parametrize("status", ["needs_data", "unsupported"])
def test_non_executable_plan_records_gap_without_run_or_required_model(plans, status):
    service, session, _ = plans
    incomplete = session.model_copy(deep=True)
    incomplete.dataset = None
    context = service.context(incomplete, [])
    proposal = _proposal(
        None, status=status, model_path=None, max_length=None, business_questions=["请补齐数据"]
    )
    plan = service.save(incomplete, proposal, context, [])
    assert plan["status"] == status
    with pytest.raises(ValueError, match="ready"):
        service.prepare(plan["plan_id"], incomplete)
    assert not list(service.training.root.glob("wb-*"))


def test_hardware_incompatible_quantization_is_not_silently_disabled(plans):
    from src.utils.platform_utils import PlatformInfo

    cpu = PlatformInfo("cpu", False, False, False, False, False, 0.0, "CPU", 0)
    with (
        patch("src.utils.platform_utils.detect_platform", return_value=cpu),
        pytest.raises(ValueError, match="量化"),
    ):
        _save(plans, model_options={"quantization_bits": 4, "torch_dtype": "float32"})


@pytest.mark.parametrize("kind", ["mlx", "encoder", "missing_tokenizer"])
def test_incompatible_candidate_is_rejected_before_hashing_weights(plans, kind):
    service, session, model = plans
    path = model / "config.json"
    config = json.loads(path.read_text())
    if kind == "mlx":
        config["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    elif kind == "encoder":
        config["model_type"] = "bert"
        config["architectures"] = ["BertModel"]
    else:
        (model / "tokenizer.json").unlink()
    path.write_text(json.dumps(config))
    with patch(
        "src.workbench.training_plans.model_identity",
        side_effect=AssertionError("must not hash weights"),
    ):
        candidate = service.context(session, [str(model)])["models"][0]
    assert candidate["status"] == "unsupported"
    assert candidate["model_identity"] is None
    if kind == "mlx":
        assert "MLX" in candidate["issues"][0]


def test_context_explains_current_data_todo_without_original_rows(plans):
    service, session, _ = plans
    revised = session.model_copy(deep=True)
    revised.dataset = None
    context = service.context(revised, [])
    assert context["dataset"] is None
    assert context["data_readiness"]["next_action"] == "awaiting_dataset_split"
    assert "分区" in context["data_readiness"]["required_actions"][0]
    assert "materialize" in context["data_readiness"]["required_actions"][1]
    revised.full_data.status = "stale"
    context = service.context(revised, [])
    assert "现有全量资料重新校验" in context["data_readiness"]["required_actions"][0]
    assert "full-validate" in context["data_readiness"]["required_actions"][1]
    revised.full_data = None
    context = service.context(revised, [])
    assert context["data_readiness"]["next_action"] in {
        "awaiting_full_data",
        "awaiting_full_validation",
    }
    assert context["data_readiness"]["full_row_count"] is None
    assert "rows" not in context["data_readiness"]["sample"]
    assert "original" not in json.dumps(context["data_readiness"])


def test_readiness_required_actions_append_single_source_exit_lines(plans):
    """就绪说明的出口行单一来源（R87）：门禁句之后附 next_action_phrase 的人话——
    方案推荐 Agent 的「需要先完善数据」与 CLI 尾行、页面提示同词汇，不为 Agent
    另造一套出口；ready_for_training_preflight 不在门禁表里，required_actions
    保持为空，也不在此编造出口。"""
    from src.workbench.intake_models import Question
    from src.workbench.intake_service import next_action_phrase

    service, session, _ = plans
    # 夹具基态：全量已确认且分区在场——就绪态没有门禁句，出口行也不出现。
    ready = service.context(session, [])["data_readiness"]
    assert ready["next_action"] == "ready_for_training_preflight"
    assert ready["required_actions"] == []

    # Agent 产物三态（R86 已给 CLI 尾行出口；R87 补方案上下文同一词汇）。
    questioning = session.model_copy(deep=True)
    questioning.analysis.questions = [
        Question(question_id="q1", question="类别以哪次审核为准？", why="同一工单存在两次审核。")
    ]
    gapped = session.model_copy(deep=True)
    gapped.analysis.questions = []
    gapped.analysis.capability_gaps = ["任务需要多源组合，当前入口未配置。"]
    recipeless = session.model_copy(deep=True)
    recipeless.analysis.questions = []
    recipeless.analysis.capability_gaps = []
    recipeless.preview = None
    for state, expected, gate_keyword in (
        (questioning, "needs_business_answers", "business.questions"),
        (gapped, "needs_capability", "缺口"),
        (recipeless, "needs_recipe", "样例预览"),
    ):
        readiness = service.context(state, [])["data_readiness"]
        assert readiness["next_action"] == expected
        lines = readiness["required_actions"]
        assert len(lines) == 2, f"{expected} 应为门禁句+出口行两元素"
        assert gate_keyword in lines[0], f"{expected} 的门禁句保持在出口行之前"
        assert lines[1] == next_action_phrase(expected)
    answers = service.context(questioning, [])["data_readiness"]["required_actions"][1]
    assert "analyze --answer" in answers
    capability = service.context(gapped, [])["data_readiness"]["required_actions"][1]
    assert "add-source" in capability and "调整目标" in capability
    assert "baseline-analyze" not in capability
    recipe = service.context(recipeless, [])["data_readiness"]["required_actions"][1]
    assert "analyze" in recipe and "重新分析" in recipe
    assert "baseline-analyze" not in recipe

    # R82 nit-① 随同一机制收口：needs_data_revision 的就绪说明此前缺零密钥重分析
    # 出口，现在两条路径都进入出口行（改 counts 把状态机路由到该态）。
    broken = session.model_copy(deep=True)
    broken.preview.counts["invalid"] = 1
    readiness = service.context(broken, [])["data_readiness"]
    assert readiness["next_action"] == "needs_data_revision"
    exit_line = readiness["required_actions"][1]
    assert exit_line == next_action_phrase("needs_data_revision")
    assert "analyze" in exit_line and "baseline-analyze" in exit_line


@pytest.mark.parametrize(
    ("expected", "gate_keyword"),
    [
        ("awaiting_analysis", "数据分析"),
        ("needs_labels", "监督标签"),
        ("review_preview", "真实样例"),
        ("awaiting_full_data", "样例依据"),
        ("awaiting_full_validation", "实际校验"),
        ("needs_full_data_revision", "full_issues"),
        ("review_full_data", "全量转换"),
        ("awaiting_dataset_split", "分区"),
    ],
)
def test_readiness_exit_lines_cover_remaining_gating_states(plans, expected, gate_keyword):
    """其余门禁态的出口行补钉（R87 nit-① + R88 should-fix）：状态机八态逐一
    路由进 context()，门禁句在前、next_action_phrase 单源出口行在后——连同
    R87 主测试四态，门禁表 12 态在 context() 面全部有 [1] 等值钉；stale 覆盖
    句只换门禁句、出口行仍取同一 next_action_phrase(action)，其等值由
    awaiting_full_* 两态的钉确定性传递；13 态审计在方案上下文面上不留只验
    词汇不验等值的缺口（ready 态空列表钉在 R87 主测试）。"""
    from src.workbench.intake_service import next_action_phrase

    service, session, _ = plans
    state = session.model_copy(deep=True)
    if expected == "awaiting_analysis":
        state.analysis = None
    elif expected == "needs_labels":
        # 与判定单一来源同口径：needs_label 行正是 answer-sheet 清单筛出的行；
        # counts 不参与该路由（needs_data_revision 门只读 invalid/conflict 计数）。
        state.preview.rows[0].status = "needs_label"
    elif expected == "review_preview":
        state.confirmed_revision = None
    elif expected == "awaiting_full_data":
        state.full_data = None
    elif expected == "awaiting_full_validation":
        state.full_data = None
        state.source.scope = "full"
    elif expected == "needs_full_data_revision":
        state.full_data.status = "needs_revision"
    elif expected == "review_full_data":
        state.full_data.status = "review"
    else:
        # awaiting_dataset_split：全量已确认但独立分区缺失（dataset=None）。
        state.dataset = None
    readiness = service.context(state, [])["data_readiness"]
    assert readiness["next_action"] == expected
    lines = readiness["required_actions"]
    assert len(lines) == 2, f"{expected} 应为门禁句+出口行两元素"
    assert gate_keyword in lines[0], f"{expected} 的门禁句保持在出口行之前"
    assert lines[1] == next_action_phrase(expected)
