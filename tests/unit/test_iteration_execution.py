"""Service-level guards: authorization scope, duplicate submission, stop, resume."""

import json
import sys
from types import SimpleNamespace

import pytest

from src.workbench import iteration_execution as execution_module
from src.workbench.intake_models import FullDataValidation, IntakeSession
from src.workbench.iteration_execution import IterationExecutionService, authorization_valid

IDENTITY = "it-" + "a" * 32


def _session(**fields):
    base = dict(
        session_id="s" * 32,
        goal="判断类别",
        revision=4,
        source=type("Source", (), {"digest": "source-digest"})(),
        sources={"main": type("Source", (), {"digest": "source-digest"})()},
        analysis=None,
        full_data=None,
        dataset=None,
        label_verification={"verdict": "verified"},
    )
    base.update(fields)
    return IntakeSession.model_construct(**base)


def test_authorization_binds_goal_sources_analysis_and_full_data():
    full = FullDataValidation.model_construct(status="confirmed", confirmed_revision=3)
    snapshot = _session(full_data=full)
    assert authorization_valid(_session(full_data=full), snapshot)
    # The worker's own materialization only touches the dataset; scope stays valid.
    assert authorization_valid(_session(full_data=full, dataset=object()), snapshot)
    assert not authorization_valid(_session(goal="另一个目标", full_data=full), snapshot)
    changed_source = type("Source", (), {"digest": "other"})()
    assert not authorization_valid(
        _session(
            full_data=full,
            sources={"main": changed_source, "extra": changed_source},
        ),
        snapshot,
    )
    assert not authorization_valid(_session(full_data=None), snapshot)
    other_full = FullDataValidation.model_construct(status="confirmed", confirmed_revision=4)
    assert not authorization_valid(_session(full_data=other_full), snapshot)


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    live = _session(
        full_data=FullDataValidation.model_construct(status="confirmed", confirmed_revision=3)
    )
    iterations = {
        IDENTITY: {
            "iteration_id": IDENTITY,
            "session_id": live.session_id,
            "goal": live.goal,
            "status": "confirmed",
        }
    }
    claims = []
    launches = []

    class IterationsFake:
        def __init__(self, iteration_root, training_root, evaluation_root):
            pass

        def get(self, iteration_id):
            return iterations[iteration_id]

        def claim_execution(self, iteration_id, execution_id):
            claims.append((iteration_id, execution_id))
            return iterations[iteration_id]

    class IntakeFake:
        def __init__(self, root):
            pass

        def load(self, session_id):
            return live

    monkeypatch.setattr(execution_module, "IterationService", IterationsFake)
    monkeypatch.setattr(execution_module, "IntakeService", IntakeFake)
    monkeypatch.setattr(
        execution_module.IterationExecutionService,
        "_launch",
        lambda self, record: launches.append(record) or dict(record),
    )
    service = IterationExecutionService(
        tmp_path / "executions",
        tmp_path / "intake",
        tmp_path / "iterations",
        tmp_path / "training",
        tmp_path / "eval",
    )
    return {
        "service": service,
        "live": live,
        "iterations": iterations,
        "claims": claims,
        "launches": launches,
        "monkeypatch": monkeypatch,
    }


def test_fresh_authorization_snapshots_session_and_claims_iteration(harness):
    started = harness["service"].start(IDENTITY, harness["live"])
    assert started["status"] == "queued"
    assert started["session_revision"] == harness["live"].revision
    assert started["options"] == {
        "acknowledge_warnings": False,
        "independent_rows_confirmed": False,
    }
    assert harness["claims"] == [(IDENTITY, IDENTITY)]
    assert len(harness["launches"]) == 1
    directory = harness["service"]._directory(IDENTITY)
    snapshot = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    assert snapshot["session_id"] == harness["live"].session_id
    assert snapshot["revision"] == harness["live"].revision


def test_authorization_rejects_stale_or_unconfirmed_iterations(harness):
    stale = _session(revision=5, full_data=harness["live"].full_data)
    with pytest.raises(ValueError, match="任务已更新"):
        harness["service"].start(IDENTITY, stale)
    harness["live"].full_data = None
    with pytest.raises(ValueError, match="确认全量数据"):
        harness["service"].start(IDENTITY, harness["live"])
    harness["iterations"][IDENTITY]["status"] = "prepared"
    with pytest.raises(ValueError, match="已确认范围"):
        harness["service"].start(IDENTITY, harness["live"])
    assert harness["service"].get(IDENTITY) is None


def test_duplicate_submission_never_relaunches_or_retrains(harness):
    harness["service"].start(IDENTITY, harness["live"])
    again = harness["service"].start(IDENTITY, harness["live"])
    assert again["status"] == "queued"
    assert len(harness["launches"]) == 1


def test_dead_worker_is_reported_not_silently_requeued(harness, harness2=None):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["worker_pid"] = 999999
    service._write(record)
    harness["monkeypatch"].setattr(execution_module, "_pid_alive", lambda pid: False)
    result = service.start(IDENTITY, harness["live"])
    assert result["status"] == "failed"
    # R99 去重:message 只报事实,排查指向(worker.log 位置/建议句)由 summarize_execution
    # 单处输出——精确钉取代旧 OR 钉:带 worker.log 的旧措辞分支彻底退场。
    assert result["message"] == "后台执行进程已退出且未完成。"
    assert harness["launches"] and len(harness["launches"]) == 1


def test_launch_missing_script_fails_without_phantom_log_path(tmp_path):
    """从未启动的失败不落幻影日志路径(R99):缺后台执行脚本时 worker.log 尚不存在,
    记录不得携带 log_path——摘要据此回退泛指句,不指去一个空路径。"""
    service = IterationExecutionService(
        tmp_path / "executions",
        tmp_path / "intake",
        tmp_path / "iterations",
        tmp_path / "training",
        tmp_path / "eval",
        project_root=tmp_path,
    )
    record = {
        "iteration_id": IDENTITY,
        "project_root": str(tmp_path),
        "python_executable": sys.executable,
        "status": "queued",
    }
    # 生产前置:start/_authorize 在 _launch 前已建执行目录
    service._directory(IDENTITY).mkdir(parents=True, exist_ok=True)
    result = service._launch(record)
    assert result["status"] == "failed"
    assert "缺少后台执行脚本" in result["message"]
    assert "log_path" not in result


def test_launch_success_persists_log_path(tmp_path, monkeypatch):
    """启动成功即落盘真实日志路径(R99):摘要据此给出 worker.log 完整位置;
    Popen 打桩,不真起后台进程。project_root 指向 tmp 并造假脚本——不依赖真实
    仓库文件存在(采纳 r99-reviewer nit-1,与缺脚本测试对称自包含)。"""
    service = IterationExecutionService(
        tmp_path / "executions",
        tmp_path / "intake",
        tmp_path / "iterations",
        tmp_path / "training",
        tmp_path / "eval",
        project_root=tmp_path,
    )
    launched = []

    class FakePopen:
        pid = 4242

    monkeypatch.setattr(
        execution_module.subprocess, "Popen", lambda *a, **k: launched.append(a) or FakePopen()
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "workbench_iterate.py").write_text("# test stub\n", encoding="utf-8")
    record = {
        "iteration_id": IDENTITY,
        "project_root": str(service.project_root),
        "python_executable": sys.executable,
        "status": "queued",
    }
    # 生产前置:start/_authorize 在 _launch 前已建执行目录
    service._directory(IDENTITY).mkdir(parents=True, exist_ok=True)
    result = service._launch(record)
    assert result["status"] == "queued"
    assert result["worker_pid"] == 4242
    # R102 登记-1:启动 message 是纯事实短句——进度的解释与恢复边界由摘要单源输出,
    # 不在 message 里重复第二套口径
    assert result["message"] == "后台执行已启动。"
    assert result["log_path"] == str(service._directory(IDENTITY) / "worker.log")
    # append 模式打开即创建:路径不止是字符串,文件真实在场
    assert (service._directory(IDENTITY) / "worker.log").exists()
    assert launched and launched[0][0][0] == sys.executable


def test_warning_ack_resume_requires_explicit_acknowledgement(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "awaiting_warning_ack"
    record["run_id"] = "wb-" + "1" * 32
    service._write(record)
    unchanged = service.start(IDENTITY, harness["live"])
    assert unchanged["status"] == "awaiting_warning_ack"
    assert len(harness["launches"]) == 1
    resumed = service.start(IDENTITY, harness["live"], acknowledge_warnings=True)
    assert resumed["status"] == "queued"
    assert resumed["options"]["acknowledge_warnings"] is True
    assert len(harness["launches"]) == 2


def test_terminal_states_are_final(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "completed"
    record["evaluation_id"] = "e" * 32
    service._write(record)
    duplicate = service.start(IDENTITY, harness["live"])
    assert duplicate["evaluation_id"] == record["evaluation_id"]
    assert len(harness["launches"]) == 1
    blocked = service.start(IDENTITY, harness["live"])  # completed stays idempotent
    assert blocked["status"] == "completed"
    for terminal in ("blocked", "failed", "stopped"):
        record["status"] = terminal
        service._write(record)
        with pytest.raises(ValueError, match="不能重复提交"):
            service.start(IDENTITY, harness["live"])


def test_stop_touches_request_stops_training_and_is_final(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "training"
    record["run_id"] = "wb-" + "2" * 32
    service._write(record)
    stopped = []
    training_instances = []

    class TrainingFake:
        def __init__(self, training_root, project_root=None, python_executable=None):
            training_instances.append(self)

        def stop(self, run_id):
            stopped.append(run_id)

    harness["monkeypatch"].setattr(execution_module, "TrainingRunService", TrainingFake)
    result = service.stop(IDENTITY)
    assert result["status"] == "stopped"
    assert stopped == [record["run_id"]]
    assert (service._directory(IDENTITY) / "stop.request").exists()
    with pytest.raises(ValueError, match="不能重复提交"):
        service.start(IDENTITY, harness["live"])
    assert service.stop(IDENTITY)["status"] == "stopped"


def test_stop_unknown_iteration_is_an_error(harness):
    with pytest.raises(ValueError, match="尚未提交自动执行"):
        harness["service"].stop("it-" + "b" * 32)


def test_worker_honors_preexisting_stop_request_without_advancing(harness):
    service = harness["service"]
    record = service.start(IDENTITY, harness["live"])
    record["status"] = "queued"
    service._write(record)
    (service._directory(IDENTITY) / "stop.request").touch()
    assert service.run_worker(IDENTITY) is False
    assert service.get(IDENTITY)["status"] == "stopped"


def _worker_world(tmp_path, monkeypatch):
    """R100 六态 worker 仿真世界：补齐 run_worker 走完全程所需的协作面。

    IterationsFake 在最小夹具之上加 training（get_status/runner）、prepare/start、
    _effective_training_run、evaluations.compare、bind_evaluation。live 会话用
    json-safe 载荷构建——授权快照要走 model_dump_json→model_validate_json 真实
    往返，model_construct 的宽松对象过不了 validate。dataset 在 start 之后 setattr：
    快照保持干净，live 侧带 evaluation_suite 供 needs_materialization 判断
    （dataset 只由 worker 自己的物化触碰，不破坏授权范围）。
    """
    src = {"name": "主表", "digest": "d-main", "format": "csv", "columns": ["a"], "rows": []}
    live = IntakeSession.model_validate(
        {
            "session_id": "s" * 32,
            "goal": "判断类别",
            "revision": 4,
            "source": src,
            "sources": {"main": dict(src)},
            "profile": {},
            "label_verification": {"verdict": "verified"},
            "full_data": {
                "source": dict(src),
                "profile": {},
                "sample_source_digest": "d-main",
                "approved_recipe_digest": "d-recipe",
                "sample_confirmed_revision": 3,
                "status": "confirmed",
                "confirmed_revision": 3,
            },
        }
    )
    run_id = "wb-" + "3" * 32
    parent_run_id = "wb-" + "4" * 32
    iteration = {
        "iteration_id": IDENTITY,
        "session_id": live.session_id,
        "goal": live.goal,
        "status": "confirmed",
        "evaluation_suite": "suite-r100",
        "model_path": "/models/base",
        "parent_run_id": parent_run_id,
        "evaluation_protocol": {
            "scorer": "classification_exact",
            "fields": (),
            "max_new_tokens": 64,
            "strip_whitespace": True,
        },
    }
    iterations = {IDENTITY: iteration}
    knobs = {
        "prepared_run": {"status": "prepared", "preflight": {"status": "passed", "issues": []}},
        "report": SimpleNamespace(status="completed", notes=[], evaluation_id="ev-r100"),
        "bound": None,
    }

    class RunnerFake:
        def get_status(self, run_id):
            return "exited"

    class TrainingFake:
        # _iterations() 会用 self._training() 覆写 service.training——真实
        # TrainingRunService 走 _load(run.json) 会 raise,这里按其构造签名整体打桩。
        runner = RunnerFake()

        def __init__(self, training_root, project_root=None, python_executable=None):
            pass

        def get_status(self, run_id):
            if run_id == parent_run_id:
                return {"status": "succeeded", "model_path": "/models/base", "output_dir": None}
            return knobs["prepared_run"]

        def stop(self, run_id):
            pass

    class EvaluationsFake:
        def compare(self, live, models, protocol):
            return knobs["report"]

    class IterationsFake:
        evaluations = EvaluationsFake()

        def __init__(self, iteration_root, training_root, evaluation_root):
            pass

        def get(self, iteration_id):
            return iterations[iteration_id]

        def claim_execution(self, iteration_id, execution_id):
            return iterations[iteration_id]

        def prepare(self, iteration_id, live, execution_id=None):
            iteration.update(status="prepared", new_run_id=run_id)
            return iteration

        def start(self, iteration_id, live, acknowledge_warnings=False, execution_id=None):
            iteration.update(status="running")
            return iteration

        def _effective_training_run(self, run_id):
            return {"status": "succeeded", "model_path": "/models/base", "output_dir": "/runs/cur"}

        def bind_evaluation(self, iteration_id, live, evaluation_id, execution_id=None):
            knobs["bound"] = evaluation_id

    class IntakeFake:
        def __init__(self, root):
            pass

        def load(self, session_id):
            return live

    launches = []
    monkeypatch.setattr(execution_module, "IterationService", IterationsFake)
    monkeypatch.setattr(execution_module, "TrainingRunService", TrainingFake)
    monkeypatch.setattr(execution_module, "IntakeService", IntakeFake)
    monkeypatch.setattr(execution_module, "RELEASE_SETTLE_SECONDS", 0)
    monkeypatch.setattr(
        execution_module.IterationExecutionService,
        "_launch",
        lambda self, record: launches.append(record) or dict(record),
    )
    service = IterationExecutionService(
        tmp_path / "executions",
        tmp_path / "intake",
        tmp_path / "iterations",
        tmp_path / "training",
        tmp_path / "eval",
    )
    service.start(IDENTITY, live)
    live.dataset = SimpleNamespace(evaluation_suite="suite-r100")
    return {"service": service, "live": live, "iteration": iteration, "knobs": knobs}


def test_worker_authorization_drift_blocks_with_pure_fact_message(tmp_path, monkeypatch):
    """R100 六态收敛①：授权失效的 message 只报事实。「请重新核对后提出新的改进轮次」
    的行动指向由 summarize_execution 的 blocked 尾句单处输出，不与 message 同屏堆叠。"""
    world = _worker_world(tmp_path, monkeypatch)
    world["live"].goal = "另一个目标"
    assert world["service"].run_worker(IDENTITY) is False
    record = world["service"].get(IDENTITY)
    assert record["status"] == "blocked"
    assert record["message"] == "授权后业务目标、资料或数据方案已变化，原授权不能继续。"
    assert "提出新的改进轮次" not in record["message"]


def test_worker_training_blocked_message_is_pure_fact(tmp_path, monkeypatch):
    """R100 六态收敛②：训练准备未通过的 message 只报事实，问题清单原样落盘；
    「请查看问题后处理」退场——摘要 blocked 分支必给「处理问题后提出新的改进轮次」。"""
    world = _worker_world(tmp_path, monkeypatch)
    world["knobs"]["prepared_run"] = {"status": "blocked", "issues": ["数据版本过期"]}
    assert world["service"].run_worker(IDENTITY) is False
    record = world["service"].get(IDENTITY)
    assert record["status"] == "blocked"
    assert record["message"] == "训练准备未通过。"
    assert record["issues"] == ["数据版本过期"]


def test_worker_preflight_blocked_message_is_pure_fact(tmp_path, monkeypatch):
    """R100 六态收敛③：启动前数据校验未通过的 message 只报事实。"""
    world = _worker_world(tmp_path, monkeypatch)
    world["knobs"]["prepared_run"] = {
        "status": "prepared",
        "preflight": {
            "status": "blocked",
            "issues": [{"severity": "blocking", "message": "答案丢失"}],
        },
        "issues": ["长度不足"],
    }
    assert world["service"].run_worker(IDENTITY) is False
    record = world["service"].get(IDENTITY)
    assert record["status"] == "blocked"
    assert record["message"] == "启动前数据校验未通过。"
    assert record["issues"] == ["长度不足"]


def test_worker_preflight_warning_pauses_with_pure_fact_message(tmp_path, monkeypatch):
    """R100 六态收敛④：等确认暂停的 message 只报事实。「请查看后在原入口确认继续」
    的恢复指引由 summarize_execution 的 awaiting 分支单处输出（勾选句+命令行恢复）。"""
    world = _worker_world(tmp_path, monkeypatch)
    world["knobs"]["prepared_run"] = {
        "status": "prepared",
        "preflight": {
            "status": "warnings",
            "issues": [{"severity": "warning", "message": "3 行答案截断"}],
        },
    }
    assert world["service"].run_worker(IDENTITY) is False
    record = world["service"].get(IDENTITY)
    assert record["status"] == "awaiting_warning_ack"
    assert record["message"] == "预检存在需核对的提示。"
    assert record["issues"] == [{"severity": "warning", "message": "3 行答案截断"}]


def test_worker_incomplete_evaluation_blocks_with_pure_fact_message(tmp_path, monkeypatch):
    """R100 六态收敛⑤：开发集对照未完整完成的 message 只报事实，notes 转警示清单；
    「请查看评测报告」的查看指向交给摘要/页面。"""
    world = _worker_world(tmp_path, monkeypatch)
    world["knobs"]["report"] = SimpleNamespace(
        status="partial", notes=["两题生成失败"], evaluation_id=None
    )
    assert world["service"].run_worker(IDENTITY) is False
    record = world["service"].get(IDENTITY)
    assert record["status"] == "blocked"
    assert record["message"] == "开发集对照未完整完成。"
    assert record["issues"] == [{"severity": "warning", "message": "两题生成失败"}]


def test_worker_completed_message_is_pure_fact(tmp_path, monkeypatch):
    """R100 六态收敛⑥：完成的 message 只报事实。三态手抄「决定采用、继续或停止」
    退场——摘要 completed 句是四态口径（含「证据不足」），message 里的旧三态
    与之同屏漂移，收敛后词汇只在摘要单处。"""
    world = _worker_world(tmp_path, monkeypatch)
    assert world["service"].run_worker(IDENTITY) is True
    record = world["service"].get(IDENTITY)
    assert record["status"] == "completed"
    assert record["message"] == "开发集对照已完成。"
    assert record["evaluation_id"] == "ev-r100"
    assert world["knobs"]["bound"] == "ev-r100"
    assert "决定采用、继续或停止" not in record["message"]


def test_authorize_requires_verified_label_verification(harness):
    """数据修订后盲标核验失效时,自动执行在授权入口即被拒绝,不起后台进程。"""
    service = harness["service"]
    live = harness["live"]
    live.label_verification = None
    identity = "it-" + "b" * 32
    harness["iterations"][identity] = {
        "iteration_id": identity,
        "session_id": live.session_id,
        "goal": live.goal,
        "status": "confirmed",
    }
    with pytest.raises(ValueError, match="盲标核验"):
        service.start(identity, live)
    assert len(harness["launches"]) == 0
