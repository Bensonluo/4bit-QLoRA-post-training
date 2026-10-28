"""A completed task run must not initialize MLflow while displayed read-only."""

from copy import deepcopy
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.unit.test_business_evaluation import task

ROOT = Path(__file__).resolve().parents[2]


def tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in root.rglob("*")
    }


def test_display_completed_run_does_not_initialize_mlflow(monkeypatch, tmp_path):
    from src.workbench import local_models, registry_link, training_runs
    from ui import config, runtime
    from ui.components import agent_settings

    root = tmp_path / "workspace"
    _, session = task.__wrapped__(root / "outputs/workbench")
    run = {
        "run_id": "completed-fixture",
        "session_id": session.session_id,
        "status": "succeeded",
        "model_path": "/nonexistent/base",
        "output_dir": "/nonexistent/adapter",
        "dataset_version": session.dataset.version,
        "config": {},
        "preflight": {"status": "passed", "issues": []},
        "issues": [],
    }

    class TrainingFixture:
        def __init__(self, directory, project_root=None, *, read_only=False):
            assert read_only
            self.root = Path(directory)

        def list_runs(self, session_id=None):
            return [deepcopy(run)] if session_id == session.session_id else []

        def get_status(self, run_id):
            assert run_id == run["run_id"]
            return deepcopy(run)

        def read_logs(self, run_id, tail=100):
            return ""

    original = registry_link.run_registration_status
    calls = []

    def watch_real_registry(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(registry_link, "run_registration_status", watch_real_registry)
    monkeypatch.setattr(training_runs, "TrainingRunService", TrainingFixture)
    monkeypatch.setattr(local_models, "discover_local_models", lambda: [])
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"file://{root / 'mlruns'}")
    monkeypatch.setattr(runtime, "DISPLAY_ONLY", True)
    monkeypatch.setattr(agent_settings, "DISPLAY_ONLY", True)
    for name in ("PROVIDER", "BASE_URL", "MODEL", "API_KEY"):
        monkeypatch.delenv(f"TUNESMITH_AGENT_{name}", raising=False)

    before = tree(root)
    page = AppTest.from_file(str(ROOT / "ui/views/07_Data_Intake.py"), default_timeout=45).run()
    assert not page.exception, [item.message for item in page.exception]
    assert any(item.value == session.goal for item in page.title)
    assert any("训练完成" in item.label for item in page.expander)
    assert any("展示环境暂不读取本地模型注册状态" in item.value for item in page.caption)
    assert not calls
    assert not (root / "outputs/workbench/training/mlflow.db").exists()
    assert tree(root) == before
