"""Workbench handoff preserves data identity and executes the existing SFT worker."""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("datasets")
pytest.importorskip("peft")

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from src.tracking.runner import TrainingRunner
from src.workbench.intake_service import IntakeService
from src.workbench.training_runs import TrainingRunService, write_json
from tests.unit.test_data_materialize import _full


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    model = tmp_path / "tiny-model"
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "yes": 4}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 2), ("[EOS]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=5,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_positions=64,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=1,
        )
    ).save_pretrained(model)
    intake = IntakeService(tmp_path / "intake")
    session = _full(intake)
    session = intake.materialize_dataset(session.session_id, session.revision)
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(
        Path(__file__).resolve().parents[2] / "scripts" / "workbench_train.py",
        project / "scripts" / "workbench_train.py",
    )
    service = TrainingRunService(
        tmp_path / "runs", project_root=project, python_executable=sys.executable
    )
    return intake, session, service, model


def _prepare(environment, **options):
    _, session, service, model = environment
    return service.prepare(
        session,
        model,
        max_length=32,
        training_options={
            "num_epochs": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "logging_steps": 1,
        },
        **options,
    )


def _wait(service, run_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        record = service.get_status(run_id)
        if record["status"] in {"succeeded", "failed", "stopped"}:
            return record
        time.sleep(0.1)
    service.stop(run_id)
    pytest.fail("Local tiny-model worker did not finish within 60 seconds")


def test_prepare_uses_matching_model_tokenizer_and_confirmed_data(environment):
    _, session, service, model = environment
    record = _prepare(environment)
    assert record["status"] == "prepared"
    assert record["preflight"]["status"] == "passed"
    assert record["config"]["model"]["name"] == str(model)
    assert record["config"]["data"]["dataset_name"] == session.dataset.paths["train"]
    assert record["config"]["data"]["validation_split"] == 0
    assert record["config"]["data"]["dataset_loader"] == "alpaca"
    assert record["config"]["logging"]["use_wandb"] is False
    assert service.list_runs(session.session_id)[0]["run_id"] == record["run_id"]
    assert service.list_runs("unrelated") == []


def test_changed_session_or_model_cannot_start(environment):
    intake, session, service, model = environment
    record = _prepare(environment)
    changed = intake.answer(session.session_id, "修正数据目标")
    with pytest.raises(ValueError, match="数据版本已变化"):
        service.start(record["run_id"], changed)
    config_path = model / "tokenizer_config.json"
    config_path.write_text(config_path.read_text() + "\n")
    with pytest.raises(ValueError, match="文件发生变化"):
        service.start(record["run_id"], session)


def test_same_size_and_timestamp_weight_change_cannot_reuse_prepared_identity(environment):
    _, session, service, model = environment
    record = _prepare(environment)
    weights = next(model.glob("*.safetensors"))
    before = weights.stat()
    content = bytearray(weights.read_bytes())
    content[-1] ^= 1
    weights.write_bytes(content)
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="文件发生变化"):
        service.start(record["run_id"], session)


def test_missing_local_weights_and_invalid_parameters_are_reviewable_blockers(environment):
    _, session, service, model = environment
    record = service.prepare(session, model / "missing", max_length=32)
    assert record["status"] == "blocked"
    assert record["issues"][-1]["severity"] == "blocking"
    record = service.prepare(session, model, training_options={"output_dir": "/tmp/other"})
    assert record["status"] == "blocked"


def test_runner_uses_explicit_python_and_excludes_arbitrary_agent_secrets(environment, monkeypatch):
    _, session, service, _ = environment
    monkeypatch.setenv("TUNESMITH_AGENT_API_KEY", "not-for-training")
    monkeypatch.setenv("MY_CUSTOM_CREDENTIAL", "private-token")
    record = _prepare(environment)
    service.python_executable = "/tmp/selected-python"
    process = MagicMock()
    process.pid = 12345
    process.poll.return_value = None
    with patch("src.tracking.runner.subprocess.Popen", return_value=process) as spawn:
        started = service.start(record["run_id"], session)
    assert started["status"] == "running"
    args, kwargs = spawn.call_args
    assert args[0][0] == "/tmp/selected-python"
    assert "TUNESMITH_AGENT_API_KEY" not in kwargs["env"]
    assert "MY_CUSTOM_CREDENTIAL" not in kwargs["env"]
    with pytest.raises(ValueError, match="未启动"):
        service.start(record["run_id"], session)
    for path in (service.root / record["run_id"]).glob("*.json"):
        assert "private-token" not in path.read_text()


def test_warning_confirmation_is_specific_to_actual_tokenizer(environment, monkeypatch):
    _, session, service, model = environment
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model, local_files_only=True)
    # pad==eos is verified safe (test_sft_eos_contract); the tokenizer-specific
    # warning used here is a declared context length below max_length, which only
    # the actual tokenizer file on disk can report.
    tokenizer.model_max_length = 16
    tokenizer.save_pretrained(model)
    record = _prepare(environment)
    assert record["status"] == "prepared"
    assert record["preflight"]["status"] == "warnings"
    with pytest.raises(ValueError, match="警告"):
        service.start(record["run_id"], session)


def test_real_tiny_sft_worker_writes_traceable_adapter_and_survives_ui_reload(environment):
    _, session, service, _ = environment
    record = _prepare(environment)
    service.start(record["run_id"], session)
    result = _wait(service, record["run_id"])
    assert result["status"] == "succeeded", service.read_logs(record["run_id"], tail=100)
    assert Path(result["artifacts"]["adapter_weights"]).is_file()
    assert result["metrics"]["train_loss"] >= 0
    manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text())
    assert manifest["dataset_version"] == session.dataset.version
    assert "sha256" in manifest["model_identity"]["weights"][0]
    assert manifest["mlflow_run_id"] == result["mlflow_run_id"]
    assert result["mlflow_status"] in {"recorded", "unavailable", "not_started", "lookup_failed"}
    reloaded = TrainingRunService(service.root, service.project_root)
    assert reloaded.get_status(record["run_id"])["status"] == "succeeded"
    assert "Training Complete" in reloaded.read_logs(record["run_id"], tail=100)
    Path(result["artifacts"]["adapter_weights"]).write_bytes(b"modified")
    assert reloaded.get_status(record["run_id"])["failure"]["type"] == "ArtifactMismatch"


def test_real_mlflow_run_can_be_reopened_with_dataset_and_config_tags(environment):
    pytest.importorskip("mlflow")
    from mlflow.tracking import MlflowClient

    _, session, service, _ = environment
    record = _prepare(environment)
    service.start(record["run_id"], session)
    result = _wait(service, record["run_id"])
    assert result["status"] == "succeeded", service.read_logs(record["run_id"], tail=100)
    assert result["mlflow_status"] == "recorded"
    assert result["mlflow_tracking_uri"].startswith("sqlite:///")
    assert result["mlflow_run_id"]
    assert result["mlflow_run_id"] != record["run_id"]
    client = MlflowClient(tracking_uri=result["mlflow_tracking_uri"])
    run = client.get_run(result["mlflow_run_id"])
    assert run.data.tags["workbench.run_id"] == record["run_id"]
    assert run.data.tags["workbench.dataset_version"] == session.dataset.version
    assert run.data.tags["workbench.config_digest"] == record["config_digest"]
    assert run.data.params["data.dataset_name"] == session.dataset.paths["train"]
    assert run.info.status == "FINISHED"
    assert any("loss" in name for name in run.data.metrics)
    manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text())
    assert manifest["mlflow_run_id"] == run.info.run_id
    reread = TrainingRunService(service.root, service.project_root).get_status(record["run_id"])
    assert reread["mlflow_run_id"] == run.info.run_id


def test_failed_worker_retains_failure_and_logs(environment):
    _, session, service, _ = environment
    record = _prepare(environment, lora_options={"target_modules": ["nonexistent_module"]})
    service.start(record["run_id"], session)
    result = _wait(service, record["run_id"])
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "training"
    assert result["artifacts"] == {}
    assert "nonexistent_module" in service.read_logs(record["run_id"], tail=100)


def test_stop_real_worker_before_training_does_not_claim_completion(environment):
    _, session, service, _ = environment
    record = _prepare(environment)
    service.start(record["run_id"], session)
    service.stop(record["run_id"])
    result = _wait(service, record["run_id"])
    assert result["status"] == "stopped"
    assert result["artifacts"] == {}


def test_running_receipt_is_not_success_when_process_has_exited(environment):
    _, session, service, _ = environment
    record = _prepare(environment)
    directory = service.root / record["run_id"]
    record["status"] = "running"
    write_json(directory / "run.json", record)
    write_json(directory / "result.json", {"run_id": record["run_id"], "status": "running"})
    with patch.object(TrainingRunner, "get_status", return_value="failed"):
        result = service.get_status(record["run_id"])
    assert result["status"] == "failed"
    assert result["failure"]["type"] == "MissingResult"


def test_separate_runner_instances_do_not_overwrite_each_others_runs(tmp_path):
    first = TrainingRunner(str(tmp_path))
    second = TrainingRunner(str(tmp_path))
    process = MagicMock()
    process.pid = 123
    process.poll.return_value = None
    with patch("src.tracking.runner.subprocess.Popen", return_value=process):
        first.launch_training("sft", {}, "one")
        second.launch_training("sft", {}, "two")
    first._record_exit("one", 0)
    latest = TrainingRunner(str(tmp_path))
    assert set(latest.list_all_runs()) == {"one", "two"}
    assert latest.get_run_info("one")["returncode"] == 0
    second.delete_run("two")
    latest = TrainingRunner(str(tmp_path))
    assert latest.list_all_runs() == ["one"]
    assert latest.get_run_info("one")["returncode"] == 0


def test_service_refreshes_metadata_when_another_process_registers_run(environment):
    _, _, service, _ = environment
    existing = service.runner
    external = TrainingRunner(str(service.project_root))
    process = MagicMock()
    process.pid = 123
    process.poll.return_value = None
    with patch("src.tracking.runner.subprocess.Popen", return_value=process):
        external.launch_training("sft", {}, "external-run")
    assert service.runner is existing
    assert service.runner.get_run_info("external-run")["pid"] == 123


def test_mlflow_reference_records_exact_run_id_without_claiming_missing_runs(monkeypatch):
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(variable, "1")
    from scripts.workbench_train import mlflow_reference

    client = MagicMock()
    client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="exp")
    client.search_runs.return_value = [
        SimpleNamespace(info=SimpleNamespace(run_id="actual-mlflow-id"))
    ]
    module = ModuleType("mlflow.tracking")
    module.MlflowClient = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", module)
    config = {
        "logging": {
            "mlflow_tracking_uri": "/tmp/local-tracking",
            "mlflow_experiment_name": "workbench-sft",
        }
    }
    record = {"run_id": "wb-123", "dataset_version": "data-version", "config_digest": "digest"}
    reference = mlflow_reference(config, record)
    assert reference["mlflow_run_id"] == "actual-mlflow-id"
    assert reference["mlflow_tracking_uri"] == "/tmp/local-tracking"
    assert reference["mlflow_status"] == "recorded"
    client.set_tag.assert_any_call("actual-mlflow-id", "workbench.dataset_version", "data-version")
    client.search_runs.return_value = []
    reference = mlflow_reference(config, record)
    assert reference["mlflow_run_id"] is None
    assert reference["mlflow_status"] == "not_started"
