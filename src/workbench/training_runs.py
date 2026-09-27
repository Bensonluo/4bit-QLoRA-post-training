"""Hand confirmed dataset versions to the existing SFT subprocess runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.workbench.intake_models import IntakeSession
from src.workbench.materialize import dataset_is_current
from src.workbench.sources import content_digest
from src.workbench.training_preflight import load_local_tokenizer, preflight_dataset

_RUNNERS: dict[str, Any] = {}
TERMINAL = {"succeeded", "failed", "stopped"}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(path: Path) -> dict[str, Any]:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("请选择包含 config.json、tokenizer 和模型权重的本地模型目录。")
    weights = sorted([*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")])
    if not weights:
        raise ValueError("本地模型目录缺少 safetensors 或 pytorch_model 权重；不会自动下载。")
    files = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix in {".json", ".txt", ".model", ".tiktoken"}:
            files[item.name] = file_digest(item)
    return {
        "path": str(path),
        "config_and_tokenizer_hashes": files,
        "weights": [
            {"name": item.name, "size": item.stat().st_size, "sha256": file_digest(item)}
            for item in weights
        ],
        "weight_identity_note": "基础模型权重、配置及 tokenizer 文件均记录实际内容 SHA256。",
    }


def training_tokenizer(path: str | Path):
    tokenizer = load_local_tokenizer(path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "tokenizer 同时缺少 pad/eos；当前训练加载器不会扩展模型词嵌入，不能擅自增加 token 后启动。"
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _removed_environment() -> list[str]:
    # Explicit runtime allowlist: arbitrary BYOK env names never reach training.
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "SYSTEMROOT",
        "WINDIR",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONUTF8",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
    }
    prefixes = ("OMP_", "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_")
    return [name for name in os.environ if name not in allowed and not name.startswith(prefixes)]


class TrainingRunService:
    def __init__(
        self,
        root: str | Path,
        project_root: str | Path | None = None,
        python_executable: str | None = None,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = (
            Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        )
        self.python_executable = python_executable or sys.executable

    @property
    def runner(self):
        from src.tracking.runner import TrainingRunner

        key = str(self.project_root)
        if key not in _RUNNERS:
            _RUNNERS[key] = TrainingRunner(str(self.project_root))
        else:
            # Another CLI/UI process may have registered runs since this instance loaded.
            _RUNNERS[key]._run_meta = _RUNNERS[key]._load_meta()
        return _RUNNERS[key]

    def _directory(self, run_id):
        if not re.fullmatch(r"wb-[0-9a-f]{32}", run_id):
            raise ValueError("无效运行 ID。")
        return self.root / run_id

    def _load(self, run_id):
        path = self._directory(run_id) / "run.json"
        if not path.exists():
            raise ValueError("找不到该训练运行。")
        return json.loads(path.read_text(encoding="utf-8"))

    def prepare(
        self,
        session: IntakeSession,
        model_path: str | Path,
        *,
        max_length: int = 1024,
        training_options: dict | None = None,
        lora_options: dict | None = None,
        model_options: dict | None = None,
    ) -> dict:
        from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
        from config.sft import SFTConfig
        from src.utils.platform_utils import get_platform

        run_id = "wb-" + uuid4().hex
        directory = self._directory(run_id)
        directory.mkdir()
        path = Path(model_path).expanduser().resolve()
        record = {
            "run_id": run_id,
            "session_id": session.session_id,
            "session_revision": session.revision,
            "dataset_version": session.dataset.version if session.dataset else None,
            "model_path": str(path),
            "output_dir": str(directory / "model"),
            "status": "blocked",
            "issues": [],
            "config": None,
            "preflight": None,
            "artifacts": {},
            "metrics": {},
            "failure": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if not dataset_is_current(session):
                raise ValueError("请先确认当前全量数据并生成独立分区。")
            # 盲标核验门禁：监督信号的业务含义必须经用户独立复现（隐藏答案对抽样行作答
            # 并与数据标签一致）。核验与全量来源/配方摘要绑定，数据或方案变化后自动失效。
            verification = getattr(session, "label_verification", None)
            if not (
                isinstance(verification, dict) and verification.get("verdict") == "verified"
            ):
                raise ValueError(
                    "请先完成盲标核验：对抽样的已标注行隐藏答案独立作答，并与数据标签一致，"
                    "再准备训练。这是确认监督信号业务含义的必要步骤，不能跳过。"
                )
            if type(max_length) is not int or max_length <= 0:
                raise ValueError("max_length 必须是正整数。")
            train_options, model_overrides = dict(training_options or {}), dict(model_options or {})
            if "output_dir" in train_options or any(
                key in model_overrides for key in ("name", "max_length", "trust_remote_code")
            ):
                raise ValueError(
                    "运行输出、模型路径、长度和远程代码策略由本次已确认交接固定，不能在附加参数中覆盖。"
                )
            platform = get_platform()
            model = ModelConfig(
                **{
                    "name": str(path),
                    "quantization_bits": 4 if platform.is_cuda else None,
                    "torch_dtype": "bfloat16" if platform.is_cuda else "float32",
                    "trust_remote_code": False,
                    "use_flash_attention": False,
                    "max_length": max_length,
                    **model_overrides,
                }
            )
            training = TrainingConfig(
                **{
                    "output_dir": record["output_dir"],
                    "num_epochs": 1,
                    "bf16": platform.is_cuda and model.torch_dtype == "bfloat16",
                    "fp16": False,
                    **train_options,
                }
            )
            if any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
                for value in (
                    training.num_epochs,
                    training.batch_size,
                    training.gradient_accumulation_steps,
                    training.learning_rate,
                )
            ):
                raise ValueError("训练轮数、batch、梯度累积及学习率必须是有限正数。")
            if (
                type(training.batch_size) is not int
                or type(training.gradient_accumulation_steps) is not int
            ):
                raise ValueError("batch 与梯度累积必须是正整数。")
            lora = LoRAConfig(
                **{"r": 8, "lora_alpha": 16, "target_modules": "all-linear", **(lora_options or {})}
            )
            logging = LoggingConfig(
                use_tensorboard=False,
                use_mlflow=True,
                mlflow_tracking_uri=f"sqlite:///{self.root / 'mlflow.db'}",
                mlflow_experiment_name="workbench-sft",
                mlflow_run_name=run_id,
            )
            data = DataConfig(**session.dataset.data_config)
            SFTConfig(model=model, training=training, lora=lora, logging=logging, data=data)
            record["model_identity"] = model_identity(path)
            preflight = preflight_dataset(session, training_tokenizer(path), max_length)
            record["preflight"] = preflight
            record["issues"] = preflight["issues"]
            config = {
                "model": asdict(model),
                "training": asdict(training),
                "lora": asdict(lora),
                "data": asdict(data),
                "logging": asdict(logging),
            }
            record["config"] = config
            record["config_digest"] = content_digest(config)
            record["dataset"] = session.dataset.model_dump()
            if preflight["status"] != "blocked":
                record["status"] = "prepared"
            write_json(directory / "session.json", session.model_dump())
            write_json(directory / "config.json", config)
        except (ValueError, TypeError, OSError, ImportError) as exc:
            record["issues"].append(
                {"code": "prepare_failed", "severity": "blocking", "message": str(exc)}
            )
        write_json(directory / "run.json", record)
        return record

    def start(
        self,
        run_id: str,
        session: IntakeSession,
        *,
        acknowledge_warnings: bool = False,
        recover_technical_failures: bool = False,
    ) -> dict:
        record = self._load(run_id)
        if type(recover_technical_failures) is not bool:
            raise ValueError("技术恢复授权必须明确为布尔值。")
        if recover_technical_failures and record.get("recovery_parent_run_id"):
            raise ValueError("技术重试不能再次授权自动恢复。")
        if record["status"] != "prepared":
            raise ValueError("只有已准备且未启动的运行可以启动；失败或停止后请准备新的运行。")
        if (
            not dataset_is_current(session)
            or session.session_id != record["session_id"]
            or session.dataset.model_dump() != record["dataset"]
        ):
            raise ValueError("业务方案或数据版本已变化，请重新准备训练。")
        if model_identity(Path(record["model_path"])) != record["model_identity"]:
            raise ValueError("本地模型或 tokenizer 文件发生变化，请重新准备训练。")
        report = preflight_dataset(
            session,
            training_tokenizer(record["model_path"]),
            record["config"]["model"]["max_length"],
        )
        if report["status"] == "blocked":
            raise ValueError("启动前数据校验未通过，请处理预检问题。")
        if report["status"] == "warnings" and not acknowledge_warnings:
            raise ValueError("预检存在需核对的警告，请查看并明确确认后启动。")
        directory = self._directory(run_id)
        if (directory / "stop.request").exists():
            raise ValueError("此运行已取消，不能启动。")
        # Exclusive marker prevents duplicate starts from separate UI/CLI requests.
        try:
            (directory / "started").touch(exist_ok=False)
        except FileExistsError:
            raise ValueError("此运行已经提交启动，不能重复提交。") from None
        config = dict(record["config"])
        config["workbench"] = {"run_dir": str(directory), "run_id": run_id}
        record["recover_technical_failures"] = recover_technical_failures
        record["acknowledge_warnings"] = acknowledge_warnings
        record["execution"] = {
            "project_root": str(self.project_root),
            "python_executable": self.python_executable,
        }
        # Worker must see authorization before launch, even if it fails immediately.
        write_json(directory / "run.json", record)
        try:
            self.runner.launch_training(
                "workbench_sft",
                config,
                run_id,
                python_executable=self.python_executable,
                env_remove=_removed_environment(),
            )
            record["status"] = "running"
            record["started_at"] = datetime.now(timezone.utc).isoformat()
        except (OSError, ValueError) as exc:
            record["status"] = "failed"
            record["failure"] = {"stage": "launch", "type": type(exc).__name__, "message": str(exc)}
        write_json(directory / "run.json", record)
        return record

    def get_status(self, run_id: str) -> dict:
        record = self._load(run_id)
        receipt = self._directory(run_id) / "result.json"
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if result.get("run_id") != run_id:
                raise ValueError("运行结果身份不匹配。")
            record.update(result)
        if record["status"] in {"running", "stopping", "unknown"}:
            status = self.runner.get_status(run_id)
            if status in {"finished", "failed"}:
                requested_stop = (self._directory(run_id) / "stop.request").exists()
                record["status"] = "stopped" if requested_stop else "failed"
                record["failure"] = {
                    "stage": "worker",
                    "type": "UserStopped" if requested_stop else "MissingResult",
                    "message": "训练已按请求停止。"
                    if requested_stop
                    else "训练进程已退出但没有有效结果记录，请查看日志。",
                }
            elif status == "unknown":
                record["status"] = "unknown"
            elif (self._directory(run_id) / "stop.request").exists():
                record["status"] = "stopping"
        if record["status"] == "succeeded":
            try:
                manifest = json.loads(Path(record["artifacts"]["manifest"]).read_text())
                if (
                    manifest["run_id"] != run_id
                    or manifest["dataset_version"] != record["dataset_version"]
                    or manifest["config_digest"] != record["config_digest"]
                ):
                    raise ValueError("训练产物身份不匹配。")
                for artifact in manifest["files"].values():
                    if file_digest(Path(artifact["path"])) != artifact["sha256"]:
                        raise ValueError("训练产物已修改，内容哈希不匹配。")
            except (OSError, ValueError, KeyError) as exc:
                record["status"] = "failed"
                record["failure"] = {
                    "stage": "artifacts",
                    "type": "ArtifactMismatch",
                    "message": str(exc),
                }
        recovery = self._directory(run_id) / "recovery.json"
        record["recovery"] = json.loads(recovery.read_text()) if recovery.exists() else None
        return record

    def read_logs(self, run_id: str, tail: int = 100) -> str:
        self._load(run_id)
        if type(tail) is not int or not 1 <= tail <= 10000:
            raise ValueError("日志行数必须在 1 到 10000 之间。")
        return self.runner.read_recent_logs(run_id, tail=tail)

    def stop(self, run_id: str) -> dict:
        record = self.get_status(run_id)
        directory = self._directory(run_id)
        # Cancellation is visible even while dispatcher waits or prepare hashes weights.
        (directory / "stop.request").touch()
        from src.tracking.runner import _metadata_lock

        with _metadata_lock(directory / "recovery.lock"):
            recovery = directory / "recovery.json"
            if recovery.exists():
                state = json.loads(recovery.read_text())
                if state.get("child_run_id"):
                    self.stop(state["child_run_id"])
                state.update(
                    status="cancelled", reason="用户停止请求已取消自动恢复并转发至技术重试。"
                )
                write_json(recovery, state)
        if record["status"] not in {"running", "stopping", "unknown"}:
            return self.get_status(run_id)
        self.runner.stop_training(run_id)
        record["status"] = "stopping"
        write_json(self._directory(run_id) / "run.json", record)
        return self.get_status(run_id)

    def list_runs(self, session_id: str | None = None) -> list[dict]:
        records = [
            self.get_status(path.parent.name) for path in sorted(self.root.glob("wb-*/run.json"))
        ]
        return [
            record for record in records if session_id is None or record["session_id"] == session_id
        ]
