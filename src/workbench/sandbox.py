"""OS-isolated, JSON-only long-tail transforms; never fall back to host execution.

The import check is an interface aid. The security boundary is Docker or macOS
Seatbelt, including when transform code circumvents Python-level restrictions.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 5
    cpu_seconds: int = 3
    memory_mb: int = 256
    max_input_bytes: int = 1024 * 1024
    max_output_bytes: int = 2 * 1024 * 1024

    def validate(self):
        for value in asdict(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("隔离资源边界必须是有限正数。")


@dataclass(frozen=True)
class TransformCase:
    name: str
    rows: list[dict[str, Any]]
    expected_rows: list[dict[str, Any]] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    kind: Literal["business", "counterexample"] = "business"
    expect_error: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    status: Literal["passed", "failed", "unavailable"]
    backend: str
    source_digest: str
    rows: list[dict[str, Any]] | None = None
    error: str = ""
    failure_kind: str = ""


@dataclass(frozen=True)
class ValidationReport:
    status: Literal["passed", "failed", "unavailable"]
    backend: str
    source_digest: str
    cases_digest: str
    limits: dict[str, Any]
    cases: list[dict[str, Any]]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def check_source(source: str) -> None:
    """Reject unsupported contracts, without claiming static checking is isolation."""
    if len(source.encode()) > 128 * 1024:
        raise ValueError("适配代码超过 128 KiB。")
    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "transform"
    ]
    if len(functions) != 1 or len(functions[0].args.args) != 2:
        raise ValueError("代码必须定义唯一的 transform(rows, config) 函数。")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
            )
            if any(name not in {"re", "json", "math"} for name in names) or getattr(
                node, "level", 0
            ):
                raise ValueError("适配代码仅允许导入 re、json、math 标准库。")


class TransformSandbox:
    def __init__(
        self,
        *,
        backend: str = "auto",
        docker_image: str = "python:3.12-slim",
        limits: SandboxLimits | None = None,
    ):
        if backend not in {"auto", "docker", "macos"}:
            raise ValueError("隔离后端必须是 auto、docker 或 macos。")
        self.requested_backend = backend
        self.docker_image = docker_image
        self.limits = limits or SandboxLimits()
        self.limits.validate()
        self.backend = ""
        self.runtime = ""
        self.unavailable_reason = ""

    def detect(self) -> bool:
        if self.backend:
            return True
        reasons = []
        if self.requested_backend in {"auto", "docker"} and shutil.which("docker"):
            try:
                found = subprocess.run(
                    ["docker", "image", "inspect", "--format", "{{.Id}}", self.docker_image],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if found.returncode == 0 and found.stdout.strip().startswith("sha256:"):
                    self.backend, self.runtime = "docker", found.stdout.strip()
                    return True
                reasons.append("Docker 不可访问，或指定 Python 镜像不在本地；未自动拉取镜像。")
            except (OSError, subprocess.TimeoutExpired):
                reasons.append("Docker 探测未完成。")
        if (
            self.requested_backend in {"auto", "macos"}
            and platform.system() == "Darwin"
            and shutil.which("sandbox-exec")
        ):
            profile = '(version 1)(deny default)(allow process-exec)(allow file-read* (literal "/") (subpath "/usr/bin") (subpath "/usr/lib") (subpath "/System"))(allow sysctl-read)'
            try:
                probe = subprocess.run(
                    ["sandbox-exec", "-p", profile, "/usr/bin/true"], capture_output=True, timeout=5
                )
                if probe.returncode == 0:
                    self.backend, self.runtime = "macos", str(Path(sys.executable).resolve())
                    return True
                reasons.append("macOS 系统隔离被当前进程权限拒绝。")
            except (OSError, subprocess.TimeoutExpired):
                reasons.append("macOS 系统隔离探测未完成。")
        self.unavailable_reason = (
            "；".join(reasons)
            or "未发现可用的真实隔离后端。需要本地 Docker Python 镜像或 macOS sandbox-exec。"
        )
        return False

    def _mac_profile(self, directory: Path) -> str:
        python = Path(self.runtime)
        stdlib = Path(sysconfig.get_path("stdlib")).resolve()
        library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
        literals = {
            "/",
            str(python),
            str(library),
            str(library.resolve()),
            "/dev/null",
            "/dev/urandom",
            "/dev/random",
        }
        # Read only libraries linked by this interpreter, not the Homebrew tree.
        linked = subprocess.run(
            ["/usr/bin/otool", "-L", str(python)],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        for line in linked.stdout.splitlines()[1:]:
            path = line.strip().split(" (", 1)[0]
            if path.startswith("/"):
                literals.update((path, str(Path(path).resolve())))
        quote = json.dumps
        allowed_files = " ".join(f"(literal {quote(path)})" for path in sorted(literals))
        metadata_paths = {str(parent) for path in literals for parent in Path(path).parents}
        metadata_paths.update(("/System/Cryptexes/OS", "/System/Cryptexes/Rosetta"))
        metadata = " ".join(f"(literal {quote(path)})" for path in sorted(metadata_paths))
        return f"""(version 1)
(deny default)
(allow process-exec (literal {quote(str(python))}))
(allow sysctl-read)
(allow file-read-metadata {metadata})
(allow file-read* {allowed_files} (subpath {quote(str(stdlib))})
 (subpath "/usr/lib") (subpath "/System/Library")
 (subpath "/System/Volumes/Preboot/Cryptexes/OS/usr/lib")
 (subpath "/System/Volumes/Preboot/Cryptexes/OS/System/Library")
 (subpath {quote(str(directory))}))
(deny file-read* (subpath {quote(str(stdlib / "site-packages"))}))
(allow file-write* (literal "/dev/null")
 (literal {quote(str(directory / "stdout.json"))})
 (literal {quote(str(directory / "stderr.txt"))}))
"""

    def run(
        self, source: str, rows: list[dict[str, Any]], config: dict[str, Any] | None = None
    ) -> ExecutionResult:
        digest = _digest(source)
        try:
            check_source(source)
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("输入必须是 JSON 对象列表。")
            if config is not None and not isinstance(config, dict):
                raise ValueError("适配配置必须是 JSON 对象。")
            request = _json({"rows": rows, "config": config or {}})
            if len(request.encode()) > self.limits.max_input_bytes:
                raise ValueError("输入超过本次隔离验证允许的数据大小。")
        except (ValueError, TypeError, SyntaxError) as exc:
            return ExecutionResult(
                "failed", self.backend, digest, error=str(exc), failure_kind="contract"
            )
        if not self.detect():
            return ExecutionResult(
                "unavailable", "", digest, error=self.unavailable_reason, failure_kind="backend"
            )
        with tempfile.TemporaryDirectory(prefix="tunesmith-transform-") as temp:
            directory = Path(temp).resolve()
            (directory / "request.json").write_text(request)
            (directory / "transform.py").write_text(source)
            worker = Path(__file__).resolve().parents[2] / "scripts" / "sandbox_worker.py"
            shutil.copyfile(worker, directory / "worker.py")
            directory.chmod(0o755)
            for path in directory.iterdir():
                path.chmod(0o644)
            args = [
                str(self.limits.max_output_bytes),
                str(self.limits.cpu_seconds),
                str(self.limits.memory_mb),
            ]
            container = "tunesmith-transform-" + directory.name.rsplit("-", 1)[-1]
            if self.backend == "docker":
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--name",
                    container,
                    "--network=none",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    "--user=65534:65534",
                    "--pids-limit=16",
                    f"--memory={self.limits.memory_mb}m",
                    f"--memory-swap={self.limits.memory_mb}m",
                    "--cpus=1",
                    "--tmpfs=/tmp:rw,noexec,nosuid,size=8m",
                    "--mount",
                    f"type=bind,source={directory},target=/work,readonly",
                    "--workdir=/work",
                    "--entrypoint=python",
                    self.runtime,
                    "-I",
                    "-S",
                    "-B",
                    "/work/worker.py",
                    "/work/request.json",
                    "/work/transform.py",
                    *args,
                ]
            else:
                try:
                    profile = self._mac_profile(directory)
                except (OSError, subprocess.SubprocessError) as exc:
                    return ExecutionResult(
                        "unavailable",
                        self.backend,
                        digest,
                        error=f"无法构建真实隔离配置：{type(exc).__name__}",
                        failure_kind="backend",
                    )
                command = [
                    "sandbox-exec",
                    "-p",
                    profile,
                    self.runtime,
                    "-I",
                    "-S",
                    "-B",
                    str(directory / "worker.py"),
                    str(directory / "request.json"),
                    str(directory / "transform.py"),
                    *args,
                ]
            stdout = directory / "stdout.json"
            stderr = directory / "stderr.txt"
            try:
                with stdout.open("wb") as out, stderr.open("wb") as err:
                    process = subprocess.Popen(
                        command,
                        cwd=directory,
                        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "LANG": "C.UTF-8"},
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                    )
                    deadline = time.monotonic() + self.limits.timeout_seconds
                    while process.poll() is None:
                        oversized = any(
                            path.stat().st_size > self.limits.max_output_bytes
                            for path in (stdout, stderr)
                        )
                        memory_exceeded = False
                        if self.backend == "macos":
                            memory_sample = subprocess.run(
                                ["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                                capture_output=True,
                                text=True,
                                timeout=1,
                            )
                            if memory_sample.returncode == 0 and memory_sample.stdout.strip():
                                memory_exceeded = (
                                    int(memory_sample.stdout.strip()) > self.limits.memory_mb * 1024
                                )
                            elif process.poll() is None:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                                return ExecutionResult(
                                    "failed",
                                    self.backend,
                                    digest,
                                    error="无法监督隔离进程内存，已终止。",
                                    failure_kind="execution",
                                )
                        if oversized or memory_exceeded or time.monotonic() >= deadline:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            reason = (
                                "output_limit"
                                if oversized
                                else "memory_limit"
                                if memory_exceeded
                                else "timeout"
                            )
                            return ExecutionResult(
                                "failed",
                                self.backend,
                                digest,
                                error=f"隔离执行触及 {reason}，进程已终止。",
                                failure_kind=reason,
                            )
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            process.wait(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
                if process.returncode:
                    diagnostic = stderr.read_text(errors="replace")[:500]
                    return ExecutionResult(
                        "failed",
                        self.backend,
                        digest,
                        error=f"隔离进程未正常完成（PID {process.pid}，退出码 {process.returncode}）。{diagnostic}",
                        failure_kind="execution",
                    )
                if stdout.stat().st_size > self.limits.max_output_bytes:
                    raise ValueError("输出超过隔离验证大小限制。")
                result = json.loads(stdout.read_text())
                if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
                    raise ValueError("隔离输出不符合 JSON 响应结构。")
                if not result["ok"]:
                    return ExecutionResult(
                        "failed",
                        self.backend,
                        digest,
                        error=str(result.get("error", "转换失败"))[:500],
                        failure_kind="transform",
                    )
                output = result.get("rows")
                if not isinstance(output, list) or any(not isinstance(row, dict) for row in output):
                    raise ValueError("转换输出必须是 JSON 对象列表。")
                _json(output)
                return ExecutionResult("passed", self.backend, digest, rows=output)
            except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
                if "process" in locals() and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return ExecutionResult(
                    "failed", self.backend, digest, error=str(exc)[:500], failure_kind="execution"
                )
            finally:
                if self.backend == "docker":
                    # Killing the client alone does not terminate its container.
                    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                        subprocess.run(
                            ["docker", "rm", "-f", container], capture_output=True, timeout=5
                        )

    def validate(self, source: str, cases: list[TransformCase]) -> ValidationReport:
        if not cases or {case.kind for case in cases} != {"business", "counterexample"}:
            raise ValueError("冻结适配前必须提供真实业务期望样例和反例。")
        if any(not case.expect_error and case.expected_rows is None for case in cases):
            raise ValueError("每个样例必须明确预期输出或预期转换拒绝。")
        results = []
        for case in cases:
            result = self.run(source, case.rows, case.config)
            passed = (
                (result.status == "failed" and result.failure_kind == "transform")
                if case.expect_error
                else result.status == "passed" and result.rows == case.expected_rows
            )
            results.append(
                {
                    "name": case.name,
                    "kind": case.kind,
                    "passed": passed,
                    "execution_status": result.status,
                    "error": result.error,
                }
            )
        status = (
            "unavailable"
            if any(case["execution_status"] == "unavailable" for case in results)
            else "passed"
            if all(case["passed"] for case in results)
            else "failed"
        )
        limits = {
            **asdict(self.limits),
            "memory_enforcement": "sampled_rss" if self.backend == "macos" else "container_cgroup",
        }
        return ValidationReport(
            status,
            self.backend,
            _digest(source),
            _digest(_json([asdict(case) for case in cases])),
            limits,
            results,
        )
