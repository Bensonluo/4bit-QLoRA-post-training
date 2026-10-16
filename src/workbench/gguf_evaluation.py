"""Evaluate an existing GGUF export locally, without a server or chat template."""

from __future__ import annotations

import json
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any

from src.workbench.business_evaluation import (
    EvaluationModel,
    EvaluationProtocol,
    Generation,
    _model_identity,
    generation_token_proof,
    verify_generation_token_proof,
)
from src.workbench.model_export import _files, plan_model_export
from src.workbench.training_runs import TrainingRunService, file_digest


def _export_evidence(directory: Path) -> dict:
    try:
        evidence = json.loads((directory / "export_evidence.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("找不到有效 GGUF 导出证据；请先显式导出到独立目录。") from exc
    if (
        not isinstance(evidence, dict)
        or evidence.get("export_format") != "gguf"
        or evidence.get("tokenizer_directory") != "hf_tokenizer"
    ):
        raise ValueError("需要携带 hf_tokenizer 的 GGUF 导出；旧导出请重新导出到新目录。")
    return evidence


def gguf_evaluation_model(
    run: dict, export_dir: str | Path, runtime_path: str | Path
) -> EvaluationModel:
    """Bind a verified, already exported artifact to its actual training run."""
    directory = Path(export_dir).expanduser().resolve()
    evidence = _export_evidence(directory)
    if any(
        run.get(key) != evidence.get(key)
        for key in ("run_id", "session_id", "dataset_version", "config_digest")
    ):
        raise ValueError("GGUF 导出不属于当前训练运行。")
    model = EvaluationModel(
        "导出 GGUF",
        str(directory / "hf_tokenizer"),
        gguf_path=str(directory / f"model-{evidence.get('quantization')}.gguf"),
        gguf_runtime_path=str(Path(runtime_path).expanduser().resolve()),
    )
    verified_gguf_model(model)
    return model


def verified_gguf_model(model: EvaluationModel) -> dict[str, Any]:
    """Verify bytes, training lineage, tokenizer and native runner as one identity."""
    from src.utils.gguf_runtime_tools import evaluation_runtime_identity

    if not model.gguf_path or not model.gguf_runtime_path or model.adapter_path or model.revision:
        raise ValueError("GGUF 必须绑定独立运行器，不能再叠加 adapter 或 HF revision。")
    supplied = Path(model.gguf_path).expanduser()
    if supplied.is_symlink():
        raise ValueError("GGUF 评测不接受符号链接模型。")
    gguf = supplied.resolve()
    directory = gguf.parent
    evidence = _export_evidence(directory)
    if gguf.name != f"model-{evidence.get('quantization')}.gguf":
        raise ValueError("请求的 GGUF 文件与导出证据不一致。")
    tokenizer = directory / "hf_tokenizer"
    if Path(model.base_model).expanduser().resolve() != tokenizer or tokenizer.is_symlink():
        raise ValueError("GGUF 必须使用同一导出内已验证的 hf_tokenizer。")
    adapter = Path(evidence.get("adapter_dir") or "").expanduser().resolve()
    run_root = adapter.parent.parent
    if not run_root.is_dir() or not (adapter.parent / "run.json").is_file():
        raise ValueError("本地训练来源不可核验；不能把独立 GGUF 冒充工作台训练产物。")
    run = TrainingRunService(run_root).get_status(evidence.get("run_id") or "")
    plan = plan_model_export(
        run,
        directory,
        export_format="gguf",
        quantization=str(evidence.get("quantization")),
        _verify_content=True,
    )
    if plan["status"] != "already_exported":
        raise ValueError("GGUF 来源或内容验证失败：" + "；".join(plan["reasons"]))
    files = _files(tokenizer)
    merged = evidence.get("merged_source_files")
    if (
        not files
        or not isinstance(merged, dict)
        or any(merged.get(name) != entry for name, entry in files.items())
        or "tokenizer_config.json" not in files
    ):
        raise ValueError("携带的 tokenizer 与实际转换的 HF tokenizer 文件不同或不完整。")
    source_base = Path(evidence["base_model_path"])
    return {
        "format": "gguf",
        "gguf": {
            "directory": str(directory),
            "file": str(gguf),
            "sha256": file_digest(gguf),
            "export_evidence_sha256": file_digest(directory / "export_evidence.json"),
            "quantization": evidence["quantization"],
            "metadata": evidence["conversion"]["metadata"],
        },
        "tokenizer": {"directory": str(tokenizer), "files": files},
        "runtime": evaluation_runtime_identity(Path(model.gguf_runtime_path)),
        "source_training": {
            **{
                key: evidence[key]
                for key in ("run_id", "session_id", "dataset_version", "config_digest")
            },
            "requested_model": {"base_model": str(source_base), "adapter_path": str(adapter)},
            "identity": {"base": _model_identity(source_base), "adapter": _model_identity(adapter)},
        },
    }


class GGUFLocalEvaluationRuntime:
    """Own one CPU child process; each question gets a fresh native context."""

    timeout_seconds = 300
    max_line_bytes = 32 * 1024 * 1024

    def __init__(self, model: EvaluationModel, *, identity: dict | None = None):
        self.process = None
        self.reader = None
        self.writer: Thread | None = None
        self.resources = ExitStack()
        # Lifetime spans calls to generate; close() owns the ExitStack.
        self.stderr = self.resources.enter_context(tempfile.TemporaryFile())  # noqa: SIM115
        self.messages: Queue = Queue(maxsize=2)
        try:
            from transformers import AutoTokenizer

            self.identity = identity if identity is not None else verified_gguf_model(model)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model.base_model, local_files_only=True, trust_remote_code=False
            )
            self.eos = self.tokenizer.eos_token_id
            if type(self.eos) is not int or self.eos < 0:
                raise ValueError("当前 GGUF 同题协议要求源 tokenizer 有一个明确的 EOS token。")
            self.process = subprocess.Popen(
                [
                    str(self.identity["runtime"]["binary_path"]),
                    "--model",
                    str(model.gguf_path),
                    "--eos-token-id",
                    str(self.eos),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
            )
            self.reader = Thread(target=self._read_lines, daemon=True)
            self.reader.start()
            ready = self._receive()
            context = ready.get("model_context_length")
            if ready.get("status") != "ready" or type(context) is not int or context <= 0:
                raise ValueError("GGUF 运行器未返回有效模型就绪消息。")
            self.context = context
        except Exception:
            self.close()
            raise

    def _read_lines(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while True:
            line = self.process.stdout.readline(self.max_line_bytes + 1)
            message: Any = line
            if len(line) > self.max_line_bytes or (line and not line.endswith(b"\n")):
                message = ValueError("GGUF 运行器返回超长或不完整 JSON 行。")
            try:
                self.messages.put(message, timeout=1)
            except Full:
                if self.process.poll() is None:
                    self.process.terminate()
                return
            if not line or isinstance(message, Exception):
                return

    def _receive(self) -> dict:
        try:
            line = self.messages.get(timeout=self.timeout_seconds)
        except Empty as exc:
            self.close()
            raise ValueError("GGUF 运行器响应超时，已释放本次进程；未把部分输出作为答案。") from exc
        if isinstance(line, Exception):
            raise line
        if not line:
            raise ValueError("GGUF 运行器提前退出；未获得完整生成证据。")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("GGUF 运行器没有返回 JSON 对象。")
        if value.get("status") == "error":
            raise ValueError(f"GGUF 生成失败：{value.get('error')}")
        return value

    def _send(self, request: bytes) -> None:
        assert self.process is not None and self.process.stdin is not None
        pipe = self.process.stdin
        results: Queue = Queue(maxsize=1)

        def write() -> None:
            try:
                pipe.write(request)
                pipe.flush()
                results.put(None)
            except Exception as exc:
                results.put(exc)

        self.writer = Thread(target=write, daemon=True)
        self.writer.start()
        try:
            error = results.get(timeout=self.timeout_seconds)
        except Empty as exc:
            self.close()
            raise ValueError("GGUF 运行器输入超时，已释放本次进程。") from exc
        self.writer.join(timeout=2)
        if error is not None:
            raise ValueError(f"GGUF 运行器无法接收完整请求：{error}")

    def generate(self, prompt: str, protocol: EvaluationProtocol) -> Generation:
        if self.process is None or self.process.poll() is not None:
            raise ValueError("GGUF 运行器已关闭，不能继续生成。")
        protocol.validate()
        prompt_ids = self.tokenizer(prompt, truncation=False)["input_ids"]
        if not prompt_ids or len(prompt_ids) + protocol.max_new_tokens > self.context:
            raise ValueError("样本与生成长度超过模型上下文；没有静默截断业务输入。")
        request = (
            json.dumps(
                {
                    "prompt": prompt,
                    "max_new_tokens": protocol.max_new_tokens,
                    "expected_prompt_ids": prompt_ids,
                },
                ensure_ascii=False,
            ).encode()
            + b"\n"
        )
        if len(request) > 16 * 1024 * 1024:
            raise ValueError("GGUF 评测请求超过本地运行器的有界输入限制。")
        self._send(request)
        response = self._receive()
        tokens = response.get("generated_token_ids")
        vocabulary_size = self.identity["gguf"]["metadata"]["token_count"]
        if (
            response.get("status") != "completed"
            or response.get("prompt_token_ids") != prompt_ids
            or not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or not 0 <= token < vocabulary_size for token in tokens)
        ):
            raise ValueError("GGUF 返回的输入或生成 token 证据无效。")
        # Same canonical decoder as HF evaluation. Native token pieces can differ
        # for special tokens or partial UTF-8, so they are not business answers.
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if not isinstance(text, str):
            raise ValueError("HF tokenizer 未返回有效解码文本。")
        proof = generation_token_proof(prompt, text, prompt_ids, tokens, self.eos)
        if response.get("finish_reason") != proof["finish_reason"]:
            raise ValueError("GGUF 运行器的停止原因与实际 EOS token 不一致。")
        generation = Generation(text, proof["finish_reason"] == "length", len(tokens), proof)
        verify_generation_token_proof(
            {
                "prompt": prompt,
                "output": text,
                "generated_tokens": len(tokens),
                "truncated": generation.truncated,
                "token_proof": proof,
            },
            {"max_new_tokens": protocol.max_new_tokens},
        )
        return generation

    def close(self) -> None:
        process = self.process
        if process is not None:
            try:
                # Terminate only our child. No local inference server is started.
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                if self.writer is not None:
                    self.writer.join(timeout=2)
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        pipe.close()
                if self.reader is not None:
                    self.reader.join(timeout=2)
                self.process = None
        self.resources.close()


def verify_gguf_report_identity(outcome: Any) -> None:
    """Check saved GGUF identity bindings without reading model files or receipts."""
    import os
    import re

    from src.workbench.sources import content_digest

    error = "GGUF 评测请求与记录的产物、工具或训练来源身份不一致。"

    def require(condition: Any) -> None:
        if not condition:
            raise ValueError(error)

    def sha256(value: Any) -> bool:
        return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None

    def path(value: Any) -> str:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(error)
        # resolve() would consult the current filesystem, changing a historical
        # report's meaning if a path moved. Compare declared paths lexically.
        return os.path.normcase(os.path.abspath(os.path.expanduser(value)))

    require(isinstance(outcome, dict))
    requested, identity = outcome.get("requested_model"), outcome.get("identity")
    gguf_requested = isinstance(requested, dict) and (
        requested.get("gguf_path") or requested.get("gguf_runtime_path")
    )
    gguf_identity = isinstance(identity, dict) and (
        "gguf" in identity or identity.get("format") == "gguf"
    )
    if not (gguf_requested or gguf_identity):
        return
    rows = outcome.get("rows")
    if (
        identity == {}
        and isinstance(rows, list)
        and rows
        and all(isinstance(row, dict) and "output" in row and row["output"] is None for row in rows)
    ):
        return
    require(isinstance(requested, dict) and isinstance(identity, dict))
    require(identity.get("format") == "gguf")
    require(not requested.get("adapter_path") and not requested.get("revision"))
    require("base" not in identity and "adapter" not in identity)
    gguf, runtime, tokenizer = (
        identity.get("gguf"),
        identity.get("runtime"),
        identity.get("tokenizer"),
    )
    require(all(isinstance(value, dict) for value in (gguf, runtime, tokenizer)))
    require(path(requested.get("gguf_path")) == path(gguf.get("file")))
    require(path(requested.get("gguf_runtime_path")) == path(runtime.get("binary_path")))
    require(path(requested.get("base_model")) == path(tokenizer.get("directory")))
    tokenizer_files = tokenizer.get("files")
    require(isinstance(tokenizer_files, dict) and tokenizer_files)
    require(
        all(
            isinstance(name, str)
            and name
            and isinstance(entry, dict)
            and type(entry.get("size")) is int
            and entry["size"] >= 0
            and sha256(entry.get("sha256"))
            for name, entry in tokenizer_files.items()
        )
    )
    require(sha256(gguf.get("sha256")) and sha256(gguf.get("export_evidence_sha256")))
    require(sha256(runtime.get("sha256")))
    source = identity.get("source_training")
    require(isinstance(source, dict))
    require(
        all(
            isinstance(source.get(key), str) and source[key].strip()
            for key in ("run_id", "session_id", "dataset_version", "config_digest")
        )
    )
    require(sha256(source["config_digest"]))
    source_request, source_identity = source.get("requested_model"), source.get("identity")
    require(isinstance(source_request, dict) and isinstance(source_identity, dict))
    require(not source_request.get("gguf_path") and not source_request.get("gguf_runtime_path"))
    require("gguf" not in source_identity and source_identity.get("format") != "gguf")
    for key, request_key in (("base", "base_model"), ("adapter", "adapter_path")):
        recorded = source_identity.get(key)
        require(isinstance(recorded, dict))
        require(path(recorded.get("directory")) == path(source_request.get(request_key)))
        require(sha256(recorded.get("content_digest")))
        files = recorded.get("files")
        require(isinstance(files, dict) and files)
        require(
            all(isinstance(name, str) and name and sha256(digest) for name, digest in files.items())
        )
        require(recorded["content_digest"] == content_digest(files))
