"""Tool preparation checks command, ownership and failure contracts without downloads."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import setup_gguf_tools as setup
from src.utils import gguf_runtime_tools as runtime_tools


def checkout(directory):
    (directory / ".git").mkdir(parents=True)
    for name in (
        "convert_hf_to_gguf.py",
        "CMakeLists.txt",
        "conversion/__init__.py",
        "gguf-py/gguf/__init__.py",
        "requirements/requirements-convert_hf_to_gguf.txt",
    ):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("official fixture")


def quantizer(directory):
    path = directory / "build" / "bin" / "llama-quantize"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("binary fixture")
    path.chmod(0o755)
    return path


def runtime_binary(build):
    path = build / "bin" / "tunesmith-gguf-eval"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("static bridge fixture")
    path.chmod(0o755)
    return path


@pytest.fixture
def runner(monkeypatch):
    state = {
        "commands": [],
        "fail": None,
        "head": setup.REVISION,
        "origin": setup.REPOSITORY,
        "dirty": "",
        "create_binary": True,
        "quantizer_code": 1,
        "quantizer_help": "usage: llama-quantize\nQ4_K_M",
        "create_runtime": True,
        "runtime_code": 0,
        "runtime_version": {
            "status": "ready",
            "protocol": runtime_tools.PROTOCOL,
            "revision": setup.REVISION,
        },
        "runtime_flags": dict(runtime_tools.STATIC_CPU_FLAGS),
    }

    def run(arguments, **kwargs):
        assert isinstance(arguments, list)
        assert all(isinstance(argument, str) for argument in arguments)
        expected_kwargs = {"capture_output": True, "text": True, "check": False}
        if arguments[-1] == "--version":
            expected_kwargs["timeout"] = 10
        assert kwargs == expected_kwargs
        state["commands"].append(arguments)
        if state["fail"] and state["fail"](arguments):
            return subprocess.CompletedProcess(arguments, 7, "", "fixture subprocess failure")
        if arguments[0] == "git":
            if arguments[1] == "init":
                (Path(arguments[2]) / ".git").mkdir()
            else:
                directory = Path(arguments[2])
                if not (directory / ".git").is_dir():
                    return subprocess.CompletedProcess(arguments, 128, "", "not a git repository")
                operation = arguments[3:]
                if operation[0] == "checkout":
                    for name in (
                        "convert_hf_to_gguf.py",
                        "CMakeLists.txt",
                        "conversion/__init__.py",
                        "gguf-py/gguf/__init__.py",
                    ):
                        path = directory / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("official fixture")
                elif operation == ["rev-parse", "--show-toplevel"]:
                    return subprocess.CompletedProcess(arguments, 0, str(directory.resolve()), "")
                elif operation == ["rev-parse", "HEAD"]:
                    return subprocess.CompletedProcess(arguments, 0, state["head"], "")
                elif operation == ["remote", "get-url", "origin"]:
                    return subprocess.CompletedProcess(arguments, 0, state["origin"], "")
                elif operation[0] == "status":
                    return subprocess.CompletedProcess(arguments, 0, state["dirty"], "")
        elif arguments[:2] == ["cmake", "-S"] and arguments[2] == str(
            runtime_tools.BRIDGE_SOURCE_DIR
        ):
            build = Path(arguments[arguments.index("-B") + 1])
            build.mkdir(parents=True, exist_ok=True)
            (build / "CMakeCache.txt").write_text(
                "\n".join(f"{key}:BOOL={value}" for key, value in state["runtime_flags"].items())
                + f"\nCMAKE_HOME_DIRECTORY:INTERNAL={runtime_tools.BRIDGE_SOURCE_DIR}\n"
            )
        elif arguments[:2] == ["cmake", "--build"]:
            if "tunesmith-gguf-eval" in arguments:
                if state["create_runtime"]:
                    runtime_binary(Path(arguments[2]))
            elif state["create_binary"]:
                quantizer(Path(arguments[2]).parent)
        elif arguments[0].endswith("llama-quantize"):
            return subprocess.CompletedProcess(
                arguments, state["quantizer_code"], state["quantizer_help"], ""
            )
        elif arguments[-1] == "--version":
            return subprocess.CompletedProcess(
                arguments, state["runtime_code"], json.dumps(state["runtime_version"]), ""
            )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(setup.subprocess, "run", run)
    return state


def test_fresh_preparation_pins_official_revision_and_prints_only_json_stdout(
    tmp_path, runner, capsys
):
    destination = tmp_path / "tool cache with spaces"
    assert setup.main(["--destination", str(destination), "--jobs", "2"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "ready"
    assert result["revision"] == setup.REVISION
    assert result["llama_cpp_dir"] == str(destination.resolve())
    assert result["python_executable"] == sys.executable
    assert result["reused_checkout"] is False
    bridge = destination / "build-tunesmith-eval" / "bin" / "tunesmith-gguf-eval"
    assert result["evaluation_runtime_path"] == str(bridge.resolve())
    assert result["gguf_runtime_path"] == result["evaluation_runtime_path"]
    assert result["evaluation_runtime_identity"] == runtime_tools.evaluation_runtime_identity(
        bridge
    )
    assert Path(str(bridge) + ".json").is_file()
    assert shlex.split(result["train_export_options"]) == result["train_export_args"]
    commands = runner["commands"]
    fetch = next(command for command in commands if "fetch" in command)
    assert fetch[-4:] == ["--depth", "1", "origin", setup.REVISION]
    assert any(command[-3:] == ["add", "origin", setup.REPOSITORY] for command in commands)
    assert [sys.executable, "-B", str(destination / "convert_hf_to_gguf.py"), "--help"] in commands
    build = next(command for command in commands if command[:2] == ["cmake", "--build"])
    assert build[-4:] == ["--target", "llama-quantize", "--parallel", "2"]
    bridge_build = next(command for command in commands if "tunesmith-gguf-eval" in command)
    assert "--clean-first" in bridge_build
    assert bridge_build[-4:] == ["--target", "tunesmith-gguf-eval", "--parallel", "2"]
    assert "编译 llama-quantize" in captured.err
    assert not any("pip" in command or "llama-cli" in command for command in commands)


def test_repeat_reuses_clean_checkout_without_fetching_or_switching_revision(tmp_path, runner):
    destination = tmp_path / "tools"
    setup.prepare(destination, 1)
    runner["commands"].clear()
    result = setup.prepare(destination, 1)
    assert result["reused_checkout"] is True
    assert not any(
        "fetch" in command or "checkout" in command or "init" in command
        for command in runner["commands"]
    )
    assert not any("tunesmith-gguf-eval" in command for command in runner["commands"])


@pytest.mark.parametrize(
    "conflict", ["plain_directory", "file", "symlink", "dirty", "wrong_head", "wrong_origin"]
)
def test_existing_foreign_or_modified_destination_is_never_overwritten(tmp_path, runner, conflict):
    destination = tmp_path / "tools"
    if conflict == "file":
        destination.write_text("user file")
    elif conflict == "symlink":
        target = tmp_path / "user tools"
        checkout(target)
        destination.symlink_to(target, target_is_directory=True)
    elif conflict == "plain_directory":
        destination.mkdir()
        (destination / "user.txt").write_text("user contents")
    else:
        checkout(destination)
        runner["dirty"] = " M convert_hf_to_gguf.py" if conflict == "dirty" else ""
        runner["head"] = "a" * 40 if conflict == "wrong_head" else setup.REVISION
        runner["origin"] = (
            "https://example.invalid/foreign" if conflict == "wrong_origin" else setup.REPOSITORY
        )
    with pytest.raises(ValueError):
        setup.prepare(destination, 1)
    assert destination.exists()
    assert not any(
        command[0] == "cmake" or "fetch" in command or "checkout" in command
        for command in runner["commands"]
    )
    if conflict == "plain_directory":
        assert (destination / "user.txt").read_text() == "user contents"
    elif conflict == "file":
        assert destination.read_text() == "user file"


@pytest.mark.parametrize("phase", ["fetch", "converter", "configure", "build"])
def test_process_failure_never_emits_success_and_leaves_no_partial_clone(
    tmp_path, runner, capsys, phase
):
    destination = tmp_path / "tools"
    predicates = {
        "fetch": lambda command: "fetch" in command,
        "converter": lambda command: command[0] == sys.executable,
        "configure": lambda command: command[:2] == ["cmake", "-S"],
        "build": lambda command: command[:2] == ["cmake", "--build"],
    }
    runner["fail"] = predicates[phase]
    assert setup.main(["--destination", str(destination)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fixture subprocess failure" in captured.err
    assert not list(tmp_path.glob(".tunesmith-llama-*"))
    if phase == "fetch":
        assert not destination.exists()
    elif phase == "converter":
        assert "不会自动 pip 安装或修改项目环境" in captured.err
        assert not any(command[0] == "cmake" for command in runner["commands"])


@pytest.mark.parametrize("failure", ["missing_binary", "bad_exit", "bad_help"])
def test_build_directory_or_failed_binary_cannot_masquerade_as_ready(
    tmp_path, runner, capsys, failure
):
    runner["create_binary"] = failure != "missing_binary"
    runner["quantizer_code"] = 9 if failure == "bad_exit" else 1
    runner["quantizer_help"] = (
        "usage: unrelated program" if failure == "bad_help" else "usage: llama-quantize\nQ4_K_M"
    )
    assert setup.main(["--destination", str(tmp_path / "tools")]) == 2
    assert capsys.readouterr().out == ""


def test_check_only_does_not_create_missing_directory_or_build_existing_tools(tmp_path, runner):
    destination = tmp_path / "tools"
    with pytest.raises(ValueError, match="不会下载或创建"):
        setup.prepare(destination, 1, check_only=True)
    assert not destination.exists()
    assert runner["commands"] == []
    checkout(destination)
    quantizer(destination)
    with pytest.raises(ValueError, match="缺少可执行"):
        setup.prepare(destination, 1, check_only=True)
    setup.prepare(destination, 1)
    runner["commands"].clear()
    result = setup.prepare(destination, 1, check_only=True)
    assert result["status"] == "ready" and result["check_only"] is True
    assert not any(command[0] == "cmake" or "fetch" in command for command in runner["commands"])


@pytest.mark.parametrize("phase", ["configure", "build", "missing_binary", "version", "dynamic"])
def test_evaluation_bridge_failure_never_claims_tools_ready(tmp_path, runner, capsys, phase):
    if phase == "configure":
        runner["fail"] = lambda command: (
            command[:2] == ["cmake", "-S"] and command[2] == str(runtime_tools.BRIDGE_SOURCE_DIR)
        )
    elif phase == "build":
        runner["fail"] = lambda command: "tunesmith-gguf-eval" in command
    elif phase == "missing_binary":
        runner["create_runtime"] = False
    elif phase == "version":
        runner["runtime_version"]["protocol"] = "foreign_protocol"
    else:
        runner["runtime_flags"]["BUILD_SHARED_LIBS"] = "ON"
    assert setup.main(["--destination", str(tmp_path / "tools")]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("changed", ["source", "binary", "manifest"])
def test_check_only_refuses_stale_bridge_and_prepare_rebuilds_it(
    tmp_path, runner, monkeypatch, changed
):
    source = tmp_path / "bridge-source"
    source.mkdir()
    for name in ("main.cpp", "CMakeLists.txt"):
        (source / name).write_text("original source")
    monkeypatch.setattr(runtime_tools, "BRIDGE_SOURCE_DIR", source)
    destination = tmp_path / "tools"
    prepared = setup.prepare(destination, 1)
    binary = Path(prepared["evaluation_runtime_path"])
    if changed == "source":
        implementation = source / "main.cpp"
        before = implementation.stat()
        implementation.write_text("modified source")
        os.utime(implementation, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif changed == "binary":
        binary.write_text("modified executable")
    else:
        Path(str(binary) + ".json").unlink()
    runner["commands"].clear()
    with pytest.raises(ValueError):
        setup.prepare(destination, 1, check_only=True)
    assert not any(command[0] == "cmake" for command in runner["commands"])
    result = setup.prepare(destination, 1)
    assert result["status"] == "ready"
    build = next(
        command
        for command in runner["commands"]
        if command[:2] == ["cmake", "--build"] and "tunesmith-gguf-eval" in command
    )
    assert "--clean-first" in build


def test_missing_system_tool_has_actionable_error_without_success(tmp_path, monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git is not installed")

    monkeypatch.setattr(setup.subprocess, "run", missing)
    destination = tmp_path / "tools"
    assert setup.main(["--destination", str(destination)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Git、CMake 和 C/C++ 编译器" in captured.err
    assert not destination.exists()


def test_incomplete_checkout_is_refused_before_python_or_build(tmp_path, runner):
    destination = tmp_path / "tools"
    checkout(destination)
    (destination / "conversion" / "__init__.py").unlink()
    with pytest.raises(ValueError, match="checkout 不完整"):
        setup.prepare(destination, 1)
    assert not any(command[0] in {"cmake", sys.executable} for command in runner["commands"])


def test_default_cache_and_job_argument_are_bounded_and_validated(tmp_path, runner, monkeypatch):
    monkeypatch.setattr(setup.sys, "platform", "darwin")
    monkeypatch.setattr(setup.Path, "home", lambda: tmp_path)
    assert (
        setup.default_destination()
        == tmp_path / "Library" / "Caches" / "tunesmith" / "llama.cpp" / setup.REVISION[:12]
    )
    with pytest.raises(SystemExit) as error:
        setup.main(["--jobs", "0"])
    assert error.value.code == 2
    assert runner["commands"] == []
