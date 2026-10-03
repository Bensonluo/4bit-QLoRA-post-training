"""Conversion refusal, atomic publication, and tokenizer contract regressions."""

import json
import subprocess
from pathlib import Path

import pytest

from src.utils.gguf_export import convert_to_gguf, gguf_toolchain, verify_gguf_tokenizer
from tests.unit.test_gguf_metadata import _fields, _write


def _tools(tmp_path, *, quantizer=True):
    root = tmp_path / "tools with spaces"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("# converter")
    (root / "gguf-py" / "gguf").mkdir(parents=True)
    (root / "gguf-py" / "gguf" / "writer.py").write_text("# writer")
    if quantizer:
        binary = root / "build" / "bin" / "llama-quantize"
        binary.parent.mkdir(parents=True)
        binary.write_text("# quantizer")
        binary.chmod(0o755)
    return root


def _source(tmp_path):
    source = tmp_path / "HF with spaces"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"vocab_size": 3}))
    (source / "tokenizer_config.json").write_text("{}")
    return source


def _runtime(monkeypatch, *, failure=None, mutation=None, wrong_type=False, unquantized=False):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == len(calls):
            raise subprocess.CalledProcessError(2, command, stderr="specific external failure")
        if "--outfile" in command:
            output = Path(command[command.index("--outfile") + 1])
            file_type = 1
        else:
            output = Path(command[2])
            file_type = 15 if not wrong_type else 1
        fields = [
            (key, kind, file_type if key == "general.file_type" else value)
            for key, kind, value in _fields()
        ]
        if file_type == 15 and not unquantized:
            _write(output, fields=fields, tensors=[("weight", [256], 12, 0)], payload=b"\0" * 144)
        else:
            _write(output, fields=fields)
        if mutation:
            mutation(len(calls))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("src.utils.gguf_export.subprocess.run", run)
    monkeypatch.setattr("src.utils.gguf_export.verify_gguf_tokenizer", lambda *_: None)
    return calls


def test_two_stages_preserve_paths_and_publish_only_valid_final(tmp_path, monkeypatch):
    source, tools = _source(tmp_path), _tools(tmp_path)
    calls = _runtime(monkeypatch)
    target = tmp_path / "output" / "model.gguf"
    result = convert_to_gguf(source, target, llama_cpp_dir=tools)
    assert len(calls) == 2
    assert calls[0][calls[0].index("--outfile") - 1] == str(source) and calls[0][-1] == "f16"
    assert calls[1][-1] == "Q4_K_M"
    assert result["metadata"]["file_type"] == 15
    assert result["business_evaluation_status"] == "not_run"
    assert target.is_file() and list(target.parent.iterdir()) == [target]
    assert result["toolchain"]["files"]["convert_hf_to_gguf.py"]["sha256"]


def test_f16_needs_no_quantizer(tmp_path, monkeypatch):
    source, tools = _source(tmp_path), _tools(tmp_path, quantizer=False)
    calls = _runtime(monkeypatch)
    result = convert_to_gguf(
        source, tmp_path / "model.gguf", llama_cpp_dir=tools, quantization="f16"
    )
    assert len(calls) == 1 and result["metadata"]["file_type"] == 1


@pytest.mark.parametrize("failure", [1, 2])
def test_each_external_failure_is_visible_and_leaves_no_published_file(
    tmp_path, monkeypatch, failure
):
    source, tools = _source(tmp_path), _tools(tmp_path)
    calls = _runtime(monkeypatch, failure=failure)
    target = tmp_path / "out" / "model.gguf"
    with pytest.raises(ValueError, match="specific external failure"):
        convert_to_gguf(source, target, llama_cpp_dir=tools)
    assert len(calls) == failure and not target.exists()
    assert list(target.parent.iterdir()) == []


def test_wrong_actual_precision_is_not_attested(tmp_path, monkeypatch):
    source, tools = _source(tmp_path), _tools(tmp_path)
    _runtime(monkeypatch, wrong_type=True)
    target = tmp_path / "model.gguf"
    with pytest.raises(ValueError, match="实际精度"):
        convert_to_gguf(source, target, llama_cpp_dir=tools)
    assert not target.exists()


def test_tool_source_mutation_during_conversion_blocks_publication(tmp_path, monkeypatch):
    source, tools = _source(tmp_path), _tools(tmp_path)
    _runtime(
        monkeypatch,
        mutation=lambda n: (tools / "gguf-py" / "gguf" / "writer.py").write_text(str(n)),
    )
    target = tmp_path / "model.gguf"
    with pytest.raises(ValueError, match="工具发生变化"):
        convert_to_gguf(source, target, llama_cpp_dir=tools)
    assert not target.exists()


def test_concurrent_output_is_preserved(tmp_path, monkeypatch):
    source, tools = _source(tmp_path), _tools(tmp_path)
    target = tmp_path / "model.gguf"
    _runtime(monkeypatch, mutation=lambda n: target.write_text("foreign") if n == 2 else None)
    with pytest.raises(ValueError, match="未覆盖"):
        convert_to_gguf(source, target, llama_cpp_dir=tools)
    assert target.read_text() == "foreign"


def test_existing_output_is_never_changed(tmp_path):
    source = _source(tmp_path)
    target = tmp_path / "model.gguf"
    target.write_text("existing")
    with pytest.raises(ValueError, match="不会覆盖"):
        convert_to_gguf(source, target)
    assert target.read_text() == "existing"


def test_tool_absence_and_invalid_precision_do_not_install_anything(tmp_path, monkeypatch):
    monkeypatch.delenv("TUNESMITH_LLAMA_CPP", raising=False)
    with pytest.raises(ValueError, match="setup_gguf_tools"):
        gguf_toolchain(None, "q4_k_m")
    with pytest.raises(ValueError, match="不支持"):
        gguf_toolchain(tmp_path, "arbitrary")


@pytest.fixture
def source_tokenizer(tmp_path, monkeypatch):
    source = _source(tmp_path)

    class Tokenizer:
        bos_token_id = None
        eos_token_id = 2
        pad_token_id = 0
        add_bos_token = False
        add_eos_token = False
        chat_template = None

        def __len__(self):
            return 2

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: Tokenizer())
    metadata = {
        "token_count": 3,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "add_bos_token": False,
        "add_eos_token": False,
        "chat_template": {},
    }
    return source, metadata


def test_padding_vocabulary_and_unused_bos_alias_do_not_change_prompt(source_tokenizer):
    source, metadata = source_tokenizer
    verify_gguf_tokenizer(source, metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_count", 2),
        ("eos_token_id", None),
        ("eos_token_id", 1),
        ("pad_token_id", 2),
        ("add_bos_token", True),
        ("add_eos_token", None),
    ],
)
def test_tokenizer_contract_mismatch_is_refused(source_tokenizer, field, value):
    source, metadata = source_tokenizer
    metadata[field] = value
    with pytest.raises(ValueError):
        verify_gguf_tokenizer(source, metadata)


def test_template_dropped_by_converter_is_refused(source_tokenizer):
    source, metadata = source_tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    type(tokenizer).chat_template = "{{ messages }}"
    with pytest.raises(ValueError, match="聊天模板"):
        verify_gguf_tokenizer(source, metadata)


def test_actual_jinja_precedence_cannot_be_replaced_by_config_template(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    source = _source(tmp_path)
    (source / "config.json").write_text('{"vocab_size":4}')
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"UNK": 0, "BOS": 1, "EOS": 2, "PAD": 3}, unk_token="UNK")
        ),
        unk_token="UNK",
        bos_token="BOS",
        eos_token="EOS",
        pad_token="PAD",
    )
    tokenizer.save_pretrained(source)
    config_file = source / "tokenizer_config.json"
    config = json.loads(config_file.read_text())
    config["chat_template"] = "CONFIG TEMPLATE"
    config_file.write_text(json.dumps(config))
    (source / "chat_template.jinja").write_text("ACTUAL HF TEMPLATE")
    metadata = {
        "token_count": 4,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "add_bos_token": False,
        "add_eos_token": False,
        "chat_template": {"default": "CONFIG TEMPLATE"},
    }
    with pytest.raises(ValueError, match="聊天模板"):
        verify_gguf_tokenizer(source, metadata)
    metadata["chat_template"]["default"] = "ACTUAL HF TEMPLATE"
    verify_gguf_tokenizer(source, metadata)


@pytest.mark.parametrize("name", ["bos_token_id", "pad_token_id"])
def test_missing_source_special_ids_are_not_silently_accepted(source_tokenizer, name):
    from transformers import AutoTokenizer

    source, metadata = source_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(source)
    type(tokenizer).bos_token_id = 1
    metadata[name] = None
    with pytest.raises(ValueError, match="缺少"):
        verify_gguf_tokenizer(source, metadata)


def test_float_tensors_with_quantized_declaration_are_refused(tmp_path, monkeypatch):
    source, tools = _source(tmp_path), _tools(tmp_path)
    _runtime(monkeypatch, unquantized=True)
    target = tmp_path / "model.gguf"
    with pytest.raises(ValueError, match="没有包含所选精度"):
        convert_to_gguf(source, target, llama_cpp_dir=tools)
    assert not target.exists()


@pytest.mark.parametrize("location,name", [("bin", "libggml.dylib"), ("build/lib", "libggml.so.1")])
def test_tool_identity_covers_executed_adjacent_and_versioned_libraries(tmp_path, location, name):
    tools = _tools(tmp_path)
    if location == "bin":
        (tools / "build" / "bin" / "llama-quantize").rename(tools / "llama-quantize")
        (tools / "bin").mkdir()
        (tools / "llama-quantize").rename(tools / "bin" / "llama-quantize")
    library = tools / location / name
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_text("old")
    first = gguf_toolchain(tools, "q4_k_m", verify_content=True)
    library.write_text("new")
    assert first != gguf_toolchain(tools, "q4_k_m", verify_content=True)


def test_windows_release_quantizer_can_be_discovered(tmp_path):
    tools = _tools(tmp_path)
    original = tools / "build" / "bin" / "llama-quantize"
    replacement = original.parent / "Release" / "llama-quantize.exe"
    replacement.parent.mkdir()
    original.rename(replacement)
    assert gguf_toolchain(tools, "q4_k_m")["quantizer"] == str(replacement)


def test_hashed_source_is_executed_even_with_timestamp_valid_stale_pycache(tmp_path, monkeypatch):
    import base64
    import os
    import py_compile

    source, tools = _source(tmp_path), _tools(tmp_path, quantizer=False)
    package = tools / "conversion"
    package.mkdir()
    helper = package / "helper.py"
    helper.write_text("VALUE = 'OLD'\n")
    stat = helper.stat()
    py_compile.compile(str(helper), doraise=True)
    helper.write_text("VALUE = 'NEW'\n")
    os.utime(helper, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    fields = [
        (key, kind, "NEW" if key == "general.architecture" else value)
        for key, kind, value in _fields()
    ]
    blob = _write(tmp_path / "template.gguf", fields=fields)
    (tools / "convert_hf_to_gguf.py").write_text(
        "import sys, base64\nfrom pathlib import Path\nfrom conversion import helper\n"
        f"blob = base64.b64decode({base64.b64encode(blob)!r})\n"
        "output = Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
        "output.write_bytes(blob.replace(b'NEW', helper.VALUE.encode()))\n"
    )
    monkeypatch.setattr("src.utils.gguf_export.verify_gguf_tokenizer", lambda *_: None)
    result = convert_to_gguf(
        source, tmp_path / "model.gguf", llama_cpp_dir=tools, quantization="f16"
    )
    assert result["metadata"]["architecture"] == "NEW"
