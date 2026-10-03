"""GGUF handoff structure checks are distinct from consumer loading evidence."""

import struct
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from src.utils.gguf_metadata import read_gguf_metadata

_CODES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
Field = tuple[str, int, Any]
Tensor = tuple[str, list[int], int, int]


def _value(kind: int, value: Any, endian: str) -> bytes:
    if kind in _CODES:
        return struct.pack(endian + _CODES[kind], value)
    if kind == 8:
        data = str(value).encode("utf-8")
        return struct.pack(endian + "Q", len(data)) + data
    if kind == 9:
        subtype, items = value
        return struct.pack(endian + "IQ", subtype, len(items)) + b"".join(
            _value(subtype, item, endian) for item in items
        )
    return bytes(value)


def _fields() -> list[Field]:
    return [
        ("general.architecture", 8, "llama"),
        ("general.file_type", 4, 1),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.tokens", 9, (8, ["unknown", "BOS", "EOS"])),
        ("tokenizer.ggml.bos_token_id", 4, 1),
        ("tokenizer.ggml.eos_token_id", 4, 2),
        ("tokenizer.ggml.padding_token_id", 4, 0),
        ("tokenizer.ggml.add_bos_token", 7, 1),
        ("tokenizer.ggml.add_eos_token", 7, 0),
        ("tokenizer.chat_template", 8, "{{ messages[0]['content'] }}"),
    ]


def _write(
    path: Path,
    *,
    fields: list[Field] | None = None,
    tensors: list[Tensor] | None = None,
    payload: bytes = b"\x00" * 6,
    version: int = 3,
    endian: str = "<",
    alignment: int = 32,
) -> bytes:
    fields = _fields() if fields is None else fields
    tensors = [("token.weight", [3], 1, 0)] if tensors is None else tensors
    blob = b"GGUF" + struct.pack(endian + "IQQ", version, len(tensors), len(fields))
    for key, kind, value in fields:
        blob += (
            _value(8, key, endian) + struct.pack(endian + "I", kind) + _value(kind, value, endian)
        )
    for name, dimensions, kind, offset in tensors:
        blob += _value(8, name, endian) + struct.pack(endian + "I", len(dimensions))
        blob += b"".join(struct.pack(endian + "Q", dimension) for dimension in dimensions)
        blob += struct.pack(endian + "IQ", kind, offset)
    blob += b"\x00" * ((-len(blob)) % alignment) + payload
    path.write_bytes(blob)
    return blob


@pytest.mark.parametrize("version,endian", [(2, "<"), (3, "<"), (2, ">"), (3, ">")])
def test_reads_complete_metadata_and_special_tokens(
    tmp_path: Path, version: int, endian: str
) -> None:
    path = tmp_path / "model.gguf"
    _write(path, version=version, endian=endian)
    metadata = read_gguf_metadata(path)
    assert metadata["version"] == version
    assert metadata["endian"] == ("little" if endian == "<" else "big")
    assert metadata["tensor_count"] == 1 and metadata["tensor_types"] == [1]
    assert metadata["architecture"] == "llama" and metadata["file_type"] == 1
    assert metadata["tokenizer_model"] == "gpt2" and metadata["token_count"] == 3
    assert metadata["bos_token_id"] == 1 and metadata["eos_token_id"] == 2
    assert metadata["pad_token_id"] == 0
    assert metadata["add_bos_token"] is True and metadata["add_eos_token"] is False
    assert metadata["chat_template"] == {"default": "{{ messages[0]['content'] }}"}
    assert "实际可加载性与业务效果尚需独立验证" in metadata["validation_scope"]


def test_reads_default_and_named_chat_templates(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write(
        path,
        fields=[
            *_fields(),
            ("tokenizer.chat_templates", 9, (8, ["tool_use"])),
            ("tokenizer.chat_template.tool_use", 8, "{{ tools }}"),
        ],
    )
    templates = read_gguf_metadata(path)["chat_template"]
    assert templates == {"default": "{{ messages[0]['content'] }}", "tool_use": "{{ tools }}"}


def test_optional_template_and_special_token_fields_remain_explicit(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write(path, fields=_fields()[:4])
    metadata = read_gguf_metadata(path)
    assert metadata["chat_template"] == {}
    assert all(
        metadata[key] is None
        for key in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "add_bos_token",
            "add_eos_token",
        )
    )


def test_large_vocabulary_is_skipped_without_materialization(tmp_path: Path) -> None:
    path = tmp_path / "large-vocab.gguf"
    fields = _fields()
    fields[3] = ("tokenizer.ggml.tokens", 9, (8, ["x" * 64] * 50_000))
    _write(path, fields=fields)
    tracemalloc.start()
    try:
        metadata = read_gguf_metadata(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert metadata["token_count"] == 50_000
    assert "tokens" not in metadata
    assert peak < 512 * 1024, (
        "vocabulary strings must not be retained or loaded as one large buffer"
    )


def test_rejects_every_truncated_prefix(tmp_path: Path) -> None:
    path = tmp_path / "truncated.gguf"
    complete = _write(path)
    for length in range(len(complete)):
        path.write_bytes(complete[:length])
        with pytest.raises(ValueError):
            read_gguf_metadata(path)


@pytest.mark.parametrize(
    "field_index,replacement",
    [
        (0, ("general.architecture", 8, "")),
        (0, ("general.architecture", 4, 1)),
        (1, ("general.file_type", 8, "1")),
        (1, ("general.file_type", 5, -1)),
        (2, ("tokenizer.ggml.model", 8, "")),
        (3, ("tokenizer.ggml.tokens", 9, (8, []))),
        (3, ("tokenizer.ggml.tokens", 9, (4, [1, 2, 3]))),
        (4, ("tokenizer.ggml.bos_token_id", 4, 3)),
        (4, ("tokenizer.ggml.bos_token_id", 5, -1)),
        (4, ("tokenizer.ggml.bos_token_id", 7, 1)),
        (7, ("tokenizer.ggml.add_bos_token", 7, 2)),
    ],
)
def test_rejects_invalid_critical_metadata(
    tmp_path: Path, field_index: int, replacement: Field
) -> None:
    path = tmp_path / "invalid.gguf"
    fields = _fields()
    fields[field_index] = replacement
    _write(path, fields=fields)
    with pytest.raises(ValueError):
        read_gguf_metadata(path)


@pytest.mark.parametrize(
    "key",
    ["general.architecture", "general.file_type", "tokenizer.ggml.model", "tokenizer.ggml.tokens"],
)
def test_rejects_missing_weight_model_or_tokenizer_metadata(tmp_path: Path, key: str) -> None:
    path = tmp_path / "missing.gguf"
    _write(path, fields=[field for field in _fields() if field[0] != key])
    with pytest.raises(ValueError):
        read_gguf_metadata(path)


@pytest.mark.parametrize(
    "extra",
    [
        [("general.architecture", 8, "qwen")],
        [("custom", 99, b"")],
        [("tokenizer.chat_templates", 9, (8, ["tool_use"]))],
        [("tokenizer.chat_template.tool_use", 8, "{{ tools }}")],
        [
            ("tokenizer.chat_templates", 9, (8, ["tool_use", "tool_use"])),
            ("tokenizer.chat_template.tool_use", 8, "{{ tools }}"),
        ],
        [
            ("tokenizer.chat_templates", 9, (8, ["default"])),
            ("tokenizer.chat_template.default", 8, "duplicate"),
        ],
        [("general.alignment", 4, 0)],
        [("general.alignment", 4, 3)],
    ],
)
def test_rejects_duplicate_unknown_or_inconsistent_metadata(
    tmp_path: Path, extra: list[Field]
) -> None:
    path = tmp_path / "invalid.gguf"
    _write(path, fields=[*_fields(), *extra])
    with pytest.raises(ValueError):
        read_gguf_metadata(path)


@pytest.mark.parametrize(
    "tensor",
    [
        ("", [3], 1, 0),
        ("bad", [], 1, 0),
        ("bad", [1] * 5, 1, 0),
        ("bad", [0], 1, 0),
        ("bad", [1 << 63, 1 << 63], 1, 0),
        ("bad", [3], 999, 0),
        ("bad", [3], 1, 1),
        ("bad", [3], 1, (1 << 64) - 32),
    ],
)
def test_rejects_invalid_tensor_table_or_data_bounds(tmp_path: Path, tensor: Tensor) -> None:
    path = tmp_path / "invalid.gguf"
    _write(path, tensors=[tensor])
    with pytest.raises(ValueError):
        read_gguf_metadata(path)


def test_rejects_missing_weights_and_duplicate_or_overlapping_tensors(tmp_path: Path) -> None:
    path = tmp_path / "invalid.gguf"
    for tensors, payload in [
        ([], b""),
        ([("weight", [3], 1, 0)], b""),
        ([("weight", [3], 1, 0)] * 2, b"\x00" * 38),
        ([("weight", [3], 1, 0), ("another", [3], 1, 0)], b"\x00" * 38),
    ]:
        _write(path, tensors=tensors, payload=payload)
        with pytest.raises(ValueError):
            read_gguf_metadata(path)


def test_reads_separate_aligned_tensors_and_quantized_structure(tmp_path: Path) -> None:
    path = tmp_path / "quantized.gguf"
    fields = _fields()
    fields[1] = ("general.file_type", 4, 15)
    _write(
        path,
        fields=fields,
        tensors=[("norm", [3], 1, 0), ("weight", [256], 12, 32)],
        payload=b"\x00" * 176,
    )
    metadata = read_gguf_metadata(path)
    assert metadata["tensor_count"] == 2
    assert metadata["tensor_types"] == [1, 12]
    assert metadata["file_type"] == 15


@pytest.mark.parametrize(
    "position,code,value", [(4, "I", 1), (8, "Q", 100_001), (16, "Q", 100_001)]
)
def test_header_counts_and_versions_are_bounded(
    tmp_path: Path, position: int, code: str, value: int
) -> None:
    path = tmp_path / "invalid.gguf"
    blob = bytearray(_write(path))
    struct.pack_into("<" + code, blob, position, value)
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        read_gguf_metadata(path)


def test_rejects_huge_string_and_array_lengths_before_allocation_or_seek(tmp_path: Path) -> None:
    path = tmp_path / "invalid.gguf"
    complete = _write(path)
    for marker in (b"general.architecture", b"unknown"):
        blob = bytearray(complete)
        position = complete.index(marker) - 8
        struct.pack_into("<Q", blob, position, (1 << 64) - 1)
        path.write_bytes(blob)
        with pytest.raises(ValueError, match="长度.*限制"):
            read_gguf_metadata(path)
    blob = bytearray(complete)
    # Token count immediately precedes the first token's string length.
    struct.pack_into("<Q", blob, complete.index(b"unknown") - 16, 2_000_001)
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="数组数量.*限制"):
        read_gguf_metadata(path)


def test_unknown_fields_are_skipped_but_nested_array_depth_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "metadata.gguf"
    _write(
        path,
        fields=[*_fields(), ("extra.scores", 9, (6, [1.0, 2.0])), ("extra.note", 8, "ignored")],
    )
    assert read_gguf_metadata(path)["token_count"] == 3
    nested: Any = (8, ["nested"])
    for _ in range(5):
        nested = (9, [nested])
    _write(path, fields=[*_fields(), ("extra.nested", 9, nested)])
    with pytest.raises(ValueError, match="嵌套数组.*限制"):
        read_gguf_metadata(path)


def test_invalid_utf8_names_and_nonfiles_fail_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "invalid.gguf"
    blob = bytearray(_write(path))
    blob[blob.index(b"general.architecture")] = 255
    path.write_bytes(blob)
    with pytest.raises(ValueError, match="UTF-8"):
        read_gguf_metadata(path)
    for missing in (tmp_path, tmp_path / "missing"):
        with pytest.raises(ValueError, match="普通文件"):
            read_gguf_metadata(missing)


def test_invalid_magic_and_oversized_template_list_fail_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "invalid.gguf"
    complete = _write(path)
    path.write_bytes(b"NOT!" + complete[4:])
    with pytest.raises(ValueError, match="文件标识"):
        read_gguf_metadata(path)
    _write(
        path,
        fields=[
            *_fields(),
            ("tokenizer.chat_templates", 9, (8, [f"template_{i}" for i in range(257)])),
        ],
    )
    with pytest.raises(ValueError, match="命名聊天模板数量.*限制"):
        read_gguf_metadata(path)


def test_bf16_payload_and_custom_alignment_bounds(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write(
        path,
        fields=[*_fields(), ("general.alignment", 4, 64)],
        tensors=[("weight", [3], 30, 0)],
        alignment=64,
    )
    metadata = read_gguf_metadata(path)
    assert metadata["tensor_types"] == [30] and metadata["data_offset"] % 64 == 0
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="截断"):
        read_gguf_metadata(path)
