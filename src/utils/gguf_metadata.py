"""Bounded GGUF structure inspection without model or vocabulary materialization.

This validates metadata, tensor tables and payload bounds, not quantized block
contents, architecture support or inference quality. A real loader remains a
separate handoff check. Format follows ggml-org/llama.cpp's GGUF v2/v3 reader.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, BinaryIO

_MAX_TENSORS = 100_000
_MAX_METADATA = 100_000
_MAX_ARRAY = 2_000_000
_MAX_TOTAL_ARRAY_ELEMENTS = 8_000_000
_MAX_STRING_BYTES = 16 * 1024 * 1024
_MAX_RETAINED_BYTES = 16 * 1024 * 1024
_MAX_TEMPLATE_NAMES = 256
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "B",
    10: "Q",
    11: "q",
    12: "d",
}
_INTEGER_TYPES = {0, 1, 2, 3, 4, 5, 10, 11}
_TENSOR_TYPES = {0, 1, 2, 3, *range(6, 31), 34, 35, 39, 40, 41, 42}
_UNQUANTIZED_BYTES = {0: 4, 1: 2, 24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2}
_STRING_KEYS = {"general.architecture", "tokenizer.ggml.model", "tokenizer.chat_template"}
_INTEGER_KEYS = {
    "general.file_type",
    "general.alignment",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
}
_BOOL_KEYS = {"tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"}


class _Reader:
    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream = stream
        self.size = size
        self.endian = "<"
        self.retained_bytes = 0
        self.array_elements = 0

    def check(self, count: int) -> None:
        if count < 0 or count > self.size - self.stream.tell():
            raise ValueError("GGUF 内容截断：字段长度或偏移超出文件。")

    def read(self, count: int) -> bytes:
        self.check(count)
        value = self.stream.read(count)
        if len(value) != count:
            raise ValueError("GGUF 内容截断：文件未包含完整字段。")
        return value

    def skip(self, count: int) -> None:
        self.check(count)
        self.stream.seek(count, os.SEEK_CUR)

    def integer(self, code: str) -> int:
        return int(struct.unpack(self.endian + code, self.read(struct.calcsize(code)))[0])

    def string(self, *, capture: bool, limit: int = _MAX_STRING_BYTES) -> str | None:
        length = self.integer("Q")
        if length > limit:
            raise ValueError("GGUF 字符串长度超过有界读取限制。")
        if not capture:
            self.skip(length)
            return None
        self.retained_bytes += length
        if self.retained_bytes > _MAX_RETAINED_BYTES:
            raise ValueError("GGUF 元数据文本总量超过有界读取限制。")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("GGUF 元数据名称或文本不是有效 UTF-8。") from exc

    def array_header(self) -> tuple[int, int]:
        element_type, count = self.integer("I"), self.integer("Q")
        if element_type not in {*_SCALARS, 8, 9}:
            raise ValueError("GGUF 数组包含未知值类型。")
        self.array_elements += count
        if count > _MAX_ARRAY or self.array_elements > _MAX_TOTAL_ARRAY_ELEMENTS:
            raise ValueError("GGUF 数组数量超过有界读取限制。")
        return element_type, count

    def skip_value(self, value_type: int, *, depth: int = 0) -> None:
        if value_type in _SCALARS:
            if value_type == 7:
                if self.integer("B") not in (0, 1):
                    raise ValueError("GGUF 布尔字段值无效。")
            else:
                self.skip(struct.calcsize(_SCALARS[value_type]))
        elif value_type == 8:
            self.string(capture=False)
        elif value_type == 9:
            if depth >= 4:
                raise ValueError("GGUF 嵌套数组超过有界读取限制。")
            element_type, count = self.array_header()
            if element_type in _SCALARS and element_type != 7:
                self.skip(struct.calcsize(_SCALARS[element_type]) * count)
            else:
                for _ in range(count):
                    self.skip_value(element_type, depth=depth + 1)
        else:
            raise ValueError("GGUF 元数据包含未知值类型。")


def _read_fields(reader: _Reader, count: int) -> tuple[dict[str, Any], int | None]:
    values: dict[str, Any] = {}
    seen: set[str] = set()
    token_count: int | None = None
    template_count = 0
    for _ in range(count):
        key = reader.string(capture=True, limit=4096)
        if key is None or not key or key in seen:
            raise ValueError("GGUF 元数据键为空或重复。")
        seen.add(key)
        value_type = reader.integer("I")
        if key in _STRING_KEYS or key.startswith("tokenizer.chat_template."):
            if value_type != 8:
                raise ValueError("GGUF 结构关键文本字段类型无效。")
            if key.startswith("tokenizer.chat_template."):
                template_count += 1
                if template_count > _MAX_TEMPLATE_NAMES:
                    raise ValueError("GGUF 命名聊天模板数量超过有界读取限制。")
            values[key] = reader.string(capture=True)
        elif key in _INTEGER_KEYS:
            if value_type not in _INTEGER_TYPES:
                raise ValueError("GGUF 结构关键整数字段类型无效。")
            values[key] = reader.integer(_SCALARS[value_type])
        elif key in _BOOL_KEYS:
            if value_type != 7:
                raise ValueError("GGUF 结构关键布尔字段类型无效。")
            value = reader.integer("B")
            if value not in (0, 1):
                raise ValueError("GGUF 布尔字段值无效。")
            values[key] = bool(value)
        elif key in {"tokenizer.ggml.tokens", "tokenizer.chat_templates"}:
            if value_type != 9:
                raise ValueError("GGUF 词表或模板清单不是字符串数组。")
            element_type, length = reader.array_header()
            if element_type != 8:
                raise ValueError("GGUF 词表或模板清单不是字符串数组。")
            if key == "tokenizer.ggml.tokens":
                token_count = length
                for _ in range(length):
                    reader.string(capture=False)
            else:
                if length > _MAX_TEMPLATE_NAMES:
                    raise ValueError("GGUF 命名聊天模板数量超过有界读取限制。")
                values[key] = [reader.string(capture=True, limit=4096) for _ in range(length)]
        else:
            reader.skip_value(value_type)
    return values, token_count


def _tensor_bounds(reader: _Reader, count: int, alignment: int) -> tuple[int, list[int]]:
    names: set[str] = set()
    spans: list[tuple[int, int]] = []
    types: set[int] = set()
    for _ in range(count):
        name = reader.string(capture=True, limit=4096)
        if name is None or not name or name in names:
            raise ValueError("GGUF 张量名称为空或重复。")
        names.add(name)
        dimensions = reader.integer("I")
        if not 1 <= dimensions <= 4:
            raise ValueError("GGUF 张量维度数量无效。")
        elements = 1
        for _ in range(dimensions):
            dimension = reader.integer("Q")
            if dimension < 1:
                raise ValueError("GGUF 张量维度为空。")
            elements *= dimension
        tensor_type, offset = reader.integer("I"), reader.integer("Q")
        if tensor_type not in _TENSOR_TYPES:
            raise ValueError("GGUF 张量包含未知或不支持的编码类型。")
        if offset % alignment:
            raise ValueError("GGUF 张量数据偏移未按声明对齐。")
        # Exact sizes for scalar encodings; quantized encodings are checked at a
        # conservative 1-bit lower bound, not by their full block-size contract.
        minimum_bytes = (
            elements * _UNQUANTIZED_BYTES[tensor_type]
            if tensor_type in _UNQUANTIZED_BYTES
            else (elements + 7) // 8
        )
        spans.append((offset, minimum_bytes))
        types.add(tensor_type)
    data_start = (reader.stream.tell() + alignment - 1) // alignment * alignment
    if data_start >= reader.size:
        raise ValueError("GGUF 张量数据缺失或内容截断。")
    previous_end = 0
    for offset, minimum_bytes in sorted(spans):
        if offset < previous_end or data_start + offset + minimum_bytes > reader.size:
            raise ValueError("GGUF 张量数据重叠、缺失或内容截断。")
        previous_end = offset + minimum_bytes
    return data_start, sorted(types)


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
    """Read bounded v2/v3 metadata and reject incomplete weight/tokenizer structure.

    Vocabulary strings are skipped after checking lengths; the returned metadata
    never contains the vocabulary or tensor payloads. Missing optional token IDs
    and templates remain explicit ``None`` or empty dictionaries.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError("GGUF 路径不是可读取的普通文件。")
    with source.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        reader = _Reader(stream, size)
        if reader.read(4) != b"GGUF":
            raise ValueError("GGUF 文件标识无效。")
        version_bytes = reader.read(4)
        version = int.from_bytes(version_bytes, "little")
        if version not in (2, 3):
            version = int.from_bytes(version_bytes, "big")
            reader.endian = ">"
        if version not in (2, 3):
            raise ValueError("GGUF 仅支持有界检查版本 2 或 3。")
        tensor_count, metadata_count = reader.integer("Q"), reader.integer("Q")
        if not 1 <= tensor_count <= _MAX_TENSORS or not 1 <= metadata_count <= _MAX_METADATA:
            raise ValueError("GGUF 缺少模型权重/元数据，或数量超过有界读取限制。")
        values, token_count = _read_fields(reader, metadata_count)
        architecture, tokenizer_model = (
            values.get("general.architecture"),
            values.get("tokenizer.ggml.model"),
        )
        file_type = values.get("general.file_type")
        if (
            not architecture
            or not architecture.strip()
            or type(file_type) is not int
            or file_type < 0
        ):
            raise ValueError("GGUF 缺少有效模型架构或文件类型。")
        if not tokenizer_model or not tokenizer_model.strip() or not token_count:
            raise ValueError("GGUF 缺少有效 tokenizer 或非空词表。")
        alignment = values.get("general.alignment", 32)
        if not 1 <= alignment <= 65536 or alignment & (alignment - 1):
            raise ValueError("GGUF 对齐参数必须为有界的正二次幂。")
        data_start, tensor_types = _tensor_bounds(reader, tensor_count, alignment)
        template_names = values.get("tokenizer.chat_templates", [])
        if any(not name for name in template_names) or len(set(template_names)) != len(
            template_names
        ):
            raise ValueError("GGUF 命名聊天模板清单为空或重复。")
        templates = {
            key.removeprefix("tokenizer.chat_template."): value
            for key, value in values.items()
            if key.startswith("tokenizer.chat_template.")
        }
        if set(templates) != set(template_names):
            raise ValueError("GGUF 命名聊天模板与声明清单不一致。")
        if "tokenizer.chat_template" in values:
            if "default" in templates:
                raise ValueError("GGUF 默认聊天模板定义重复。")
            templates["default"] = values["tokenizer.chat_template"]
        ids = {
            label: values.get(key)
            for label, key in (
                ("bos_token_id", "tokenizer.ggml.bos_token_id"),
                ("eos_token_id", "tokenizer.ggml.eos_token_id"),
                ("pad_token_id", "tokenizer.ggml.padding_token_id"),
            )
        }
        if any(value is not None and not 0 <= value < token_count for value in ids.values()):
            raise ValueError("GGUF 特殊 token ID 超出词表范围。")
        if os.fstat(stream.fileno()).st_size != size:
            raise ValueError("读取 GGUF 时文件长度发生变化。")
    return {
        "version": version,
        "endian": "little" if reader.endian == "<" else "big",
        "tensor_count": tensor_count,
        "tensor_types": tensor_types,
        "data_offset": data_start,
        "architecture": architecture,
        "file_type": file_type,
        "tokenizer_model": tokenizer_model,
        "token_count": token_count,
        "chat_template": templates,
        **ids,
        "add_bos_token": values.get("tokenizer.ggml.add_bos_token"),
        "add_eos_token": values.get("tokenizer.ggml.add_eos_token"),
        "validation_scope": "GGUF 元数据、张量表与数据偏移已核查；量化数据仅核查保守长度下界，完整块内容、实际可加载性与业务效果尚需独立验证。",
    }
