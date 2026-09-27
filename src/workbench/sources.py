"""Non-destructive sample reading and factual profiles (no model inference)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from src.workbench.intake_models import SampleSource, SourceRow

_CSV_READ_LOCK = threading.Lock()


def _read_csv(text: str, delimiter: str) -> tuple[list[str], list[tuple[int, dict[str, Any]]]]:
    # csv's field limit is process-global. Serialize our readers and restore it even
    # on malformed input; the supplied file already bounds the required field size.
    with _CSV_READ_LOCK:
        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(max(previous_limit, len(text)))
            reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
            records = []
            try:
                columns = _headers(next(reader, []))
                while True:
                    first_line = reader.line_num + 1
                    values = next(reader, None)
                    if values is None:
                        break
                    if not values:
                        continue
                    if len(values) != len(columns):
                        raise ValueError(
                            f"第 {first_line} 行有 {len(values)} 列，表头为 {len(columns)} 列；"
                            "请检查分隔符或引号，未静默丢弃该行。"
                        )
                    records.append((first_line, dict(zip(columns, values))))
            except csv.Error as exc:
                raise ValueError(f"CSV 第 {reader.line_num} 行解析失败：{exc}") from exc
            return columns, records
        finally:
            csv.field_size_limit(previous_limit)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _decode(data: bytes, encoding: str | None) -> tuple[str, str]:
    candidates = [encoding] if encoding else ["utf-8-sig", "gb18030"]
    # UTF-16 BOM(FF FE / FE FF,Excel「Unicode 文本」导出的常见形态)是明确的编码证据:
    # 0xFF/0xFE+0xFF 开头的字节永远不可能是合法 UTF-8 或 GBK,按 BOM 识别不会误判。
    if not encoding and data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        candidates.append("utf-16")
    for candidate in candidates:
        try:
            return data.decode(candidate), candidate
        except UnicodeDecodeError:
            continue
        except LookupError as exc:
            raise ValueError("无法识别指定编码，请选择有效编码。") from exc
    raise ValueError("无法解码文件，请明确指定编码；原始数据未修改。")


def _headers(values: list[Any]) -> list[str]:
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("表头必须是非空文本；请确认文件的第一行是字段名。")
    if len(values) != len(set(values)):
        raise ValueError("存在重复列名，无法明确引用字段；请先区分列名。")
    return values


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 存在重复字段：{key}")
        result[key] = value
    return result


def parse_json_value(text: str) -> Any:
    """Parse structured cells without silently discarding duplicate keys or NaN."""
    value = json.loads(text, object_pairs_hook=_unique_object)
    canonical(value)
    return value


def read_source(
    name: str,
    data: bytes,
    *,
    scope: Literal["sample", "full"] = "sample",
    encoding: str | None = None,
    delimiter: str | None = None,
) -> SampleSource:
    """Read supplied bytes only; never follow paths mentioned inside the data."""
    suffix = Path(name).suffix.lower().lstrip(".")
    records: list[tuple[int, dict[str, Any]]] = []
    actual_encoding, actual_delimiter = "", ""
    if suffix in {"csv", "jsonl"}:
        text, actual_encoding = _decode(data, encoding)
        if suffix == "csv":
            if delimiter is not None and delimiter not in (",", ";", "\t", "|"):
                raise ValueError("分隔符支持逗号、分号、Tab 或竖线。")
            try:
                sniffed = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
            except csv.Error:
                sniffed = ","
            actual_delimiter = delimiter or sniffed
            columns, records = _read_csv(text, actual_delimiter)
        else:
            columns = []
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = parse_json_value(line)
                    canonical(record)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"JSONL 第 {line_no} 行无效：{exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL 第 {line_no} 行必须是对象。")
                _headers(list(record))
                for key in record:
                    if key not in columns:
                        columns.append(key)
                records.append((line_no, record))
    elif suffix in {"xlsx", "xls"}:
        import pandas as pd

        try:
            frame = pd.read_excel(
                io.BytesIO(data), header=None, dtype=object, keep_default_na=False
            )
        except ImportError as exc:
            raise ValueError(
                "读取 Excel 缺少对应引擎，请安装 openpyxl（xlsx）或 xlrd（xls）。"
            ) from exc
        if frame.empty:
            raise ValueError("文件没有表头或数据。")
        columns = _headers(frame.iloc[0].tolist())
        for i, values in enumerate(frame.iloc[1:].itertuples(index=False, name=None), 2):
            normalized = [v.isoformat() if hasattr(v, "isoformat") else v for v in values]
            records.append((i, dict(zip(columns, normalized))))
    else:
        raise ValueError("当前数据入口支持 CSV、Excel、JSONL。")
    if not records:
        raise ValueError("文件没有数据行。")
    rows = [
        SourceRow(row_id=f"r{i:06d}", line=line, values=record)
        for i, (line, record) in enumerate(records, 1)
    ]
    return SampleSource(
        name=Path(name).name,
        digest=hashlib.sha256(data).hexdigest(),
        scope=scope,
        format=suffix,
        encoding=actual_encoding,
        delimiter=actual_delimiter,
        columns=columns,
        rows=rows,
    )


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (dict, list)):
        return "object" if isinstance(value, dict) else "array"
    if isinstance(value, (int, float)):
        return "number"
    return "text"


def profile_source(source: SampleSource) -> dict[str, Any]:
    """All counts apply exclusively to the provided file, even when called a sample."""
    fields: dict[str, Any] = {}
    evidence: list[str] = []

    def add(row_id: str) -> None:
        if row_id not in evidence:
            evidence.append(row_id)

    for row in source.rows[:3]:
        add(row.row_id)
    for column in source.columns:
        values = [row.values.get(column) for row in source.rows]
        present = [(i, v) for i, v in enumerate(values) if not is_missing(v)]
        counts = Counter(canonical(v) for _, v in present)
        missing_ids = [row.row_id for row, value in zip(source.rows, values) if is_missing(value)]
        if missing_ids:
            add(missing_ids[0])
        if present:
            shortest = min(present, key=lambda pair: len(str(pair[1])))
            longest = max(present, key=lambda pair: len(str(pair[1])))
            add(source.rows[shortest[0]].row_id)
            add(source.rows[longest[0]].row_id)
            rare = min(counts, key=counts.get)
            add(source.rows[next(i for i, v in present if canonical(v) == rare)].row_id)
        texts = [v for _, v in present if isinstance(v, str)]
        numeric_candidates = 0
        for value in texts:
            try:
                numeric_candidates += int(math.isfinite(float(value)))
            except ValueError:
                continue
        fields[column] = {
            "missing_count": len(missing_ids),
            "missing_row_ids": missing_ids[:5],
            "distinct_count": len(counts),
            "value_types": dict(Counter(_value_type(v) for _, v in present)),
            "min_length": min((len(str(v)) for _, v in present), default=0),
            "max_length": max((len(str(v)) for _, v in present), default=0),
            "numeric_text_count": numeric_candidates,
            "leading_zero_examples": [
                v for v in texts if len(v) > 1 and v.startswith("0") and v.isdigit()
            ][:3],
            "common_values": [
                {"value": json.loads(k), "count": n} for k, n in counts.most_common(5)
            ],
        }
    duplicates: dict[str, list[str]] = {}
    for row in source.rows:
        duplicates.setdefault(canonical(row.values), []).append(row.row_id)
    groups = [ids for ids in duplicates.values() if len(ids) > 1]
    return {
        "source_digest": source.digest,
        "source_scope": source.scope,
        "record_count": len(source.rows),
        "columns": fields,
        "duplicate_record_count": sum(len(ids) - 1 for ids in groups),
        "duplicate_groups": groups[:10],
        "evidence_row_ids": evidence,
        "scope_note": (
            "以上仅描述用户提供的样例，不能推断全量覆盖、比例或训练就绪。"
            if source.scope == "sample"
            else "以上描述本次提供的文件；统计通过不代表监督含义正确或已具备独立评测条件。"
        ),
    }
