"""原始表格导入：CSV / Excel / JSONL → 统一的 RawTable。

设计参考 H2O LLM Studio 的数据集导入模式：先把用户的文件读成统一形状，
再做列映射，而不是为每种格式各写一套后续流程。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.wizard.spec import FieldMapping, WizardError

_VARIANT_DELIMITERS = "、;；,，|/"


@dataclass(frozen=True)
class RawTable:
    """归一化后的原始表格：所有值统一为 str | None（空/缺失 → None）。"""

    source: str
    columns: list[str]
    rows: list[dict[str, str | None]]


def _normalize(value: object) -> str | None:
    """把单元格值归一成字符串；空值返回 None。

    Excel 数字列常被读成 float（如编码 123 → 123.0），整数浮点还原成 "123"。
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else None


def _from_dataframe(df: pd.DataFrame, source: str) -> RawTable:
    columns = [str(c) for c in df.columns.tolist()]
    rows = [
        {col: _normalize(val) for col, val in zip(columns, row)}
        for row in df.itertuples(index=False, name=None)
    ]
    return RawTable(source=source, columns=columns, rows=rows)


def import_table(path: str | Path) -> RawTable:
    """按扩展名导入表格文件，返回归一化的 RawTable。"""
    p = Path(path)
    if not p.exists():
        raise WizardError(f"文件不存在: {p}")
    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(p, dtype=str)
        elif suffix == ".jsonl":
            return _import_jsonl(p)
        else:
            raise WizardError(f"不支持的文件格式 '{suffix}'。支持: .csv / .xlsx / .xls / .jsonl。")
    except WizardError:
        raise
    except ImportError as exc:
        raise WizardError(
            f"读取 {suffix} 文件缺少依赖（{exc}）。可运行: pip install openpyxl"
        ) from exc
    except Exception as exc:  # pandas 解析错误等，统一转成用户可读消息
        raise WizardError(f"解析文件失败 {p.name}: {exc}") from exc
    if df.empty:
        raise WizardError(f"文件没有数据行: {p.name}")
    return _from_dataframe(df, source=str(p))


def _import_jsonl(p: Path) -> RawTable:
    """JSONL：每行一个 JSON 对象，列名取所有键的并集（保持首次出现顺序）。"""
    rows: list[dict[str, str | None]] = []
    columns: list[str] = []
    with open(p, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WizardError(f"JSONL 第 {line_no} 行不是合法 JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise WizardError(f"JSONL 第 {line_no} 行不是 JSON 对象。")
            row = {k: _normalize(v) for k, v in obj.items()}
            for key in row:
                if key not in columns:
                    columns.append(key)
            rows.append(row)
    if not rows:
        raise WizardError(f"文件没有数据行: {p.name}")
    # 列并集确定后统一补齐，保证每行 dict 键一致
    normalized = [{k: r.get(k) for k in columns} for r in rows]
    return RawTable(source=str(p), columns=columns, rows=normalized)


def split_variants(value: str | None) -> list[str]:
    """把变体列的一个单元格按常见分隔符切成列表（去空、去重、保序）。"""
    if value is None:
        return []
    parts: list[str] = []
    # 逐字符分隔符切割：
    tokens: list[str] = []
    current: list[str] = []
    for ch in value:
        if ch in _VARIANT_DELIMITERS:
            tokens.append("".join(current))
            current = []
        else:
            current.append(ch)
    tokens.append("".join(current))
    for token in tokens:
        text = token.strip()
        if text and text not in parts:
            parts.append(text)
    return parts


# ─── 列名智能映射建议 ────────────────────────────────────────────────

_ROLE_HINTS: dict[str, list[str]] = {
    "standard_name": ["标准名", "标准", "规范名", "standard", "canonical"],
    "query": ["查询", "别名", "原始名", "输入", "原名", "query", "alias"],
    "code": ["编码", "代码", "code", "编号"],
    "variants": ["变体", "别名集", "variants"],
    "entity_type": ["类型", "类别", "type", "category"],
    "spec": ["规格", "spec"],
}


def suggest_mapping(columns: list[str]) -> FieldMapping:
    """根据列名猜测字段映射（用户未显式指定时的默认值）。

    每个角色取第一个命中的列；standard_name 没命中时退回第一列，
    保证向导始终有一个可运行的起点让用户修正。
    """
    found: dict[str, str] = {}
    lowered = [c.lower() for c in columns]
    for role in ("standard_name", "query", "code", "variants", "entity_type", "spec"):
        for hint in _ROLE_HINTS[role]:
            for col, col_l in zip(columns, lowered):
                if col_l not in found.values() and hint in col_l:
                    found[role] = col
                    break
            if role in found:
                break
    if "standard_name" not in found:
        if not columns:
            raise WizardError("表格没有任何列，无法建议映射。")
        found["standard_name"] = columns[0]
    return FieldMapping(
        standard_name=found.get("standard_name", ""),
        query=found.get("query"),
        code=found.get("code"),
        variants=found.get("variants"),
        entity_type=found.get("entity_type"),
        spec=found.get("spec"),
    )
