"""数据集预检：训练启动前的轻量格式体检（Training Lab 提交时调用）。

行业惯例（LLaMA-Factory WebUI 的 dataset preview 步骤、「上传前先验证训练数据」
的平台实践）：错误的数据集如果到训练脚本里才失败，浪费的是一次完整启动
（下载模型 → 加载数据 → 报错）。本模块在提交时同步完成本地文件体检：

- 纯标准库（json 解析），不触碰 torch/transformers，UI 进程零额外开销；
- 识别 Alpaca（instruction/output）与 messages（role/content 列表）两种格式；
- Training Lab 的 SFT 启动器只消费 Alpaca 格式 —— messages 格式（如
  master_data 向导输出）给出明确修法，而不是让训练脚本晚几分钟才失败。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_JSON_SUFFIXES = (".json", ".jsonl")
_MAX_BAD_RECORDS_SHOWN = 3


@dataclass(frozen=True)
class DatasetInspection:
    """一次数据集体检的结果：格式、规模与问题清单。"""

    fmt: str  # "alpaca" | "messages" | "unknown" | ""（文件级失败）
    n_records: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def looks_like_local_path(dataset_name: str) -> bool:
    """区分本地文件与 HF 数据集名。

    HF 名是 ``org/name`` 形态（如 yahma/alpaca-cleaned），本身含 ``/``——
    不能拿斜杠当判据。本地路径的可靠特征：以 .json/.jsonl 结尾，或以
    /、./、../、~ 开头。无扩展名的本地目录路径会被当成 HF 名跳过检查
    （向导导出的训练集恒为 .json，不受影响）。
    """
    return dataset_name.endswith(_JSON_SUFFIXES) or dataset_name.startswith(("/", "./", "../", "~"))


def _load_records(path: Path) -> tuple[list[dict], list[str]]:
    """读取 JSON/JSONL 为记录列表；返回 (records, errors)。"""
    try:
        if path.suffix == ".jsonl":
            records: list[dict] = []
            errors: list[str] = []
            with open(path, encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        errors.append(f"第 {lineno} 行不是合法 JSON")
            return records, errors
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return [], [f"JSON 解析失败：{exc}"]
    except OSError as exc:
        return [], [f"无法读取文件：{exc}"]
    if not isinstance(data, list):
        return [], ["顶层不是 JSON 数组（应为 [{...}, ...] 的训练样本列表）"]
    bad = [i for i, rec in enumerate(data, 1) if not isinstance(rec, dict)]
    if bad:
        shown = "、".join(str(i) for i in bad[:_MAX_BAD_RECORDS_SHOWN])
        more = f" 等 {len(bad)} 条" if len(bad) > _MAX_BAD_RECORDS_SHOWN else ""
        return [], [f"第 {shown} 条记录不是 JSON 对象{more}"]
    return data, []


def _detect_format(records: list[dict]) -> str:
    """按首条记录判定格式：alpaca（instruction+output）/ messages / unknown。"""
    keys = set(records[0])
    if {"instruction", "output"} <= keys:
        return "alpaca"
    if isinstance(records[0].get("messages"), list):
        return "messages"
    return "unknown"


def inspect_dataset_file(path: Path) -> DatasetInspection:
    """体检单个本地数据集文件，返回格式、规模与问题清单。"""
    if not path.exists():
        return DatasetInspection(
            fmt="",
            n_records=0,
            errors=(
                f"数据集文件不存在：{path}。请确认路径无误；"
                f"向导导出的训练集在 outputs/wizard/<名称>/train.json。",
            ),
            warnings=(),
        )
    records, errors = _load_records(path)
    if errors:
        return DatasetInspection(fmt="", n_records=len(records), errors=tuple(errors), warnings=())
    if not records:
        return DatasetInspection(
            fmt="",
            n_records=0,
            errors=("数据集为空（0 条样本），训练无从进行。请检查导出或换一个文件。",),
            warnings=(),
        )

    fmt = _detect_format(records)
    warnings: list[str] = []
    if fmt == "alpaca":
        missing = [
            i for i, rec in enumerate(records, 1) if not ({"instruction", "output"} <= set(rec))
        ]
        if missing:
            shown = "、".join(str(i) for i in missing[:_MAX_BAD_RECORDS_SHOWN])
            more = f" 等 {len(missing)} 条" if len(missing) > _MAX_BAD_RECORDS_SHOWN else ""
            warnings.append(
                f"第 {shown} 条记录缺 instruction/output{more}——训练时读取这些字段会直接报错，建议修复。"
            )
    return DatasetInspection(fmt=fmt, n_records=len(records), errors=(), warnings=tuple(warnings))


def load_preview_records(path: Path, limit: int = 3) -> tuple[list[dict], str | None]:
    """读取前 limit 条记录供 UI 预览；返回 (records, error)。文件级失败返回 ([], 原因)。"""
    if not path.exists():
        return [], f"数据集文件不存在：{path}"
    records, errors = _load_records(path)
    if errors:
        return [], errors[0]
    return records[:limit], None


def check_dataset_for_sft(dataset_name: str) -> tuple[list[str], DatasetInspection | None]:
    """Training Lab（SFT）提交时的预检入口。

    返回 (errors, inspection)。HF 数据集名（不含路径分隔符）不做本地检查，
    返回 ([], None)；本地文件返回体检结果与按 SFT 消费方解释后的问题清单：
    messages 格式会挡下并给出正确去向，unknown 格式会挡下并列出首条字段。
    """
    if not looks_like_local_path(dataset_name):
        return [], None
    insp = inspect_dataset_file(Path(dataset_name))
    errors = list(insp.errors)
    if insp.fmt == "messages":
        errors.append(
            f"该文件是 messages 对话格式（{insp.n_records} 条），Training Lab 的 SFT 启动器"
            f"只消费 Alpaca 格式（instruction/output）。请改用域训练脚本："
            f"python domains/master_data/scripts/train.py --train-file {dataset_name}"
        )
    elif insp.fmt == "unknown":
        # 首条字段在 inspect 结果里未带回；重新读一次首条以给出可操作的修法
        records, _ = _load_records(Path(dataset_name))
        first_keys = "、".join(sorted(records[0].keys())) if records else ""
        errors.append(
            f"未识别的数据格式：首条记录字段为 [{first_keys}]，"
            f"SFT 训练需要 instruction/output 字段（Alpaca 格式）。"
            f"可用 Data Wizard 从原始表格生成合规训练集。"
        )
    elif insp.warnings:
        errors.extend(insp.warnings)  # 缺字段的记录训练时必然报错 → 按阻断处理
    return errors, insp
