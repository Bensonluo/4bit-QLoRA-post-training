"""内置演示任务:真实试用者的冷启动入口(单一来源)。

演示数据是虚构的售后工单,与 docs/validation/user-trial-log.md 的试点任务
同一份文件。入口只代替「找文件 + 填表」这一步;任务创建后的每一道关卡
(基础分析、预览核对、对比核验、盲标核验……)都与真实任务完全相同——没有
预设结论,也不跳过任何门禁。配套的演示全量文件只对「来源就是演示样例」的
任务开放(按文件内容摘要比对,同名不同内容不算),不会混进任何真实任务。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DEMO_GOAL = "根据客户首次咨询判断售后问题类型"
DEMO_DESCRIPTION = (
    "演示数据（虚构）：一行一张售后工单，客户描述来自客户首次咨询，"
    "类别是人工审核的问题类型，处理结果是事后信息。"
)
DEMO_SAMPLE_NAME = "aftercare_tickets_sample.csv"
DEMO_FULL_NAME = "aftercare_tickets_full.csv"

DEMO_SAMPLE_ENTRY_LABEL = "第一次使用？用内置演示任务开始"
DEMO_SAMPLE_BUTTON = "创建演示任务（售后工单分类）"
DEMO_FULL_BUTTON = "使用配套演示全量数据（虚构，10 条）"

_EXAMPLES_DIR = Path("data") / "custom" / "examples"


def _demo_bytes(project_root: Path, name: str) -> bytes | None:
    """读内置演示文件;不在场时返回 None(如实降级,不编造演示数据)。"""
    try:
        return (project_root / _EXAMPLES_DIR / name).read_bytes()
    except OSError:
        return None


def demo_sample(project_root: Path) -> tuple[str, bytes] | None:
    """演示样例的(文件名, 内容);文件不在场时返回 None,页面据此隐藏入口。"""
    data = _demo_bytes(project_root, DEMO_SAMPLE_NAME)
    return None if data is None else (DEMO_SAMPLE_NAME, data)


def demo_full(project_root: Path) -> tuple[str, bytes] | None:
    """演示全量文件的(文件名, 内容);文件不在场时返回 None。"""
    data = _demo_bytes(project_root, DEMO_FULL_NAME)
    return None if data is None else (DEMO_FULL_NAME, data)


def is_demo_session(source_digest: str, project_root: Path) -> bool:
    """任务来源是否就是演示样例(按内容摘要比对,同名不同内容不算)。"""
    sample = demo_sample(project_root)
    return sample is not None and source_digest == hashlib.sha256(sample[1]).hexdigest()
