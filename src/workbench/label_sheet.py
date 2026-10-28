"""Offline verification sheets that respect the selected supervision mode."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator

from src.workbench.text_workbook import text_workbook


def _verification_question_rows(items: list[dict], *, mode: str) -> Iterator[list[str]]:
    if mode == "reproduce":
        yield ["行ID", "题目输入（仅根据此列作答）", "你的盲标答案（填写后提交）"]
    elif mode == "review":
        yield ["行ID", "题目输入", "数据中的答复（供审阅）", "审阅结论（填写是或否）"]
    else:
        raise ValueError("不支持的核验方式，不能导出题目清单。")

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("核验题目格式无效，不能导出题目清单。")
        row_id, question = item.get("row_id"), item.get("input")
        if not isinstance(row_id, str) or not isinstance(question, str):
            raise ValueError("核验题目缺少行ID或输入文本，不能导出题目清单。")
        if mode == "reproduce":
            # Deliberately do not consult an answer field, even if one is supplied.
            yield [row_id, question, ""]
        else:
            answer = item.get("answer")
            if not isinstance(answer, str):
                raise ValueError("逐条审阅缺少数据中的答复，不能导出审阅清单。")
            yield [row_id, question, answer, ""]


def verification_questions_to_csv(items: list[dict], *, mode: str = "reproduce") -> bytes:
    """Keep original values in CSV for programmatic use."""
    buffer = io.StringIO()
    csv.writer(buffer).writerows(_verification_question_rows(items, mode=mode))
    return buffer.getvalue().encode("utf-8-sig")


def verification_questions_to_xlsx(items: list[dict], *, mode: str = "reproduce") -> bytes:
    """Write a literal-text workbook for offline business verification."""
    if len(items) + 1 > 1_048_576:
        raise ValueError("清单超过 Excel 的行列限制，请使用原始 CSV。")
    return text_workbook(_verification_question_rows(items, mode=mode), title="盲标核验")
