"""Literal-text XLSX handoffs that preserve data without creating formulas."""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from zipfile import ZipFile


def text_workbook(rows: Iterable[Sequence[str]], *, title: str) -> bytes:
    """Write a rectangular table as literal text; reject unrepresentable content."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    iterator = iter(rows)
    header = next(iterator, None)
    if not header or len(header) > 16_384:
        raise ValueError("清单缺少表头或超过 Excel 的行列限制，请使用原始 CSV。")
    width = len(header)
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = title
    sheet.freeze_panes = "B2" if width > 1 else "A2"
    sheet.column_dimensions["A"].width = 18
    if width > 1:
        sheet.column_dimensions["B"].width = 56
    for column in range(3, width):
        sheet.column_dimensions[get_column_letter(column)].width = 22
    if width > 2:
        sheet.column_dimensions[get_column_letter(width)].width = 60
    wrapped = Alignment(vertical="top", wrap_text=True)
    has_carriage_return = False

    def append(index: int, values: Sequence[str]) -> None:
        nonlocal has_carriage_return
        if index > 1_048_576 or len(values) != width:
            raise ValueError("清单超过 Excel 的行列限制或列数不一致，请使用原始 CSV。")
        for value in values:
            if not isinstance(value, str):
                raise ValueError("清单包含无法写入 Excel 的非文本内容。")
            has_carriage_return |= "\r" in value
            try:
                units = len(value.encode("utf-16-le")) // 2
            except UnicodeEncodeError:
                raise ValueError("清单包含无法写入 Excel 的字符，请核对原始资料。") from None
            if units > 32_767:
                raise ValueError("清单包含超过 Excel 单元格长度限制的内容，请使用原始 CSV。")
            if any(
                not (
                    char in "\t\n\r"
                    or 0x20 <= ord(char) <= 0xD7FF
                    or 0xE000 <= ord(char) <= 0xFFFD
                    or 0x10000 <= ord(char) <= 0x10FFFF
                )
                for char in value
            ):
                raise ValueError("清单包含无法写入 Excel 的字符，请使用原始 CSV。")
        sheet.append(values)
        for column in range(1, width + 1):
            cell = sheet.cell(index, column)
            cell.data_type = "s"
            if index == 1:
                cell.font = Font(bold=True)
            if column in {2, width}:
                cell.alignment = wrapped

    append(1, header)
    count = 1
    for count, row in enumerate(iterator, 2):
        append(count, row)
    sheet.auto_filter.ref = f"A1:{get_column_letter(width)}{count}"
    output = io.BytesIO()
    workbook.save(output)
    if not has_carriage_return:
        return output.getvalue()
    # XML normalizes literal CR to LF; references retain the source text.
    escaped = io.BytesIO()
    with ZipFile(io.BytesIO(output.getvalue())) as source_zip, ZipFile(escaped, "w") as target_zip:
        for entry in source_zip.infolist():
            content = source_zip.read(entry)
            if entry.filename == "xl/worksheets/sheet1.xml":
                content = content.replace(b"\r", b"&#13;")
            target_zip.writestr(entry, content)
    return escaped.getvalue()
