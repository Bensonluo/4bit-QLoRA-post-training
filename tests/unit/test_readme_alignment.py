"""README 与北极星对齐:本地链接可解析,语义安全层与场景矩阵如实在场。"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

# 北极星权威版;README 只能指向它,不能另立版本。
NORTH_STAR = "docs/plans/north-star.md"

LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
IMG_PATTERN = re.compile(r'<img\s+src="([^"]+)"')


def _local_targets(text):
    for match in LINK_PATTERN.findall(text) + IMG_PATTERN.findall(text):
        target = match.strip()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        yield target


# README 录制指南明确声明 dashboard.gif 为待录制的占位资源(保存路径已写死)。
# 除该显式占位外,任何仓库内引用都必须真实存在。
DOCUMENTED_PENDING_ASSETS = {"docs/assets/dashboard.gif"}


def test_all_local_links_resolve_to_existing_files():
    """README 引用的每个仓库内路径都必须存在——不再引用已删除文档。"""
    broken = [
        target
        for target in _local_targets(README.read_text(encoding="utf-8"))
        if target not in DOCUMENTED_PENDING_ASSETS
        and not (ROOT / target.split("#")[0]).exists()
    ]
    assert broken == [], f"README 引用了不存在的文件: {broken}"


def test_product_description_covers_semantic_safety_layer():
    """产品主路径描述必须包含语义安全层(盲标/对比/探针)与场景矩阵。"""
    text = README.read_text(encoding="utf-8")
    for term in ("盲标核验", "对比核验", "可学性探针", "场景矩阵", "语义安全"):
        assert term in text, f"README 缺少语义安全层关键词: {term}"
    # 英文主描述同样可见(关键词的英文名或中文名至少其一)。
    assert "semantic safety layer" in text.lower()


def test_readme_points_to_authoritative_north_star():
    assert (ROOT / NORTH_STAR).exists()
    assert README.read_text(encoding="utf-8").count(NORTH_STAR) >= 1


def test_license_file_exists_and_declares_mit():
    """README 与 pyproject 都声明 MIT,许可证文件必须真实存在。"""
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Permission is hereby granted" in license_text
