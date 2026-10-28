"""手工训练参数与分区设置的判断辅助（单一来源，页面与 CLI 共用）。

北极星「判断编码飞轮」的落地：零密钥路径「高级：手工配置训练参数」是
旅程中自认专家依赖最重的一步（docs/validation/user-trial-log.md 介入点
6/7），参数含义、推荐起步值与「何时该改」的引导不应只有页面一个入口。
本模块把这些话集中为一处纯函数，页面与 CLI 同源同词汇；不写状态、不
发起任何计算。

诚实边界：推荐值与学习率分档是外部指南的汇总启发（非本产品实测），
以用户自己的同题对照结果为准；分区条数提醒是 1/N 的诚实算术加粗略
经验阈值，不是统计保证，也不替用户决定比例。
"""

from __future__ import annotations

LR_TIER_DISCLAIMER = (
    "该分档是 2026 年外部指南的汇总启发（Unsloth 指南、Raschka 实践笔记等），"
    "非本产品实测，以你自己的同题对照结果为准。"
)

SFT_PROTOCOL_LABELS = {
    "alpaca_full_prompt_v1": "学习输入提示与答案（现有默认）",
    "alpaca_answer_only_v1": "只学习答案与结束标记",
}


def supervision_label(sft_protocol: str) -> str:
    """Explain the exact objective without suggesting a measured quality gain."""
    return SFT_PROTOCOL_LABELS.get(sft_protocol, f"未知监督协议：{sft_protocol}")


def learning_rate_suggestion(total_rows: int) -> tuple[float, str]:
    """按训练集条数给外部分档的起步档；保留 total_rows 参数名兼容已有调用。"""
    if total_rows < 2000:
        return 0.0001, (
            f"训练集 {total_rows} 条（< 2,000），建议学习率起步 5e-5~1e-4——"
            "小数据集用大学习率易过拟合、训练不稳，业界指南一致建议降档。"
        )
    return 0.0002, f"训练集 {total_rows} 条（≥ 2,000），建议学习率起步 2e-4——通用可靠默认。"


def learning_rate_from_statistics(statistics: dict | None) -> tuple[float | None, str]:
    """Only recommend from a recorded, positive training split count."""
    counts = statistics.get("row_counts") if isinstance(statistics, dict) else None
    train_rows = counts.get("train") if isinstance(counts, dict) else None
    if type(train_rows) is not int or train_rows <= 0:
        return None, "训练集条数缺失或无效，暂无法给出分档建议；请检查数据集版本。"
    return learning_rate_suggestion(train_rows)


def manual_training_parameter_lines() -> list[str]:
    """手工训练参数的逐参数大白话与推荐起步值（页面与 CLI 同源同词汇）。"""
    return [
        "参数大白话：训练轮数＝全部数据过几遍，少了学不会、多了把题背死。",
        "每设备 batch size＝每一步一起看几条数据，显存不够就保持 1。",
        "梯度累积步数＝攒几步再更新一次，相当于变相加大 batch。",
        "学习率＝每一步改动多大，太大训练发飘、太小学得慢。",
        "LoRA rank＝适配器记多大本事的容量，小任务 8 够用，越大越占显存。",
        "4-bit 量化＝显存放不下完整模型才开（仅限兼容的 NVIDIA 显卡）。",
        "最大 token 长度＝每条样本最多装多少内容，按训练前检查的建议填。",
        "监督范围＝哪些文字参与训练损失；可保留现有全提示监督，或只学习答案与结束标记。"
        "仅答案监督会逐行核验完整边界，不能截断；效果需用同题对照判断。",
        "推荐起步值（小数据）：训练轮数 1–2、每设备 batch size 1、梯度累积步数 4、"
        "学习率 0.0002（即 2e-4）、LoRA rank 8——与默认值一致；先跑通再调。",
    ]


def split_settings_guidance_lines() -> list[str]:
    """分区设置（验证/测试比例与种子）的「何时该改」引导。"""
    return [
        "验证集比例＝划出多少条不参与训练、用来边训边看；"
        "独立测试集比例＝再划出多少条完全留到最后验收。",
        "默认 0.1/0.1 在几千条记录时通常够用；记录只有几百条时 10% 只切出十几条，"
        "比例可提到 0.15–0.2，代价是训练数据变少。",
        "可复现分区种子＝同一份数据同一种子切出同一分区；只想换一种切分时才改。",
    ]


def small_test_set_line(test_count: int) -> str | None:
    """独立测试集条数的诚实算术提醒：少于 30 条时每题权重过大，结论偶然性大。

    1/N 是确定性算术；30 条是粗略经验阈值，不是统计保证。条数够大或非法
    时返回 None，不制造噪音。
    """
    if test_count <= 0 or test_count >= 30:
        return None
    return (
        f"独立测试集共 {test_count} 条——每条约占最终通过率 {100 / test_count:.0f} 个百分点；"
        "少于 30 条时结论偶然性大（粗略经验，不是统计保证）。"
    )
