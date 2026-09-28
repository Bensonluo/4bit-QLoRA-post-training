"""Learnability probe: cheapest evidence that this data can teach this task at all.

Runs the base model zero-shot on a small fixed development sample and compares
exact-match with the majority-class guess baseline. The output is evidence with
explicit sample-size limits — never a verdict that the task is impossible, and
never a claim of business quality.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.workbench.sources import content_digest

# 弱信号候选的改标签门槛(单一来源):仅基座零样本不认同是弱证据,不足以修正数据;
# 人工核对后仍不认同才修正。候选清单的 note、CLI 逐行清单的表头与候选的证据列
# 都引用这一句,三处不各说各话。
WEAK_SIGNAL_RULE = "单凭模型不认同不改标签，人工核对后仍不认同才修正数据"


def probe_learnability(
    session,
    base_model_path: str,
    *,
    runtime_factory=None,
    sample_size: int = 8,
    max_new_tokens: int = 32,
) -> dict:
    """Zero-shot probe on the development split; compare with majority baseline."""
    from src.workbench.training_preflight import _verified_partitions

    if type(sample_size) is not int or not 1 <= sample_size <= 50:
        raise ValueError("抽样数量必须是 1 到 50 之间的整数。")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须是正整数。")
    if session.dataset is None:
        raise ValueError("请先生成独立分区；探针使用固定开发集抽样。")
    recipe = session.analysis.recipe
    if recipe is None or recipe.output_format != "text" or len(recipe.targets) != 1:
        raise ValueError("可学性探针当前支持单字段文本答案任务；其他形态暂不支持。")
    verification = {"issues": []}
    rows = _verified_partitions(session, verification)["validation"]
    if not rows:
        raise ValueError("当前数据没有开发集样本，无法探测。")

    from src.workbench.business_evaluation import EvaluationModel

    model = EvaluationModel("基座零样本", base_model_path)
    runtime = (runtime_factory or _default_runtime)(model)
    sampled = rows[: min(sample_size, len(rows))]
    correct = 0
    observations = []
    try:
        from src.data.loaders import render_alpaca_prompt

        for row in sampled:
            prompt = render_alpaca_prompt({**row, "output": ""})
            generation = runtime.generate(prompt, _probe_protocol(max_new_tokens))
            answer = generation.text.strip()
            hit = answer == row["output"]
            correct += hit
            observations.append(
                {
                    "row_id": row["metadata"]["source_row_id"],
                    "expected": row["output"],
                    "generated": answer,
                    "match": hit,
                    "truncated": generation.truncated,
                }
            )
    finally:
        closer = getattr(runtime, "close", None)
        if closer is not None:
            closer()

    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["output"]] = label_counts.get(row["output"], 0) + 1
    majority_share = max(label_counts.values()) / len(rows) if label_counts else 0.0
    majority_label = max(label_counts, key=label_counts.get) if label_counts else ""
    accuracy = correct / len(sampled)

    # 标签问题候选:基座零样本与数据标签矛盾的行。基座没见过这些数据,
    # 它的分歧不带训练利益;若盲标核验中用户也不同意数据标签,则是强证据
    # (人机双信号交叉)——这正是 cleanlab 式统计检错的任务无关轻量版。
    blind = getattr(session, "label_verification", None) or {}
    blind_items = {
        item.get("row_id"): item
        for item in (blind.get("items") or [])
        if isinstance(item, dict) and item.get("match") is False
    }
    candidates = []
    for observation in observations:
        if observation["match"] or observation["truncated"]:
            continue  # 截断的生成不算分歧证据
        row_id = observation["row_id"]
        user_also_disagrees = row_id in blind_items
        candidates.append(
            {
                "row_id": row_id,
                "data_label": observation["expected"],
                "base_zero_shot": observation["generated"],
                "user_blind_answer": blind_items[row_id].get("submitted_answer")
                if user_also_disagrees
                else None,
                "evidence": (
                    "基座零样本与用户盲标都不认同数据标签——强证据,优先人工核对,建议对照原始来源行"
                    if user_also_disagrees
                    else f"仅基座零样本不认同——模型可能错,标签也可能错,弱信号供参考;{WEAK_SIGNAL_RULE}"
                ),
            }
        )
    candidates.sort(key=lambda item: 0 if item["user_blind_answer"] else 1)
    result = {
        "kind": "learnability_probe",
        "dataset_version": session.dataset.version,
        "model_path": str(base_model_path),
        "protocol": {"generation": "greedy_v1", "max_new_tokens": max_new_tokens, "shots": 0},
        "sample_size": len(sampled),
        "dev_total": len(rows),
        "zero_shot_accuracy": accuracy,
        "majority_baseline": majority_share,
        "majority_label": majority_label,
        "label_vocabulary": sorted(label_counts),
        "difference": accuracy - majority_share,
        "observations": observations,
        "label_error_candidates": candidates,
        "candidates_note": (
            f"{len(candidates)} 行是标签问题候选(基座零样本与数据标签不一致;"
            f"其中 {sum(1 for c in candidates if c['user_blind_answer'])} 行与用户盲标也不一致)。"
            "候选不等于错误——模型可能错;但优先人工核对这些行是性价比最高的数据清理。"
            "与盲标也不一致的强证据行,建议对照原始来源行(row original 与来源文件行号)"
            f"溯源确认标签后再决定改不改,不要只凭模型输出下结论。{WEAK_SIGNAL_RULE}。"
        ),
        "note": (
            f"基座零样本 {accuracy:.0%} vs 全开发集多数类「{majority_label}」{majority_share:.0%}"
            f"（抽样 {len(sampled)}/{len(rows)} 条）。这只是数据与任务是否存在可学关联的"
            "最便宜证据：样本量小、零样本差异不能预测微调效果，也不构成业务达标结论；"
            "显著低于瞎猜基线通常意味着提示格式或任务定义需要先核查。"
        ),
    }
    return result


class _ProtocolLike:
    """Local stand-in with the fields LocalEvaluationRuntime.generate reads."""

    def __init__(self, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens
        self.strip_whitespace = True
        self.fields: tuple[str, ...] = ()
        self.custom_scoring = None


def _probe_protocol(max_new_tokens: int) -> _ProtocolLike:
    return _ProtocolLike(max_new_tokens)


def _default_runtime(model):
    from src.workbench.business_evaluation import LocalEvaluationRuntime

    return LocalEvaluationRuntime(model)


def save_probe(root: str | Path, result: dict) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # 证据记录自带生成时间:文件 mtime 在复制/同步后会丢,探针何时跑的应当
    # 由记录本身说明,回看时才能判断这份证据离当前数据有多远。
    result["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = root / f"{result['dataset_version']}-{content_digest(result)[:16]}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_latest_probe_record(root: str | Path, dataset_version: str) -> tuple[Path, dict] | None:
    """回读指定数据版本最近一次保存的探针结果,连同记录文件路径;没有则返回 None。

    探针要真实加载本地基座模型,重跑成本高;结果存盘后必须能原样回看,
    页面刷新或切换会话不丢证据。按数据版本过滤:数据重物化后旧结果不会
    冒充新版本的证据。记录文件路径一并返回:证据要能溯源到出处。
    """
    root = Path(root)
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.glob(f"{dataset_version}-*.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue  # 损坏的历史记录跳过,不阻塞回读
        if isinstance(data, dict) and data.get("kind") == "learnability_probe":
            return path, data
    return None


def load_latest_probe(root: str | Path, dataset_version: str) -> dict | None:
    record = load_latest_probe_record(root, dataset_version)
    return None if record is None else record[1]


def describe_candidates(candidates: list[dict], limit: int | None = 20) -> list[str]:
    """标签问题候选的人话清单:一行一个候选,供 CLI 直接打印到 stderr。

    强证据(基座零样本与用户盲标都不认同数据标签)加 ⚠ 标记;没有候选时
    给出明确说法而不是沉默。候选不等于错误——这是人工核对清单,不是判决。

    limit 与页面对齐:页面候选多于 20 条分页展示,CLI 默认也只逐行列出
    前 20 条;截断时明确说明总数与已显示条数,并指向导出 CSV 拿全量清单,
    不谎称已经全量显示。limit=None 表示不截断(逐行全列)。
    """
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit 必须是正整数或 None。")
    if not candidates:
        return ["没有发现值得优先核对的行。"]
    strong = sum(1 for item in candidates if item.get("user_blind_answer"))
    lines = [
        f"标签问题候选 {len(candidates)} 行（其中强证据 {strong} 行）；"
        f"候选不等于错误——基座可能错，标签也可能错；{WEAK_SIGNAL_RULE}："
    ]
    listed = candidates if limit is None else candidates[:limit]
    for item in listed:
        marker = "⚠" if item.get("user_blind_answer") else "·"
        blind = (
            f"，你的盲标「{item['user_blind_answer']}」" if item.get("user_blind_answer") else ""
        )
        trace = "，建议对照原始来源行" if item.get("user_blind_answer") else ""
        lines.append(
            f"{marker} 行 {item.get('row_id', '')}：数据标签「{item.get('data_label', '')}」"
            f"，基座零样本「{item.get('base_zero_shot', '')}」{blind}{trace}"
        )
    if limit is not None and len(candidates) > limit:
        rest = len(candidates) - limit
        lines.append(
            f"……以上只列出前 {limit}/{len(candidates)} 条，其余 {rest} 条候选"
            "见导出 CSV（learnability-probe --export-csv，或页面「导出候选为 CSV」）。"
        )
    return lines


def probe_verdict_phrase(result: dict) -> str:
    """探针判定短语:三态词汇的唯一来源,页面与 CLI 同源同词汇。

    证据不是判决:高于基线不预测微调效果,低于基线只指向「先核查」而非
    「任务不可学」——扩展解释统一放在 note 里,短语本身只说方向。
    """
    delta = result.get("difference")
    if delta is None:
        return "没有可比较的探针结果"
    if delta > 0:
        return "零样本高于瞎猜基线"
    if delta == 0:
        return "零样本不低于瞎猜基线"
    return "零样本低于瞎猜基线——先核查提示格式与任务定义"


def describe_probe_verdict(result: dict) -> list[str]:
    """探针判定的人话摘要:判定行 + note 原文,供 CLI 直接打印到 stderr。

    判定行如实亮出基座零样本、瞎猜多数类基线与差异三组数字;note 自带
    抽样计数与样本量/不预测微调效果的边界,原文复述不改编。没有可读
    字段的裸记录给缺位句,不编造数字。
    """
    accuracy = result.get("zero_shot_accuracy")
    baseline = result.get("majority_baseline")
    if accuracy is None or baseline is None:
        return ["这份探针记录没有可读的判定内容。"]
    lines = [
        f"可学性探针判定：基座零样本 {accuracy:.0%} vs 瞎猜多数类基线 {baseline:.0%}"
        f"（差异 {result.get('difference', 0):+.0%}）——{probe_verdict_phrase(result)}。"
    ]
    if result.get("note"):
        lines.append(result["note"])
    return lines


def low_baseline_triage_lines(result: dict) -> list[str]:
    """零样本低于瞎猜基线时的核查方向分辨(单一来源):按记录内事实给方向,不认定原因。

    「提示模板问题还是任务定义问题」由可观察事实分流:生成被截断→分数被截断
    压低,先加长度重测(R69 同口径);输出词汇不在开发集标签全集→模型没用任务
    的答案词汇作答,模板方向;全部未截断输出完全相同→没按输入区分作答,模板
    没讲清与输入缺区分信息两方向都在;用任务词汇、按输入作答仍低于瞎猜→更
    像任务定义/标注口径问题。尾行给对号处理映射,不替用户决定。渲染时从已
    存记录现算(echo_triage_lines 先例):早期记录缺 label_vocabulary 时跳过
    词汇检查,其余分辨照常,如实降级不编造。
    """
    delta = result.get("difference")
    if delta is None or delta >= 0:
        return []
    observations = result.get("observations") or []
    if not observations:
        return []
    generated = [obs.get("generated", "") for obs in observations if not obs.get("truncated")]
    truncated_count = sum(1 for obs in observations if obs.get("truncated"))
    lines: list[str] = []
    if truncated_count:
        lines.append(
            f"{truncated_count} 条生成被截断——被截断的输出已按不匹配计，当前分数被截断压低；"
            "先加大 max_new_tokens 重测，再判断是模板还是任务定义的问题。"
        )
    vocabulary = result.get("label_vocabulary")
    unknown_answers = (
        [answer for answer in dict.fromkeys(generated) if answer not in vocabulary]
        if vocabulary is not None
        else []
    )
    if unknown_answers:
        names = "、".join(f"「{answer}」" for answer in unknown_answers)
        lines.append(
            f"基座输出 {names} 不在这份开发集的标签里出现过——模型没有用任务的答案词汇作答，"
            "先核对提示模板是否讲清了按什么口径、用什么词汇作答"
            "（补全式模板与对话型基座不匹配是常见形态）。"
        )
    elif len(generated) >= 2 and len(set(generated)) == 1:
        lines.append(
            f"模型对全部 {len(generated)} 条输入给了同一个输出「{generated[0]}」"
            "——没有按输入区分作答；提示模板没把任务讲清与输入本身缺少区分信息"
            "两种可能都在：先补清指令或换对话式模板重测，仍同答再核对输入是否足以判断。"
        )
    elif generated:
        lines.append(
            "输出用的是任务答案词汇、也按输入区分作答，方向仍低于瞎猜多数类"
            "——更像任务定义或标注口径的问题：先核对类别边界与标注规则"
            "（对照标签问题候选与盲标核验的不一致行）。"
        )
    if lines:
        lines.append(
            "对号处理：模板没讲清就补指令或换模板后重测（不动数据）；"
            "输入缺信息或类别边界不清就补输入字段、澄清标注口径（改任务定义）；"
            "两边都核对过分数仍低，如实在记录里保留低分证据，不硬修。"
        )
    return lines


def candidates_to_csv(candidates: list[dict]) -> bytes:
    """候选表导出为 CSV(带 BOM,Excel 直开);供人工核对的离线清单。"""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["行ID", "数据标签", "基座零样本输出", "你的盲标答案", "证据"])
    for item in candidates:
        writer.writerow(
            [
                item.get("row_id", ""),
                item.get("data_label", ""),
                item.get("base_zero_shot", ""),
                item.get("user_blind_answer") or "",
                item.get("evidence", ""),
            ]
        )
    return buffer.getvalue().encode("utf-8-sig")
