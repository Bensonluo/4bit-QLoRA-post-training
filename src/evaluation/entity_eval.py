"""通用实体匹配评测：任意领域 Alpaca 选择题 test 集 → eval_detail 结果。

北极星（通用微调台）的评测闭环（R120）：用户在 Data Wizard 导出的
test.json（供应商、药品、机构、零件……任意「乱写法 → 标准名」领域）上
评自己的 adapter，结果写 ``domains/entity_matching/data/results/``，
点亮「评测结果 / 模型对比」两页。

纯 stdlib——模型加载/生成在 scripts/eval_entity_match.py 懒加载，
本模块只做判分/聚合/落盘，可被单测直接覆盖。结果条目与
domains/medical_entity/eval/report.py 的 save_results 同构（页面只认
这份契约），medical_entity 降级为已验证案例而非评测唯一入口。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

# input 里的编号候选行：「1. 布洛芬片 (Z3)」/「1、布洛芬片」→ 编号 + 名字 + 可选编码
_CANDIDATE_RE = re.compile(r"^\s*(\d+)[.、]\s*(.+?)\s*(?:\(([^)]*)\))?\s*$")
# 通用模板 input 的查询行：「输入实体: 蓝星科技」（半/全角冒号均可）
_QUERY_RE = re.compile(r"输入实体[:：]\s*(.+)")


def normalize_text(text: str) -> str:
    """归一化模型输出：剥代码围栏/包裹引号/结尾标点，压缩空白。

    训练目标输出是紧凑 JSON，但生成端常见 ```json 围栏、结尾「。」与多余
    空白——判分前先归一化，避免把格式噪声误判为答错。
    """
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    s = s.strip().strip('"')
    s = re.sub(r"\s+", " ", s)
    # 标点+空白的混合尾串（如「。 」）一并剥净，而非剥到空格就停
    return s.rstrip("。；;，,！!？?. ").rstrip()


def parse_numbered_candidates(input_text: str) -> list[str]:
    """从 input 的编号列表解出候选名（按出现顺序，1-based 对应模板 match_index）。"""
    names: list[str] = []
    for line in str(input_text).splitlines():
        m = _CANDIDATE_RE.match(line)
        if m:
            names.append(m.group(2).strip())
    return names


def extract_query(input_text: str) -> str:
    """从 input 提取查询实体名；无「输入实体」行返回空串（错误分析表回退用）。"""
    m = _QUERY_RE.search(str(input_text))
    return m.group(1).strip() if m else ""


def _try_json(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def judge(expected: str, predicted: str) -> bool:
    """判分：双方均可解析为 JSON → 比 match_index（语义判分）；否则归一化串相等。

    match_index 是模板的语义答案（选中的候选编号）；index 对即判对——预测
    缺 code 或 code 不一致不推翻（编错码但选对实体的惩罚留给训练，不在
    评测层二次惩罚）。任一侧不是 JSON 时退化为归一化字符串精确匹配，
    保持对非 entity_matching 模板（自由文本答案）的通用性。
    """
    exp = _try_json(normalize_text(expected))
    pred = _try_json(normalize_text(predicted))
    if exp is not None and pred is not None:
        if "match_index" in exp or "match_index" in pred:
            exp_i = _as_int(exp.get("match_index"))
            pred_i = _as_int(pred.get("match_index"))
            return exp_i is not None and exp_i == pred_i
        return exp == pred
    return normalize_text(expected) == normalize_text(predicted)


def _resolve(candidates: list[str], raw_output: str) -> tuple[str, str | None, bool]:
    """(展示名, 编码, 是否成功解析)——match_index → 候选名；解析失败回退原文。"""
    obj = _try_json(normalize_text(raw_output))
    if obj is not None:
        idx = _as_int(obj.get("match_index"))
        if idx is not None and 1 <= idx <= len(candidates):
            code = obj.get("code")
            return candidates[idx - 1], str(code) if code is not None else None, True
        if "name" in obj:  # 容错：模型直呼其名
            return str(obj["name"]), None, True
    return normalize_text(raw_output)[:80], None, False


def build_rows(
    records: list[dict],
    predictions: list[str],
    latencies_ms: list[float],
) -> list[dict]:
    """test 记录 + 生成结果 → per_sample 行（02/03 页错误分析契约字段）。

    三列按最短长度截齐（生成中途失败时保住已产出部分）。
    """
    n = min(len(records), len(predictions), len(latencies_ms))
    rows: list[dict] = []
    for i in range(n):
        rec = records[i]
        input_text = str(rec.get("input", ""))
        expected = str(rec.get("output", ""))
        predicted = predictions[i]
        candidates = parse_numbered_candidates(input_text)
        gt_name, gt_code, _ = _resolve(candidates, expected)
        pred_name, pred_code, _ = _resolve(candidates, predicted)
        metadata = rec.get("metadata") or {}
        rows.append(
            {
                "query": extract_query(input_text) or normalize_text(input_text)[:40],
                "ground_truth": gt_name,
                "ground_truth_code": gt_code,
                "predicted_name": pred_name,
                "predicted_code": pred_code,
                "confidence": None,  # 贪心解码无校准置信度（页面 fmt_num 渲染 —）
                "difficulty": metadata.get("difficulty"),
                "entity_type": metadata.get("entity_type"),
                "correct": judge(expected, predicted),
                "latency_ms": round(float(latencies_ms[i]), 1),
                "error": None,
            }
        )
    return rows


def summarize(model_name: str, rows: list[dict]) -> dict:
    """聚合 per_sample 行 → eval_detail 单模型条目（save_results 同构契约）。"""
    total = len(rows)
    correct = sum(1 for r in rows if r.get("correct"))

    def _group(key: str) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for r in rows:
            k = r.get(key)
            if k:
                buckets.setdefault(str(k), []).append(bool(r.get("correct")))
        return {k: sum(v) / len(v) for k, v in buckets.items()}

    overall = (correct / total) if total else None
    return {
        "model": model_name,
        "total": total,
        "correct": correct,
        "overall_accuracy": overall,
        # 单答案贪心任务：命中即 RR=1，未命中无排名可言 → MRR 恒等于命中率
        "mrr": overall,
        "accuracy_by_difficulty": _group("difficulty"),
        "accuracy_by_type": _group("entity_type"),
        "avg_latency_ms": (sum(r["latency_ms"] for r in rows) / total if total else None),
        "per_sample": rows,
    }


def write_report(summaries: list[dict], results_dir: Path | str) -> Path:
    """落盘 eval_detail_<时间戳>.json（load_eval_data 按 sorted reverse 取最新）。"""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"eval_detail_{timestamp}.json"
    path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
