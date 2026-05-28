"""
评测框架核心模块。

支持：
- 多模型对比评测（精调模型 vs 通用模型 baseline）
- 按难度分层统计
- 准确率/召回率/MRR/置信度校准
- 生成对比报告
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DOMAIN_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = DOMAIN_ROOT / "data" / "results"
TEST_DIR = DOMAIN_ROOT / "data" / "test"


@dataclass
class EvalResult:
    """单条评测结果"""
    query: str
    ground_truth: str
    ground_truth_code: str
    predicted_name: str
    predicted_code: str
    predicted_index: Optional[int]
    confidence: float
    difficulty: str
    entity_type: str
    latency_ms: float
    correct: bool = False
    error: str = ""
    seen_in_train: bool = False

    @staticmethod
    def _normalize_code(code: str) -> str:
        return code.replace(" ", "").replace("国药准字", "")

    def is_correct(self) -> bool:
        gt = self._normalize_code(self.ground_truth_code)
        pred = self._normalize_code(self.predicted_code)
        return bool(gt and pred and gt == pred)


@dataclass
class EvalReport:
    """评测报告"""
    model_name: str
    results: list[EvalResult] = field(default_factory=list)
    total_time_ms: float = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct_count(self) -> int:
        return sum(1 for r in self.results if r.correct)

    def accuracy(self, difficulty: str = None, entity_type: str = None, seen: bool = None) -> float:
        filtered = self._filter(difficulty, entity_type, seen)
        if not filtered:
            return 0.0
        return sum(1 for r in filtered if r.correct) / len(filtered)

    def mrr(self, difficulty: str = None) -> float:
        """Mean Reciprocal Rank - 如果 top1 正确则为 1.0"""
        filtered = self._filter(difficulty)
        if not filtered:
            return 0.0
        return sum(1 if r.correct else 0 for r in filtered) / len(filtered)

    def avg_confidence(self, difficulty: str = None) -> float:
        filtered = self._filter(difficulty)
        if not filtered:
            return 0.0
        return sum(float(r.confidence) for r in filtered) / len(filtered)

    def confidence_calibration(self) -> dict:
        """置信度校准：高置信度时的实际准确率"""
        bins = {
            ">=0.9": [r for r in self.results if float(r.confidence) >= 0.9],
            "0.7-0.9": [r for r in self.results if 0.7 <= float(r.confidence) < 0.9],
            "0.5-0.7": [r for r in self.results if 0.5 <= float(r.confidence) < 0.7],
            "<0.5": [r for r in self.results if float(r.confidence) < 0.5],
        }
        calibration = {}
        for bin_name, results in bins.items():
            if results:
                calibration[bin_name] = {
                    "count": len(results),
                    "accuracy": sum(1 for r in results if r.correct) / len(results),
                }
            else:
                calibration[bin_name] = {"count": 0, "accuracy": 0.0}
        return calibration

    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def throughput(self) -> float:
        """条/秒"""
        if self.total_time_ms == 0:
            return 0.0
        return self.total / (self.total_time_ms / 1000)

    def _filter(self, difficulty: str = None, entity_type: str = None, seen: bool = None) -> list[EvalResult]:
        filtered = self.results
        if difficulty:
            filtered = [r for r in filtered if r.difficulty == difficulty]
        if entity_type:
            filtered = [r for r in filtered if r.entity_type == entity_type]
        if seen is not None:
            filtered = [r for r in filtered if r.seen_in_train == seen]
        return filtered

    def summary(self) -> dict:
        seen_results = [r for r in self.results if r.seen_in_train]
        unseen_results = [r for r in self.results if not r.seen_in_train]
        return {
            "model": self.model_name,
            "total": self.total,
            "correct": self.correct_count,
            "overall_accuracy": self.accuracy(),
            "accuracy_by_difficulty": {
                "easy": self.accuracy(difficulty="easy"),
                "medium": self.accuracy(difficulty="medium"),
                "hard": self.accuracy(difficulty="hard"),
            },
            "accuracy_by_type": {
                "drug": self.accuracy(entity_type="drug"),
                "hospital": self.accuracy(entity_type="hospital"),
            },
            "accuracy_by_seen": {
                "seen_count": len(seen_results),
                "seen_accuracy": sum(1 for r in seen_results if r.correct) / len(seen_results) if seen_results else 0,
                "unseen_count": len(unseen_results),
                "unseen_accuracy": sum(1 for r in unseen_results if r.correct) / len(unseen_results) if unseen_results else 0,
            },
            "mrr": self.mrr(),
            "avg_confidence": self.avg_confidence(),
            "confidence_calibration": self.confidence_calibration(),
            "avg_latency_ms": self.avg_latency(),
            "throughput_per_sec": self.throughput(),
            "total_time_sec": self.total_time_ms / 1000,
        }


class BaseModel(ABC):
    """模型接口基类"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def predict(self, query: str, candidates: list[dict]) -> dict:
        """
        输入 query 和候选列表，返回预测结果。
        candidates: [{"name": "xxx", "code": "xxx"}, ...]
        返回: {"standard_name": "xxx", "code": "xxx", "match_index": 1, "confidence": 0.95}
        """
        ...


def load_test_data(path: str = None) -> list[dict]:
    """加载测试集原始数据"""
    path = Path(path) if path else TEST_DIR / "test_raw.json"
    with open(path) as f:
        return json.load(f)


def run_evaluation(model: BaseModel, test_data: list[dict] = None, log_every: int = 50, concurrency: int = 1, train_codes: set[str] | None = None) -> EvalReport:
    """运行完整评测"""
    if test_data is None:
        test_data = load_test_data()

    report = EvalReport(model_name=model.name)
    start = time.time()
    total = len(test_data)
    completed = [0]  # 用列表以便在闭包中修改

    def _eval_one(sample):
        query = sample["query"]
        ground_truth = sample["standard_name"]
        ground_truth_code = sample["code"]
        candidates = [{"name": c["name"], "code": c["code"]} for c in sample["candidates"]]
        seen = train_codes is not None and ground_truth_code in train_codes
        try:
            t0 = time.time()
            pred = model.predict(query, candidates)
            latency = (time.time() - t0) * 1000
            result = EvalResult(
                query=query,
                ground_truth=ground_truth,
                ground_truth_code=ground_truth_code,
                predicted_name=pred.get("standard_name", ""),
                predicted_code=pred.get("code", ""),
                predicted_index=pred.get("match_index"),
                confidence=pred.get("confidence", 0.0),
                difficulty=sample["difficulty"],
                entity_type=sample["entity_type"],
                latency_ms=latency,
                seen_in_train=seen,
            )
            result.correct = result.is_correct()
        except Exception as e:
            result = EvalResult(
                query=query,
                ground_truth=ground_truth,
                ground_truth_code=ground_truth_code,
                predicted_name="",
                predicted_code="",
                predicted_index=None,
                confidence=0.0,
                difficulty=sample["difficulty"],
                entity_type=sample["entity_type"],
                latency_ms=0,
                error=str(e),
                seen_in_train=seen,
            )
        return result

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_eval_one, s): i for i, s in enumerate(test_data)}
            results = [None] * total
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed[0] += 1
                if completed[0] % log_every == 0:
                    elapsed = time.time() - start
                    speed = completed[0] / elapsed
                    eta = (total - completed[0]) / speed if speed > 0 else 0
                    acc_so_far = sum(1 for r in results[:completed[0]] if r and r.correct) / completed[0] * 100
                    print(
                        f"  [{model.name}] {completed[0]}/{total} "
                        f"({completed[0]/total*100:.0f}%) "
                        f"acc={acc_so_far:.1f}% "
                        f"speed={speed:.1f}/s "
                        f"ETA={eta:.0f}s"
                    )
            report.results = [r for r in results if r]
    else:
        for idx, sample in enumerate(test_data):
            result = _eval_one(sample)
            report.results.append(result)
            if (idx + 1) % log_every == 0:
                elapsed = time.time() - start
                speed = (idx + 1) / elapsed
                eta = (total - idx - 1) / speed if speed > 0 else 0
                acc = report.correct_count / len(report.results) * 100
                print(
                    f"  [{model.name}] {idx + 1}/{total} "
                    f"({(idx+1)/total*100:.0f}%) "
                    f"acc={acc:.1f}% "
                    f"speed={speed:.1f}/s "
                    f"ETA={eta:.0f}s"
                )

    report.total_time_ms = (time.time() - start) * 1000
    return report
