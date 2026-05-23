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

    def is_correct(self) -> bool:
        return self.predicted_code == self.ground_truth_code


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

    def accuracy(self, difficulty: str = None, entity_type: str = None) -> float:
        filtered = self._filter(difficulty, entity_type)
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
        return sum(r.confidence for r in filtered) / len(filtered)

    def confidence_calibration(self) -> dict:
        """置信度校准：高置信度时的实际准确率"""
        bins = {
            ">=0.9": [r for r in self.results if r.confidence >= 0.9],
            "0.7-0.9": [r for r in self.results if 0.7 <= r.confidence < 0.9],
            "0.5-0.7": [r for r in self.results if 0.5 <= r.confidence < 0.7],
            "<0.5": [r for r in self.results if r.confidence < 0.5],
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

    def _filter(self, difficulty: str = None, entity_type: str = None) -> list[EvalResult]:
        filtered = self.results
        if difficulty:
            filtered = [r for r in filtered if r.difficulty == difficulty]
        if entity_type:
            filtered = [r for r in filtered if r.entity_type == entity_type]
        return filtered

    def summary(self) -> dict:
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


def run_evaluation(model: BaseModel, test_data: list[dict] = None) -> EvalReport:
    """运行完整评测"""
    if test_data is None:
        test_data = load_test_data()

    report = EvalReport(model_name=model.name)
    start = time.time()

    for sample in test_data:
        query = sample["query"]
        ground_truth = sample["standard_name"]
        ground_truth_code = sample["code"]
        candidates = [{"name": c["name"], "code": c["code"]} for c in sample["candidates"]]

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
            )

        report.results.append(result)

    report.total_time_ms = (time.time() - start) * 1000
    return report
