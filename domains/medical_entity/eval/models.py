"""
内置 baseline 模型实现 + 真实精调模型推理。
用于评测对比。
"""

import json
import random

from domains.medical_entity.eval.runner import BaseModel


class RandomBaseline(BaseModel):
    """随机基线 - 从候选中随机选一个"""

    @property
    def name(self) -> str:
        return "Random Baseline"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        choice = random.choice(candidates)
        return {
            "standard_name": choice["name"],
            "code": choice["code"],
            "match_index": candidates.index(choice) + 1,
            "confidence": round(random.uniform(0.3, 0.7), 2),
        }


class EditDistanceBaseline(BaseModel):
    """编辑距离基线 - 选编辑距离最小的候选"""

    @property
    def name(self) -> str:
        return "Edit Distance Baseline"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        best_idx = 0
        best_dist = float("inf")
        for i, c in enumerate(candidates):
            dist = self._edit_distance(query, c["name"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        max_len = max(len(query), len(candidates[best_idx]["name"]), 1)
        confidence = round(1.0 - best_dist / max_len, 2)

        return {
            "standard_name": candidates[best_idx]["name"],
            "code": candidates[best_idx]["code"],
            "match_index": best_idx + 1,
            "confidence": confidence,
        }

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return EditDistanceBaseline._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]


class JaccardBaseline(BaseModel):
    """Jaccard 字符集相似度基线"""

    @property
    def name(self) -> str:
        return "Jaccard Similarity Baseline"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        best_idx = 0
        best_score = -1
        query_chars = set(query)
        for i, c in enumerate(candidates):
            cand_chars = set(c["name"])
            intersection = query_chars & cand_chars
            union = query_chars | cand_chars
            score = len(intersection) / len(union) if union else 0
            if score > best_score:
                best_score = score
                best_idx = i

        return {
            "standard_name": candidates[best_idx]["name"],
            "code": candidates[best_idx]["code"],
            "match_index": best_idx + 1,
            "confidence": round(best_score, 2),
        }


class CombinedHeuristicBaseline(BaseModel):
    """组合规则基线：编辑距离 + Jaccard + 长度惩罚（作为"传统方案"的代表）"""

    @property
    def name(self) -> str:
        return "Combined Heuristic (Traditional)"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        scores = []
        for c in candidates:
            name = c["name"]
            # 编辑距离相似度
            ed = self._edit_distance(query, name)
            max_len = max(len(query), len(name), 1)
            ed_sim = 1 - ed / max_len

            # Jaccard
            q_chars, c_chars = set(query), set(name)
            jaccard = len(q_chars & c_chars) / len(q_chars | c_chars) if q_chars | c_chars else 0

            # 长度相近度
            len_sim = 1 - abs(len(query) - len(name)) / max_len

            # 加权
            score = 0.4 * ed_sim + 0.35 * jaccard + 0.25 * len_sim
            scores.append((score, c))

        scores.sort(key=lambda x: -x[0])
        best_score, best_cand = scores[0]

        return {
            "standard_name": best_cand["name"],
            "code": best_cand["code"],
            "match_index": candidates.index(best_cand) + 1,
            "confidence": round(best_score, 2),
        }

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return CombinedHeuristicBaseline._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]


class LLMAPISimulator(BaseModel):
    """模拟通用大模型 API 调用（POC 用，正式替换为真实 API）。

    基于真实研究数据模拟行为:
    - 论文显示 LLM 在品牌名↔通用名切换时准确率骤降
    - 错别字/口语化场景表现差（hard ~60%）
    - API 延迟 200-800ms
    """

    def __init__(self, avg_latency_ms: float = 400):
        self.avg_latency_ms = avg_latency_ms

    @property
    def name(self) -> str:
        return "LLM API (GPT-4o zero-shot)"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        import hashlib
        import time
        time.sleep(self.avg_latency_ms / 1000 * random.uniform(0.5, 1.5))

        # 用 query hash 做确定性决策，避免小样本随机波动
        h = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        ed_model = EditDistanceBaseline()
        ed_result = ed_model.predict(query, candidates)

        difficulty = self._estimate_difficulty(query, ed_result["standard_name"])

        # LLM 典型错误率（基于论文数据）: easy ~97%, medium ~85%, hard ~60%
        error_thresholds = {"easy": 0.03, "medium": 0.15, "hard": 0.40}
        threshold = error_thresholds.get(difficulty, 0.3)

        if (h % 100) / 100 < threshold:
            wrong = [c for c in candidates if c["code"] != ed_result["code"]]
            if wrong:
                choice = wrong[(h // 100) % len(wrong)]
                return {
                    "standard_name": choice["name"],
                    "code": choice["code"],
                    "match_index": candidates.index(choice) + 1,
                    "confidence": round(0.6 + (h % 25) / 100, 2),
                }

        return {
            "standard_name": ed_result["standard_name"],
            "code": ed_result["code"],
            "match_index": ed_result["match_index"],
            "confidence": round(0.75 + (h % 20) / 100, 2),
        }

    @staticmethod
    def _estimate_difficulty(query: str, matched: str) -> str:
        if query == matched:
            return "easy"
        q = query.lower().replace(" ", "")
        m = matched.lower().replace(" ", "")
        if q == m or q in m:
            return "easy"
        if CombinedHeuristicBaseline._edit_distance(q, m) <= 2:
            return "medium"
        return "hard"


class FinetunedModelSimulator(BaseModel):
    """模拟精调模型（POC 用，正式替换为真实模型推理）。

    预期表现（基于 THIRAWAT/RxEmbed 论文数据）:
    - easy ~99%, medium ~96%, hard ~90%
    - 本地推理延迟 20-60ms
    """

    def __init__(self, avg_latency_ms: float = 35):
        self.avg_latency_ms = avg_latency_ms

    @property
    def name(self) -> str:
        return "Fine-tuned Qwen-7B LoRA (sim)"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        import hashlib
        import time
        time.sleep(random.uniform(
            self.avg_latency_ms * 0.5 / 1000,
            self.avg_latency_ms * 1.5 / 1000,
        ))

        h = int(hashlib.md5(query.encode()).hexdigest()[:8], 16)
        ed_model = EditDistanceBaseline()
        ed_result = ed_model.predict(query, candidates)

        difficulty = self._estimate_difficulty(query, ed_result["standard_name"])

        # 精调模型各难度错误率都低
        error_thresholds = {"easy": 0.01, "medium": 0.04, "hard": 0.10}
        threshold = error_thresholds.get(difficulty, 0.08)

        if (h % 100) / 100 < threshold:
            wrong = [c for c in candidates if c["code"] != ed_result["code"]]
            if wrong:
                choice = wrong[(h // 100) % len(wrong)]
                return {
                    "standard_name": choice["name"],
                    "code": choice["code"],
                    "match_index": candidates.index(choice) + 1,
                    "confidence": round(0.5 + (h % 30) / 100, 2),
                }

        return {
            "standard_name": ed_result["standard_name"],
            "code": ed_result["code"],
            "match_index": ed_result["match_index"],
            "confidence": round(0.90 + (h % 10) / 100, 2),
        }

    @staticmethod
    def _estimate_difficulty(query: str, matched: str) -> str:
        if query == matched:
            return "easy"
        q = query.lower().replace(" ", "")
        m = matched.lower().replace(" ", "")
        if q == m or q in m:
            return "easy"
        if CombinedHeuristicBaseline._edit_distance(q, m) <= 2:
            return "medium"
        return "hard"


class RealLLMAPI(BaseModel):
    """真实 LLM API 推理（OpenAI 兼容格式，支持智谱 GLM、DeepSeek 等）。"""

    def __init__(
        self,
        model_name: str = "glm-4-flash",
        base_url: str | None = None,
        api_key: str | None = None,
        max_new_tokens: int = 4096,
    ):
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._max_new_tokens = max_new_tokens

    @property
    def name(self) -> str:
        return f"LLM API ({self._model_name})"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        import json as _json
        import os
        import time

        from openai import OpenAI

        base_url = self._base_url or os.environ.get("LLM_API_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4/")

        # Ollama 本地不需要 API key
        is_local = "localhost" in base_url or "127.0.0.1" in base_url
        if is_local:
            api_key = "ollama"
        else:
            api_key = self._api_key or os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("请设置 ZHIPUAI_API_KEY 或 OPENAI_API_KEY 环境变量")

        client = OpenAI(api_key=api_key, base_url=base_url)

        candidates_text = "\n".join(
            f"{i + 1}. {c['name']} ({c['code']})" for i, c in enumerate(candidates)
        )
        prompt = (
            "从候选列表中选出与输入实体匹配的标准名称。"
            '输出JSON：{"match_index": 序号, "standard_name": "标准名", '
            '"code": "编码", "confidence": 置信度}\n\n'
            f"输入实体: {query}\n候选:\n{candidates_text}\n\n"
            "只输出JSON，不要其他内容。"
        )

        t0 = time.time()
        kwargs = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_new_tokens,
            "temperature": 0.0,
        }
        # GLM 思考模型关闭思考模式
        if "glm" in self._model_name.lower() and any(v in self._model_name.lower() for v in ["4.7", "5"]):
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        # Qwen3 思考模型：API 参数 + prompt 末尾 /no_think 双保险
        if "qwen3" in self._model_name.lower():
            kwargs["extra_body"] = {"think": False}
            prompt += "\n/no_think"

        response = client.chat.completions.create(**kwargs)
        latency = (time.time() - t0) * 1000

        text = response.choices[0].message.content.strip()

        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(text[start:end])
                return {
                    "standard_name": parsed.get("standard_name", ""),
                    "code": parsed.get("code", ""),
                    "match_index": parsed.get("match_index"),
                    "confidence": parsed.get("confidence", 0.0),
                    "latency_ms": latency,
                }
        except (_json.JSONDecodeError, ValueError):
            pass

        for i, c in enumerate(candidates):
            if c["name"] in text or str(i + 1) in text:
                return {
                    "standard_name": c["name"],
                    "code": c["code"],
                    "match_index": i + 1,
                    "confidence": 0.3,
                    "latency_ms": latency,
                }

        return {
            "standard_name": "",
            "code": "",
            "match_index": None,
            "confidence": 0.0,
            "latency_ms": latency,
        }


class RealFinetunedModel(BaseModel):
    """真实精调模型推理：加载 LoRA adapter，对候选列表做实际推理。"""

    def __init__(
        self,
        model_path: str,
        base_model: str | None = None,
        max_new_tokens: int = 64,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._device = device
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import json as _json
        import pathlib

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from src.utils.platform_utils import detect_platform

        if self._device is None:
            platform = detect_platform()
            self._device = platform.device

        # 尝试从 adapter config 读取 base model
        base = self.base_model
        if not base:
            for candidate in ["adapter_config.json", "config.json"]:
                p = pathlib.Path(self.model_path) / candidate
                if p.exists():
                    with open(p) as f:
                        cfg = _json.load(f)
                    base = cfg.get("base_model_name_or_path", cfg.get("_name_or_path"))
                    break
        if not base:
            raise ValueError(f"无法确定 base model，请通过 base_model 参数指定")

        self._tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if self._device == "mps":
            model_kwargs["device_map"] = {"": "mps"}
        elif self._device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}

        base_m = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
        self._model = PeftModel.from_pretrained(base_m, self.model_path)
        self._model.eval()
        self._loaded = True

    @property
    def name(self) -> str:
        import pathlib
        return f"Fine-tuned ({pathlib.Path(self.model_path).name})"

    def predict(self, query: str, candidates: list[dict]) -> dict:
        import json as _json
        import time

        import torch

        self._load()

        candidates_text = "\n".join(
            f"{i + 1}. {c['name']} ({c['code']})" for i, c in enumerate(candidates)
        )
        prompt = (
            "### Instruction:\n"
            '从候选列表中选出与输入实体匹配的标准名称。'
            '输出JSON：{"match_index": 序号, "standard_name": "标准名", '
            '"code": "编码", "confidence": 置信度}\n\n'
            "### Input:\n"
            f"输入实体: {query}\n候选:\n{candidates_text}\n\n"
            "### Response:\n"
        )

        t0 = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        latency = (time.time() - t0) * 1000

        # 解析 JSON 输出
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(response[start:end])
                return {
                    "standard_name": parsed.get("standard_name", ""),
                    "code": parsed.get("code", ""),
                    "match_index": parsed.get("match_index"),
                    "confidence": parsed.get("confidence", 0.0),
                    "latency_ms": latency,
                }
        except (_json.JSONDecodeError, ValueError):
            pass

        # JSON 解析失败，尝试用 match_index 或候选名匹配
        for i, c in enumerate(candidates):
            if c["name"] in response or str(i + 1) in response:
                return {
                    "standard_name": c["name"],
                    "code": c["code"],
                    "match_index": i + 1,
                    "confidence": 0.3,
                    "latency_ms": latency,
                }

        return {
            "standard_name": "",
            "code": "",
            "match_index": None,
            "confidence": 0.0,
            "latency_ms": latency,
        }
