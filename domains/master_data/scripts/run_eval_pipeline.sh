#!/bin/bash
# 评测流水线：机构adapter → 产品adapter → LM Studio基线 → qwen3.5-9b → 汇总

set -e
cd /Users/luopeng/Documents/GitHub/4bit-QLoRA-post-training

OUT="/private/tmp/claude-501/-Users-luopeng-Documents-GitHub-4bit-QLoRA-post-training-domains-master-data/afbe87f1-ff95-4ba7-9c8c-bc93360e5048/tasks/pipeline.output"

echo "=== 等待机构 adapter 评测完成 ===" | tee -a "$OUT"
while pgrep -f "eval_adapter_batch.py.*institution.*400" > /dev/null 2>&1; do
    sleep 30
done
echo "机构 adapter 完成" | tee -a "$OUT"

# 检查机构结果是否存在
INST_RESULT=$(ls -t domains/master_data/data/results/eval_adapter_institution_*.json 2>/dev/null | head -1)
if [ -n "$INST_RESULT" ]; then
    echo "机构结果: $INST_RESULT" | tee -a "$OUT"
else
    echo "警告: 未找到机构结果文件" | tee -a "$OUT"
fi

echo "=== 产品 adapter 400条 ===" | tee -a "$OUT"
venv/bin/python domains/master_data/scripts/eval_adapter_batch.py --task product --max-samples 400 2>&1 | tee -a "$OUT"

PROD_RESULT=$(ls -t domains/master_data/data/results/eval_adapter_product_*.json 2>/dev/null | head -1)
if [ -n "$PROD_RESULT" ]; then
    echo "产品结果: $PROD_RESULT" | tee -a "$OUT"
fi

echo "=== 检查 LM Studio ===" | tee -a "$OUT"
if curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo "LM Studio 在线，跑 gemma-4-26b 基线..." | tee -a "$OUT"
    venv/bin/python domains/master_data/eval/evaluate.py --local-model google/gemma-4-26b-a4b --task institution --max-samples 400 --concurrency 2 2>&1 | tee -a "$OUT"
    venv/bin/python domains/master_data/eval/evaluate.py --local-model google/gemma-4-26b-a4b --task product --max-samples 400 --concurrency 2 2>&1 | tee -a "$OUT"
else
    echo "LM Studio 不在线，跳过基线" | tee -a "$OUT"
fi

echo "=== 检查 qwen3.5-9b ===" | tee -a "$OUT"
if curl -s http://localhost:1234/v1/models 2>/dev/null | grep -q "qwen3.5-9b"; then
    echo "qwen3.5-9b 在线，跑评测..." | tee -a "$OUT"
    venv/bin/python domains/master_data/eval/evaluate.py --local-model qwen/qwen3.5-9b --task both --max-samples 400 --concurrency 2 2>&1 | tee -a "$OUT"
else
    echo "qwen3.5-9b 不在线，跳过" | tee -a "$OUT"
fi

echo "=== 全部完成 ===" | tee -a "$OUT"
echo "DONE" >> "$OUT"
