# Model Registry Guide — Full Model Lifecycle with MLflow

> Close the loop: after fine-tuning, **merge → register → stage → trace** your models. Every registered version links back to the exact training run that produced it (params + metrics + git commit), so you always know *which* model is in Production and *why* it's better.

---

## 📌 Table of Contents

- [Why a Model Registry](#-why-a-model-registry)
- [Prerequisites](#-prerequisites)
- [The Lifecycle at a Glance](#-the-lifecycle-at-a-glance)
- [Automatic Registration (in-training)](#-automatic-registration-in-training)
- [Manual Registration (standalone)](#-manual-registration-standalone)
- [Managing Stages: Staging → Production](#-managing-stages-staging--production)
- [Lineage: Tracing a Model to Its Training](#-lineage-tracing-a-model-to-its-training)
- [How It Works Internally](#-how-it-works-internally)
- [Troubleshooting](#-troubleshooting)

---

## 💡 Why a Model Registry

Without a registry, fine-tuned models are **anonymous files on disk**. You can't answer:
- Which adapter is the current best?
- What hyperparameters trained model v3?
- How do I roll back if Production breaks?

A Model Registry solves this by giving each trained model a **versioned, staged identity** with **lineage** back to its training run. This is MLOps 101 — and it's the missing half of most "I fine-tuned a model" tutorials.

This project uses **MLflow Model Registry** (you already use MLflow for tracking), storing everything in a local file store — zero ops, fully reproducible.

---

## 🔧 Prerequisites

```bash
# MLflow is already a dependency. Verify:
pip install mlflow    # if somehow missing

# The tracking store lives at ./outputs/mlruns by default (configurable).
```

For the **merge** step you need enough RAM/VRAM to hold the full model in memory briefly:
- 1.7B model → ~4 GB
- 7B model → ~15 GB
- 14B model → ~30 GB

Use `merge_dtype: float16` in ModelConfig if memory is tight (bf16 is default for quality).

---

## 🔄 The Lifecycle at a Glance

```
   Training Run                MLflow Tracking              Model Registry
   ────────────                ────────────────             ──────────────
   train()                     run:<id>
     ├─ logs params  ────────► params{model, lr, ...}
     ├─ logs metrics ───────► metrics{loss, eval_loss}
     └─ save_model()           (lineage link)
          │
          ▼
   merge adapter + base   ─►  log_model  ──────────────►  register_model
   (merger.py)                artifact "model"              name: Qwen3-1.7B-QLoRA
                                                            version: 3
                                                            stage: Staging  ◄── you promote
                                                                          ──► Production
```

Every version points back to `run:<id>`, which holds the params and metrics. That's your lineage.

---

## 🤖 Automatic Registration (in-training)

The simplest path: flip two config flags and the trainer does everything.

### Via LoggingConfig

```python
from config.base import LoggingConfig, ModelConfig, TrainingConfig

logging_config = LoggingConfig(
    use_mlflow=True,              # required: tracking must be on
    register_model=True,          # 🆕 enable auto-registration
    registry_model_name="Qwen3-1.7B-QLoRA",  # optional, defaults to model name
    merge_before_register=True,   # merge adapter into base before logging
    registry_stage="Staging",     # initial stage after registration
)

model_config = ModelConfig(
    name="Qwen/Qwen3-1.7B",
    quantization_bits=4,
    merge_dtype="bfloat16",       # 🆕 precision of the merged model
)
```

Then just train normally — `train()` will, after `save_model()`:

1. **Merge** the LoRA adapter into the base (via `merge_adapter_to_dir`)
2. **Log** the merged model to the active MLflow run as artifact `"model"`
3. **Register** it as a new version under `registry_model_name`
4. **Transition** to `registry_stage` (default Staging)

If registration fails for any reason, the training run is **not** affected — the model is already saved to disk, and you can register manually later.

### When NOT to use auto-registration

- Quick experiments where you don't care about lineage
- Disk space is tight (merged models are full-size; 4-bit adapters are tiny)
- Set `register_model=False` (the default) and nothing happens

---

## 🛠️ Manual Registration (standalone)

Register a model you trained earlier, or merged separately.

### Step 1: Merge (if you only have the adapter)

```bash
python scripts/merge_adapter.py \
    --adapter-dir outputs/sft/run-xxx \
    --output-dir outputs/merged/run-xxx \
    --dtype bfloat16
```

The base model is auto-resolved from `adapter_config.json`. Override with `--base-model-name`.

### Step 2: Register

```bash
python scripts/registry_cli.py register \
    --model-dir outputs/merged/run-xxx \
    --name Qwen3-1.7B-QLoRA \
    --stage Staging
```

This creates a fresh MLflow run, logs the model, registers it, and stages it.

---

## 🎚️ Managing Stages: Staging → Production

MLflow Model Registry uses three stages:

| Stage | Meaning |
|-------|---------|
| **None** | Just registered, not yet evaluated |
| **Staging** | Passed eval, ready for promotion |
| **Production** | The live model serving traffic |
| **Archived** | Retired, kept for rollback |

### List all versions

```bash
python scripts/registry_cli.py list
# or filter:
python scripts/registry_cli.py list --model-name Qwen3-1.7B-QLoRA
```

Output:
```
Model Registry
┌─────────────────────┬─────────┬─────────────┬──────────┬──────────┐
│ Name                │ Version │ Stage       │ Run ID   │ Status   │
├─────────────────────┼─────────┼─────────────┼──────────┼──────────┤
│ Qwen3-1.7B-QLoRA    │ 1       │ Archived    │ a1b2c3d4 │ READY    │
│ Qwen3-1.7B-QLoRA    │ 2       │ Staging     │ e5f6g7h8 │ READY    │
│ Qwen3-1.7B-QLoRA    │ 3       │ Production  │ i9j0k1l2 │ READY    │
└─────────────────────┴─────────┴─────────────┴──────────┴──────────┘
```

### Promote to Production

```bash
python scripts/registry_cli.py transition \
    --model-name Qwen3-1.7B-QLoRA \
    --version 3 \
    --stage Production
```

### Roll back

```bash
# Demote current Production to Archived, restore v2
python scripts/registry_cli.py transition --model-name Qwen3-1.7B-QLoRA --version 3 --stage Archived
python scripts/registry_cli.py transition --model-name Qwen3-1.7B-QLoRA --version 2 --stage Production
```

---

## 🔍 Lineage: Tracing a Model to Its Training

The killer feature. Given any model version, see exactly how it was trained:

```bash
python scripts/registry_cli.py info --model-name Qwen3-1.7B-QLoRA --version 3
```

Output:
```
┌─────────────────────────────────────────────┐
│ Qwen3-1.7B-QLoRA v3                         │
│ Stage: Production                           │
│ Run ID: i9j0k1l2                            │
│ Source: runs:/i9j0k1l2/model                │
└─────────────────────────────────────────────┘

Lineage Run Parameters:
  data.dataset_name          yahma/alpaca-cleaned
  lora.r                     16
  model.name                 Qwen/Qwen3-1.7B
  training.batch_size        1
  training.learning_rate     0.0002

Lineage Run Metrics:
  eval_loss                  1.2345
  train_loss                 0.8901
  train_runtime              4567.8
```

Now you can answer: *"Why is v3 better than v2?"* → compare their run metrics side by side.

---

## ⚙️ How It Works Internally

Three new pieces collaborate:

```
┌─────────────────────────────────────────────────────────┐
│  src/models/merger.py                                   │
│    merge_adapter_to_dir() — disk-based LoRA merge       │
│    (AutoPeftModelForCausalLM + merge_and_unload)        │
└────────────────────┬────────────────────────────────────┘
                     │ produces merged model dir
                     ▼
┌─────────────────────────────────────────────────────────┐
│  src/tracking/registry.py                               │
│    register_trained_model() — orchestrates the flow:    │
│      merge → tracker.log_model → tracker.register_model │
│              → tracker.transition_model_stage           │
└────────────────────┬────────────────────────────────────┘
                     │ called by
                     ▼
┌─────────────────────────────────────────────────────────┐
│  sft_trainer.train() / dpo_trainer.train()              │
│    after save_model() → register_trained_model(...)     │
│    (no-op unless register_model=True)                   │
└─────────────────────────────────────────────────────────┘
```

The tracker (`src/tracking/mlflow_tracker.py`) gained 5 methods that map to MLflow's API:
- `log_model` → `mlflow.transformers.log_model`
- `register_model` → `mlflow.register_model`
- `transition_model_stage` → `MlflowClient.transition_model_version_stage`
- `search_model_versions` → `MlflowClient.search_model_versions`
- `log_artifacts` → `mlflow.log_artifacts`

### Bug fixes shipped alongside

This work also fixed three pre-existing tracking issues:
1. **DPO callback not mounted** — `MLflowTrainCallback` was instantiated but never passed to `TRLDPOTrainer`, so DPO step metrics never reached MLflow. Now mounted.
2. **HF native MLflowCallback double-write** — `_get_report_to()` added `"mlflow"`, activating HF's built-in callback alongside our custom one. Removed; we use only our callback.
3. **No model lineage** — `save_model()` wrote the adapter but never linked it to the run. `log_model` now creates the link.

---

## 🐛 Troubleshooting

### "register_model=True but MLflow is not active"

`register_model` requires `use_mlflow=True`. The registry needs a tracking store to log to. Enable both.

### Merge OOMs on large models

Merging loads the full un-quantized model into memory. For 14B+ models:
- Use `merge_dtype: "float16"` (halves memory vs float32)
- Merge on a machine with enough RAM (CPU merge is fine, just slower)
- Or skip merge: set `merge_before_register=False` to register the adapter directly (loading later needs the base model)

### `AutoPeftModelForCausalLM` can't find base model

The adapter's `adapter_config.json` records the base model id. If it's wrong or inaccessible:
```bash
python scripts/merge_adapter.py --adapter-dir ... --base-model-name Qwen/Qwen3-1.7B
```

### Registration fails but training succeeded

Registration errors are **caught and logged** — they never fail the training run (the model is already on disk). Re-run registration manually:
```bash
python scripts/registry_cli.py register --model-dir outputs/merged/run-xxx --name <name>
```

### "model name contains '/'"

HF model ids contain slashes (e.g. `Qwen/Qwen3-1.7B`), which are invalid in registry names. The code auto-sanitizes by replacing `/` with `-`. If you set `registry_model_name` manually, use a slash-free name.

### Moving to a production MLflow server

This project uses a local file store (`./outputs/mlruns`). For team collaboration, point `mlflow_tracking_uri` at a real MLflow server:

```python
LoggingConfig(mlflow_tracking_uri="http://mlflow.your-company.com:5000")
```

Everything else works identically.

---

## 📚 Further Reading

- [MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html)
- [PEFT merge_and_unload](https://huggingface.co/docs/peft/developer_guides/lora#merge-lora-weights-into-the-base-model)
- [mlflow.transformers flavor](https://mlflow.org/docs/latest/models.html#transformers-flavor)
