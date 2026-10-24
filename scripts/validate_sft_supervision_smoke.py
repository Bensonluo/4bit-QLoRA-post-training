#!/usr/bin/env python3
"""Reproduce the product SFT integration smoke on the synthetic v2 dataset.

Requires a complete local model, the three-arm script's data.json, and a fresh
output directory. Product training and evaluation run without model downloads.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def judge(row, text):
    if row["form"] == "classification":
        return text.strip() == row["output"], text.strip() in {"P4", "K7", "Q9", "M2"}
    try:

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result

        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return False, False
    valid = (
        isinstance(value, dict)
        and set(value) == {"route", "ticket"}
        and all(type(v) is str for v in value.values())
    )
    return valid and value == json.loads(row["output"]), valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="Complete existing local HF model directory"
    )
    parser.add_argument(
        "--data", required=True, help="data.json emitted by validate_sft_supervision.py"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Fresh directory for product smoke evidence"
    )
    args = parser.parse_args()
    BASE = Path(args.model_path).expanduser().resolve()
    source_data = Path(args.data).expanduser().resolve()
    OUT = Path(args.output_dir).expanduser().resolve()
    if not BASE.is_dir():
        parser.error("--model-path must be an existing local model directory")
    data = json.loads(source_data.read_text())
    if len(data["train"]) != 32 or len(data["development"]) != 16:
        parser.error("--data must contain the v2 experiment's 32 train / 16 development records")
    OUT.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_CACHE"] = str(OUT / "cache/datasets")
    started = time.monotonic()

    def event(kind, **kwargs):
        value = {"event": kind, "elapsed_seconds": round(time.monotonic() - started, 3), **kwargs}
        with (OUT / "progress.jsonl").open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    import peft
    import torch
    import transformers
    from transformers import TrainerCallback

    from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
    from src.data.sft_protocol import ANSWER_PROTOCOL, render_sft_prompt, tokenize_sft_record
    from src.training.sft_trainer import SFTTrainer
    from src.workbench.business_evaluation import (
        EvaluationModel,
        EvaluationProtocol,
        LocalEvaluationRuntime,
    )

    torch.set_num_threads(4)
    for split, filename in [("train", "train.json"), ("development", "development.json")]:
        write(
            OUT / filename,
            [{k: row[k] for k in ("instruction", "input", "output")} for row in data[split]],
        )
    model_config = ModelConfig(
        name=str(BASE.resolve()),
        quantization_bits=None,
        trust_remote_code=False,
        use_flash_attention=False,
        max_length=192,
        device_map=None,
        torch_dtype="float32",
    )
    training_config = TrainingConfig(
        output_dir=str(OUT / "adapter"),
        num_epochs=2,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=1,
        gradient_checkpointing=False,
        fp16=False,
        bf16=False,
        max_grad_norm=1.0,
        seed=42,
        optim="adamw_torch",
    )
    lora_config = LoRAConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear", bias="none"
    )
    data_config = DataConfig(
        dataset_name=str(OUT / "train.json"),
        train_file=str(OUT / "train.json"),
        validation_file=str(OUT / "development.json"),
        validation_split=0.0,
        format="alpaca",
        dataset_loader="alpaca",
        sft_protocol=ANSWER_PROTOCOL,
    )
    logging_config = LoggingConfig(
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        log_memory=False,
        register_model=False,
        log_dir=str(OUT / "logs"),
        mlflow_tracking_uri=str(OUT / "unused-mlruns"),
    )
    settings = {
        "kind": "integrated product smoke, not fourth matched controlled arm",
        "synthetic_mechanism_only": True,
        "dataset_version": "v2_counterbalanced_nuisance",
        "old_holdout_accessed": False,
        "script_sha256": sha(__file__),
        "data_sha256": sha(source_data),
        "configs": {
            k: asdict(v)
            for k, v in [
                ("model", model_config),
                ("training", training_config),
                ("lora", lora_config),
                ("data", data_config),
                ("logging", logging_config),
            ]
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "evaluation": "actual LocalEvaluationRuntime(enforce_token_contract=True), product default dtype; same baseline/adapter prompt and limits; raw output, no answer extraction or EOS repair",
        "max_new_tokens": {"classification": 16, "json": 48},
        "limits": "Synthetic development counterfactuals; not held-out business generalization. Product sampler and evaluation precision differ from controlled FP32 three-arm experiment.",
        "product_sources": {
            p: sha(REPO / p)
            for p in [
                "config/base.py",
                "src/training/sft_trainer.py",
                "src/data/loaders.py",
                "src/data/base.py",
                "src/data/sft_protocol.py",
                "src/data/sft_collator.py",
                "src/models/loader.py",
                "src/workbench/business_evaluation.py",
            ]
        },
    }
    write(OUT / "settings.json", settings)
    product = SFTTrainer(model_config, training_config, lora_config, data_config, logging_config)
    product.prepare_model()
    settings["actual_training_dtype"] = str(next(product.model.parameters()).dtype)
    product.prepare_data()
    assert len(product.train_dataset) == 32 and len(product.eval_dataset) == 16
    lookup = {}
    encoded = []
    eos = product.tokenizer.eos_token_id
    for split in ("train", "development"):
        dataset = product.train_dataset if split == "train" else product.eval_dataset
        for row, item in zip(data[split], dataset, strict=True):
            expected = tokenize_sft_record(row, product.tokenizer, protocol=ANSWER_PROTOCOL)
            active = [i for i, m in zip(item["input_ids"], item["attention_mask"]) if m]
            labels = [i for i, m in zip(item["labels"], item["attention_mask"]) if m]
            assert active == expected["input_ids"] and labels == expected["labels"], row["case_id"]
            assert all(
                label == -100 for label, m in zip(item["labels"], item["attention_mask"]) if not m
            )
            prompt = render_sft_prompt(row, product.tokenizer, ANSWER_PROTOCOL)
            prefix = product.tokenizer(prompt)["input_ids"]
            assert active[: len(prefix)] == prefix and all(i == -100 for i in labels[: len(prefix)])
            assert labels[-1] == eos and sum(i == eos for i in labels) == 1
            info = {
                "split": split,
                "case_id": row["case_id"],
                "form": row["form"],
                "prompt_tokens": len(prefix),
                "active_tokens": len(active),
                "effective_supervised_tokens": sum(i != -100 for i in labels[1:]),
                "supervised_eos": 1,
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"],
            }
            encoded.append(info)
            lookup[tuple(active)] = info
    write(OUT / "prepared_supervision.json", encoded)
    product.setup_trainer()
    write(OUT / "training_arguments.json", product.trainer.args.to_dict())
    consumed = []
    original = product.trainer.data_collator

    def observed_collator(features):
        batch = original(features)
        for ids, masks, labels in zip(
            batch["input_ids"].tolist(),
            batch["attention_mask"].tolist(),
            batch["labels"].tolist(),
            strict=True,
        ):
            active = tuple(i for i, m in zip(ids, masks) if m)
            expected = lookup[active]
            assert (
                labels == expected["labels"]
                and ids == expected["input_ids"]
                and masks == expected["attention_mask"]
            )
            consumed.append(
                {
                    k: expected[k]
                    for k in [
                        "split",
                        "case_id",
                        "form",
                        "prompt_tokens",
                        "active_tokens",
                        "effective_supervised_tokens",
                        "supervised_eos",
                    ]
                }
            )
            write(OUT / "consumed_supervision.json", consumed)
            if len(consumed) == 1:
                write(OUT / "first_consumed_batch.json", {k: v.tolist() for k, v in batch.items()})
        return batch

    product.trainer.data_collator = observed_collator
    logs = []

    class EvidenceCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                record = {"step": state.global_step, **logs}
                # Store exact framework logs; do not adjust LR or losses.
                training_logs.append(record)
                write(OUT / "step_logs.json", training_logs)

    training_logs = logs
    product.trainer.add_callback(EvidenceCallback())
    write(OUT / "settings.json", settings)
    event(
        "prepared", training_rows=32, development_rows=16, dtype=settings["actual_training_dtype"]
    )
    tick = time.monotonic()
    train_result = product.train()
    training_seconds = time.monotonic() - tick
    assert product.trainer.state.global_step == 64
    product.trainer.save_state()
    train_consumed = [r for r in consumed if r["split"] == "train"]
    assert len(train_consumed) == 64
    assert {r["case_id"] for r in train_consumed} == {r["case_id"] for r in data["train"]}
    assert all(
        sum(r["case_id"] == row["case_id"] for r in train_consumed) == 2 for row in data["train"]
    )
    loss_logs = [r for r in logs if "loss" in r]
    assert len(loss_logs) == 64 and all(r["learning_rate"] == 2e-4 for r in loss_logs)
    # Product/HF already performed a final development evaluation at step 64.
    metrics = [r for r in product.trainer.state.log_history if "eval_loss" in r][-1]
    assert sum(r["split"] == "development" for r in consumed) == 16
    write(OUT / "consumed_supervision.json", consumed)
    training = {
        "global_step": 64,
        "train_metrics": train_result.metrics,
        "development_teacher_forced_metrics": metrics,
        "training_seconds": training_seconds,
        "actual_train_consumed_rows": len(train_consumed),
        "actual_eval_consumed_rows": len(consumed) - len(train_consumed),
        "adapter_sha256": sha(OUT / "adapter/adapter_model.safetensors"),
    }
    write(OUT / "training_results.json", training)
    event("training_completed", **training)
    product.trainer = None
    product.model = None
    product.tokenizer = None
    del product
    original = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    results = {
        "kind": "synthetic product integrated smoke",
        "training": training,
        "evaluations": {},
    }
    for label, adapter in [("base", None), ("trained", str(OUT / "adapter"))]:
        runtime = LocalEvaluationRuntime(
            EvaluationModel(label, str(BASE.resolve()), adapter_path=adapter),
            enforce_token_contract=True,
        )
        try:
            dtype = str(next(runtime.model.parameters()).dtype)
            observations = []
            for row in data["development"]:
                prompt = render_sft_prompt(row, runtime.tokenizer, ANSWER_PROTOCOL)
                maximum = settings["max_new_tokens"][row["form"]]
                protocol = EvaluationProtocol("classification_exact", max_new_tokens=maximum)
                tick = time.monotonic()
                generation = runtime.generate(prompt, protocol)
                correct, valid = judge(row, generation.text)
                proof = generation.token_proof
                ended = proof["finish_reason"] == "eos"
                item = {
                    "case_id": row["case_id"],
                    "form": row["form"],
                    "expected": row["output"],
                    "prompt": prompt,
                    "output": generation.text,
                    "generated_tokens": generation.generated_tokens,
                    "truncated": generation.truncated,
                    "token_proof": proof,
                    "eos_observed": ended,
                    "raw_output_matches": correct,
                    "format_valid": valid,
                    "accepted": bool(correct and ended and not generation.truncated),
                    "seconds": time.monotonic() - tick,
                }
                observations.append(item)
                event(
                    "generation",
                    model=label,
                    case_id=row["case_id"],
                    accepted=item["accepted"],
                    truncated=item["truncated"],
                )
            summary = {}
            for form in ["classification", "json", "all"]:
                selected = [r for r in observations if form == "all" or r["form"] == form]
                summary[form] = {
                    "total": len(selected),
                    **{
                        key: sum(r[key] for r in selected)
                        for key in ["accepted", "format_valid", "eos_observed", "truncated"]
                    },
                }
            results["evaluations"][label] = {
                "actual_dtype": dtype,
                "rows": observations,
                "summary": summary,
            }
            write(OUT / "results.json", results)
            event("evaluation_completed", model=label, dtype=dtype, summary=summary)
        finally:
            runtime.close()
    base_rows = results["evaluations"]["base"]["rows"]
    trained_rows = results["evaluations"]["trained"]["rows"]
    assert (
        results["evaluations"]["base"]["actual_dtype"]
        == results["evaluations"]["trained"]["actual_dtype"]
    )
    for left, right in zip(base_rows, trained_rows, strict=True):
        assert left["case_id"] == right["case_id"]
        assert left["token_proof"]["prompt_token_ids"] == right["token_proof"]["prompt_token_ids"]
        assert left["token_proof"]["eos_token_ids"] == right["token_proof"]["eos_token_ids"]
    results["status"] = "completed"
    results["wall_seconds"] = time.monotonic() - started
    results["electricity_kwh"] = None
    write(OUT / "results.json", results)
    event("completed", seconds=results["wall_seconds"])


if __name__ == "__main__":
    main()
