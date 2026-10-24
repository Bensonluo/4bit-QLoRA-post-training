#!/usr/bin/env python3
"""Reproduce the synthetic three-arm SFT mechanism experiment locally.

Requires an existing complete model and a fresh output directory. This is a
small development experiment, not business validation or final acceptance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="Complete existing local HF model directory"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Fresh directory for data, evidence and adapters"
    )
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    if args.steps < 1:
        parser.error("--steps must be positive")
    base = Path(args.model_path).expanduser().resolve()
    if not base.is_dir():
        parser.error("--model-path must be an existing local model directory")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    started = time.monotonic()

    def event(kind, **details):
        value = {"event": kind, "elapsed_seconds": round(time.monotonic() - started, 3), **details}
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(4)
    seed = 42
    from src.utils.platform_utils import get_platform

    device = get_platform().device
    dtype = torch.float32
    # Snapshot this checkout's contract so future runs disclose the actual source.
    source = REPO / "src/data/sft_protocol.py"
    snapshot = out / "sft_protocol_snapshot.py"
    if not snapshot.exists():
        snapshot.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location("r148_protocol_snapshot", snapshot)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    mapping = {"amber": "P4", "birch": "K7", "cobalt": "Q9", "dahlia": "M2"}
    train_notes = [
        "客户在电话中确认此记录。",
        "这是网页表单首次提交的内容。",
        "门店已记录该条请求。",
        "该记录由现场终端提交。",
    ]
    dev_notes = ["这是一条新的邮件受理记录。", "本条资料来自独立客服会话。"]
    rows = {"train": [], "development": []}
    for split, variants in (("train", range(4)), ("development", range(2))):
        for form in ("classification", "json"):
            for signal, route in mapping.items():
                for variant in variants:
                    # Every signal occurs with exactly the same ticket/note
                    # combinations, including across the two output forms.
                    number = (1000 if split == "train" else 9000) + variant
                    ticket = f"TK-{number}"
                    note = train_notes[variant] if split == "train" else dev_notes[variant]
                    instruction = (
                        "读取工单的 signal 字段，按训练示例学得的固定映射输出处理码。"
                        "处理码只允许 K7、M2、P4、Q9；ticket 和备注不决定处理码。"
                    )
                    if form == "classification":
                        instruction += "只输出一个处理码，不要解释、标点、代码块或其他文字。"
                        answer = route
                    else:
                        instruction += "只输出一个 JSON 对象，恰含 route 和 ticket 两个字符串字段；ticket 必须原样复制输入，不要解释或代码块。"
                        answer = json.dumps(
                            {"route": route, "ticket": ticket},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    rows[split].append(
                        {
                            "case_id": f"{split}-{form}-{signal}-{variant}",
                            "group_id": ticket,
                            "form": form,
                            "signal": signal,
                            "route": route,
                            "instruction": instruction,
                            "input": f"ticket: {ticket}\nsignal: {signal}\n备注: {note}",
                            "output": answer,
                        }
                    )
    assert len(rows["train"]) == 32 and len(rows["development"]) == 16
    assert not ({r["input"] for r in rows["train"]} & {r["input"] for r in rows["development"]})
    assert not (
        {r["group_id"] for r in rows["train"]} & {r["group_id"] for r in rows["development"]}
    )
    order = []
    rng = random.Random(seed)
    while len(order) < args.steps:
        indices = list(range(len(rows["train"])))
        rng.shuffle(indices)
        order.extend(indices)
    order = order[: args.steps]
    settings = {
        "synthetic_mechanism_only": True,
        "dataset_version": "v2_counterbalanced_nuisance",
        "known_v1_confound": "The preserved v1 encoded class in ticket-number tens. V2 counterbalances IDs and notes across every signal; no hyperparameter changes.",
        "counterfactual_design": "Each nuisance combination appears under every signal and output form. These are synthetic counterfactual cases, not independent real tickets or held-out business generalization.",
        "task": "Learn four arbitrary signal-to-route associations; emit a code or strict two-field JSON.",
        "mapping_not_in_prompt": mapping,
        "seed": seed,
        "steps": args.steps,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "scheduler": "constant",
        "warmup_steps": 0,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
            "bias": "none",
        },
        "max_training_tokens": 192,
        "padding": "right, next multiple of 8, attention-mask padding labels -100",
        "max_new_tokens": {"classification": 16, "json": 48},
        "decoding": "greedy, no inherited generation-config processors, tokenizer EOS, raw skip_special_tokens output",
        "device": device,
        "dtype": str(dtype),
        "base": str(base),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "script_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(snapshot),
        "sample_order": [rows["train"][i]["case_id"] for i in order],
        "old_holdout_accessed": False,
        "interpretation": "Controlled synthetic format/supervision evidence; not business validity, generalization, or a causal explanation of R146.",
    }
    write(out / "settings.json", settings)
    write(out / "data.json", rows)
    event("setup", device=device, training_rows=32, development_rows=16, steps=args.steps)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    eos = tokenizer.eos_token_id
    assert type(eos) is int and type(tokenizer.pad_token_id) is int
    arms = [protocol.LEGACY_PROTOCOL, protocol.ANSWER_PROTOCOL, protocol.NATIVE_PROTOCOL]
    encoded = {}
    statistics = {}
    for arm in arms:
        encoded[arm] = {}
        statistics[arm] = []
        for split, records in rows.items():
            encoded[arm][split] = []
            for row in records:
                item = protocol.tokenize_sft_record(row, tokenizer, protocol=arm)
                assert len(item["input_ids"]) <= settings["max_training_tokens"]
                prompt = protocol.render_sft_prompt(row, tokenizer, arm)
                prefix = tokenizer(prompt, add_special_tokens=arm != protocol.NATIVE_PROTOCOL)[
                    "input_ids"
                ]
                assert item["input_ids"][: len(prefix)] == prefix, row["case_id"]
                boundary_proof = protocol.tokenize_sft_record(
                    row,
                    tokenizer,
                    protocol=protocol.ANSWER_PROTOCOL if arm == protocol.LEGACY_PROTOCOL else arm,
                )
                assert item["input_ids"] == boundary_proof["input_ids"]
                answer_positions = [
                    i for i, label in enumerate(boundary_proof["labels"]) if label != -100
                ]
                assert answer_positions and boundary_proof["labels"][answer_positions[-1]] == eos
                assert all(item["labels"][i] == item["input_ids"][i] for i in answer_positions)
                if arm != protocol.LEGACY_PROTOCOL:
                    assert all(label == -100 for label in item["labels"][: len(prefix)])
                    assert sum(label == eos for label in item["labels"]) == 1
                encoded[arm][split].append(item)
                statistics[arm].append(
                    {
                        "case_id": row["case_id"],
                        "split": split,
                        "prompt_tokens": len(prefix),
                        "full_tokens": len(item["input_ids"]),
                        "effective_supervised_tokens": sum(
                            label != -100 for label in item["labels"][1:]
                        ),
                        "answer_and_eos_tokens": len(answer_positions),
                        "supervised_eos": sum(label == eos for label in item["labels"]),
                        "post_eos_tokens": len(item["input_ids"]) - answer_positions[-1] - 1,
                        "post_eos_labels": item["labels"][answer_positions[-1] + 1 :],
                        "input_ids": item["input_ids"],
                        "labels": item["labels"],
                    }
                )
    write(out / "supervision.json", statistics)
    event(
        "encoding_verified",
        max_tokens={arm: max(r["full_tokens"] for r in statistics[arm]) for arm in arms},
    )
    results = {
        "settings": settings,
        "status": "running",
        "baseline": {},
        "arms": {},
        "supervision_file": "supervision.json",
    }
    write(out / "results.json", results)

    def synchronize():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    def memory():
        if device == "mps":
            return {
                "allocated_bytes": torch.mps.current_allocated_memory(),
                "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            }
        if device == "cuda":
            return {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        return {}

    def load_model():
        torch.manual_seed(seed)
        load_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        model.generation_config = GenerationConfig()
        synchronize()
        event("model_loaded", seconds=round(time.monotonic() - load_started, 3), **memory())
        return model

    def release(model):
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    def judge(row, text):
        if row["form"] == "classification":
            return text.strip() == row["output"], text.strip() in set(mapping.values())
        try:

            def unique(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("duplicate key")
                    value[key] = item
                return value

            value = json.loads(text, object_pairs_hook=unique)
        except (ValueError, TypeError):
            return False, False
        valid = (
            isinstance(value, dict)
            and set(value) == {"route", "ticket"}
            and all(type(v) is str for v in value.values())
        )
        return valid and value == json.loads(row["output"]), valid

    def evaluate(model, arm, phase):
        model.eval()
        observations = []
        evaluation_started = time.monotonic()
        for row in rows["development"]:
            prompt = protocol.render_sft_prompt(row, tokenizer, arm)
            inputs = tokenizer(
                prompt, add_special_tokens=arm != protocol.NATIVE_PROTOCOL, return_tensors="pt"
            )
            ids = inputs["input_ids"][0].tolist()
            inputs = {key: value.to(device) for key, value in inputs.items()}
            maximum = settings["max_new_tokens"][row["form"]]
            config = GenerationConfig(
                max_new_tokens=maximum,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            question_started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=config)
            synchronize()
            generated = output[0, len(ids) :].tolist()
            text = tokenizer.decode(generated, skip_special_tokens=True)
            ended = bool(generated and generated[-1] == eos)
            truncated = len(generated) >= maximum and not ended
            correct, valid = judge(row, text)
            observation = {
                "case_id": row["case_id"],
                "form": row["form"],
                "expected": row["output"],
                "prompt": prompt,
                "prompt_ids": ids,
                "output": text,
                "generated_ids": generated,
                "generated_tokens": len(generated),
                "eos_token_id": eos,
                "eos_observed": ended,
                "truncated": truncated,
                "finish_reason": "eos" if ended else "length" if truncated else "other",
                "raw_output_matches": correct,
                "format_valid": valid,
                "accepted": bool(correct and ended and not truncated),
                "seconds": round(time.monotonic() - question_started, 6),
            }
            observations.append(observation)
            event(
                "generation",
                phase=phase,
                protocol=arm,
                case_id=row["case_id"],
                eos=ended,
                truncated=truncated,
                accepted=observation["accepted"],
                seconds=observation["seconds"],
            )
        summary = {}
        for form in ("classification", "json", "all"):
            selected = [r for r in observations if form == "all" or r["form"] == form]
            summary[form] = {
                "total": len(selected),
                "eos": sum(r["eos_observed"] for r in selected),
                "truncated": sum(r["truncated"] for r in selected),
                "format_valid": sum(r["format_valid"] for r in selected),
                "accepted": sum(r["accepted"] for r in selected),
                "acceptance_rate": sum(r["accepted"] for r in selected) / len(selected),
            }
        return {
            "summary": summary,
            "rows": observations,
            "seconds": round(time.monotonic() - evaluation_started, 3),
            "memory": memory(),
        }

    model = load_model()
    try:
        for arm in (protocol.LEGACY_PROTOCOL, protocol.NATIVE_PROTOCOL):
            results["baseline"][arm] = evaluate(model, arm, "base")
            write(out / "results.json", results)
            event("baseline_complete", protocol=arm, summary=results["baseline"][arm]["summary"])
    finally:
        model.to("cpu")
        release(model)
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    for arm in arms:
        model = load_model()
        try:
            torch.manual_seed(seed)
            model = get_peft_model(
                model,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    target_modules="all-linear",
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
            trainable = [p for p in model.parameters() if p.requires_grad]
            initialization = hashlib.sha256()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    initialization.update(name.encode())
                    initialization.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"], weight_decay=0.0)
            model.config.use_cache = False
            model.train()
            step_log = []
            training_started = time.monotonic()
            for step, index in enumerate(order, 1):
                row, item = rows["train"][index], encoded[arm]["train"][index]
                length = math.ceil(len(item["input_ids"]) / 8) * 8
                batch = {
                    "input_ids": torch.tensor(
                        [
                            item["input_ids"]
                            + [tokenizer.pad_token_id] * (length - len(item["input_ids"]))
                        ],
                        device=device,
                    ),
                    "attention_mask": torch.tensor(
                        [item["attention_mask"] + [0] * (length - len(item["attention_mask"]))],
                        device=device,
                    ),
                    "labels": torch.tensor(
                        [item["labels"] + [-100] * (length - len(item["labels"]))], device=device
                    ),
                }
                assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss: {arm} step {step}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                synchronize()
                log = {
                    "step": step,
                    "case_id": row["case_id"],
                    "form": row["form"],
                    "loss": float(loss.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(norm),
                    "seconds": round(time.monotonic() - tick, 6),
                    "input_tokens": len(item["input_ids"]),
                    "effective_supervised_tokens": int((batch["labels"][:, 1:] != -100).sum()),
                }
                step_log.append(log)
                event("training_step", protocol=arm, **log)
                if step % 8 == 0:
                    write(out / f"training-{arm}.json", step_log)
            train_seconds = time.monotonic() - training_started
            del optimizer, trainable, batch, loss
            adapter = out / "adapters" / arm
            model.save_pretrained(adapter, safe_serialization=True)
            evaluation = evaluate(model, arm, "trained")
            results["arms"][arm] = {
                "initial_lora_sha256": initialization.hexdigest(),
                "steps": step_log,
                "training_seconds": round(train_seconds, 3),
                "evaluation": evaluation,
                "adapter": str(adapter),
                "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
                "protocol_identity": protocol.protocol_identity(tokenizer, arm),
            }
            write(out / f"training-{arm}.json", step_log)
            write(out / "results.json", results)
            event(
                "arm_complete",
                protocol=arm,
                training_seconds=round(train_seconds, 3),
                summary=evaluation["summary"],
            )
        finally:
            model.to("cpu")
            release(model)
            del model
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
    assert len({r["initial_lora_sha256"] for r in results["arms"].values()}) == 1
    results["status"] = "completed"
    results["elapsed_seconds"] = round(time.monotonic() - started, 3)
    results["cost"] = {
        "wall_seconds": results["elapsed_seconds"],
        "training_seconds": sum(r["training_seconds"] for r in results["arms"].values()),
        "generation_seconds": sum(r["seconds"] for r in results["baseline"].values())
        + sum(r["evaluation"]["seconds"] for r in results["arms"].values()),
        "electricity_kwh": None,
        "reason": "No actual wall-power meter; no invented electricity measurement.",
    }
    write(out / "results.json", results)
    event(
        "completed",
        seconds=results["elapsed_seconds"],
        summaries={k: v["evaluation"]["summary"] for k, v in results["arms"].items()},
    )


if __name__ == "__main__":
    main()
