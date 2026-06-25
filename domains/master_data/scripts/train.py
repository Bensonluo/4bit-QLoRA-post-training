#!/usr/bin/env python3
"""主数据匹配模型训练脚本（messages chat 格式）。

用法:
    python domains/master_data/scripts/train.py --mac-35-4b
    python domains/master_data/scripts/train.py --mac-35-4b --epochs 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import json
import random
from dataclasses import dataclass

import torch
import typer
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

console = Console()
app = typer.Typer(add_completion=False)

DOMAIN_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FILE = str(DOMAIN_ROOT / "data" / "train" / "train.json")
DEFAULT_OUTPUT_DIR = str(DOMAIN_ROOT.parent.parent / "outputs" / "master-data-matching")


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    output_dir: str = DEFAULT_OUTPUT_DIR
    train_file: str = DEFAULT_TRAIN_FILE
    num_epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_length: int = 1024
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    save_steps: int = 100
    logging_steps: int = 10
    bf16: bool = True
    gradient_checkpointing: bool = False
    save_total_limit: int = 3
    seed: int = 42


PRESETS = {
    "mac-35-4b": TrainConfig(
        model_name="Qwen/Qwen3.5-4B",
        batch_size=2,
        gradient_accumulation_steps=4,
        max_length=1024,
        lora_r=32,
        lora_alpha=64,
        num_epochs=2,
    ),
    "mac-8b": TrainConfig(
        model_name="Qwen/Qwen3-8B",
        batch_size=1,
        gradient_accumulation_steps=8,
        max_length=1024,
        lora_r=32,
        lora_alpha=64,
        num_epochs=2,
    ),
    "poc": TrainConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        batch_size=2,
        gradient_accumulation_steps=4,
        max_length=1024,
        lora_r=32,
        lora_alpha=64,
        gradient_checkpointing=True,
        num_epochs=2,
    ),
}


@app.command()
def main(
    mac_35_4b: bool = typer.Option(False, "--mac-35-4b", help="Mac 64GB Qwen3.5-4B"),
    mac_8b: bool = typer.Option(False, "--mac-8b", help="Mac 64GB Qwen3-8B"),
    poc: bool = typer.Option(False, "--poc", help="POC 模式 4B 4-bit"),
    model_name: str = typer.Option(None, "--model-name", "-m", help="覆盖模型名"),
    epochs: int = typer.Option(None, "--epochs", "-e", help="训练轮数"),
    lr: float = typer.Option(None, "--lr", help="学习率"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="输出目录"),
    train_file: str = typer.Option(None, "--train-file", "-d", help="训练数据文件"),
    save_steps: int = typer.Option(None, "--save-steps", help="保存间隔步数"),
    resume_from: str = typer.Option(None, "--resume-from", help="从 checkpoint 恢复"),
):
    """训练主数据匹配模型（机构+产品）"""
    if mac_35_4b:
        preset = "mac-35-4b"
    elif mac_8b:
        preset = "mac-8b"
    elif poc:
        preset = "poc"
    else:
        preset = "mac-35-4b"

    config = PRESETS[preset]

    if model_name:
        config.model_name = model_name
    if epochs is not None:
        config.num_epochs = epochs
    if lr is not None:
        config.learning_rate = lr
    if output_dir:
        config.output_dir = output_dir
    if train_file:
        config.train_file = train_file
    if save_steps is not None:
        config.save_steps = save_steps

    console.print("\n[bold]主数据匹配训练[/bold]")
    console.print(f"  模式: {preset}")
    console.print(f"  模型: {config.model_name}")
    console.print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    console.print(f"  Batch: {config.batch_size} × {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    console.print(f"  Epochs: {config.num_epochs}")
    console.print(f"  数据: {config.train_file}")
    console.print(f"  输出: {config.output_dir}")
    console.print(f"  恢复: {resume_from or '无'}\n")

    if not Path(config.train_file).exists():
        console.print("[red]✗ 训练数据不存在，先运行: python domains/master_data/scripts/generate_data.py[/red]")
        raise typer.Exit(1)

    # Load dataset
    console.print("[cyan]加载数据...[/cyan]")
    with open(config.train_file) as f:
        raw_data = json.load(f)
    console.print(f"  训练样本: {len(raw_data)} 条")

    from datasets import Dataset
    # 拆分验证集（100条）
    rng = random.Random(config.seed)
    rng.shuffle(raw_data)
    val_data = raw_data[:100]
    train_data = raw_data[100:]
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    console.print(f"  训练样本: {len(train_data)} 条, 验证样本: {len(val_data)} 条")

    # Load model and tokenizer
    console.print(f"[cyan]加载模型: {config.model_name}[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    from src.utils.platform_utils import detect_platform
    platform = detect_platform()
    if platform.device == "mps":
        model_kwargs["device_map"] = {"": "mps"}
    elif platform.device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16 if platform.device == "cuda" else False,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy="steps",
        eval_steps=config.save_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    # Train with TRL SFTTrainer (supports messages format natively)
    console.print("[bold green]开始训练...[/bold green]\n")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        max_seq_length=config.max_length,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # Save
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    console.print(f"\n[bold green]✓ 训练完成: {config.output_dir}[/bold green]")


if __name__ == "__main__":
    app()
