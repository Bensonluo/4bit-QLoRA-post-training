#!/usr/bin/env python3
"""医疗实体匹配模型训练脚本。

用法:
    python scripts/train_medical_entity.py --mac      # Mac Apple Silicon
    python scripts/train_medical_entity.py --poc      # 8GB GPU POC (推荐)
    python scripts/train_medical_entity.py             # 24GB GPU 完整训练
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from config.domains.medical_entity import PRESETS
from src.training import run_sft_training
from src.utils.logging import console

app = typer.Typer(add_completion=False)


@app.command()
def main(
    mac: bool = typer.Option(False, "--mac", help="Mac 64GB (Qwen3-14B, bf16)"),
    mac_8b: bool = typer.Option(False, "--mac-8b", help="Mac 64GB (Qwen3-8B, 1 epoch 快速验证)"),
    mac_2b: bool = typer.Option(False, "--mac-2b", help="Mac 64GB (Qwen3.5-2B)"),
    mac_1b: bool = typer.Option(False, "--mac-1b", help="Mac 64GB (Qwen3-1.7B, 最快)"),
    mac_small: bool = typer.Option(False, "--mac-small", help="Mac 64GB (Qwen3-4B, 对比用)"),
    poc: bool = typer.Option(False, "--poc", help="POC 模式 (4B, 8GB VRAM)"),
    model_name: Optional[str] = typer.Option(None, "--model-name", "-m", help="覆盖模型名"),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="训练轮数（默认用 preset 配置）"),
    lr: Optional[float] = typer.Option(None, "--lr", help="学习率（默认用 preset 配置）"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="输出目录"),
    resume_from: Optional[str] = typer.Option(None, "--resume-from", help="从 checkpoint 恢复训练"),
):
    """训练医疗实体匹配模型"""

    if mac:
        preset = "mac"
    elif mac_8b:
        preset = "mac-8b"
    elif mac_2b:
        preset = "mac-2b"
    elif mac_1b:
        preset = "mac-1b"
    elif mac_small:
        preset = "mac-small"
    elif poc:
        preset = "poc"
    else:
        preset = "full"

    config = PRESETS[preset]

    if model_name:
        config.model.name = model_name
    if epochs is not None:
        config.training.num_epochs = epochs
    if lr is not None:
        config.training.learning_rate = lr
    if output_dir:
        config.training.output_dir = output_dir

    console.print(Panel.fit(
        f"[bold]医疗实体匹配训练[/bold]\n\n"
        f"模式: {preset}\n"
        f"模型: {config.model.name}\n"
        f"量化: {config.model.quantization_bits or '无'}\n"
        f"LoRA r={config.lora.r}\n"
        f"Batch: {config.training.effective_batch_size}\n"
        f"Epochs: {config.training.num_epochs}\n"
        f"数据: {config.data.train_file}\n"
        f"恢复: {resume_from or '无'}",
        border_style="green",
    ))

    # 检查数据
    if not Path(config.data.train_file).exists():
        console.print("[red]✗ 训练数据不存在，先运行: python -m domains.medical_entity.prepare_data[/red]")
        raise typer.Exit(1)

    console.print("\n[bold green]开始训练...[/bold green]\n")
    try:
        run_sft_training(
            model_config=config.model,
            training_config=config.training,
            lora_config=config.lora,
            data_config=config.data,
            logging_config=config.logging,
            resume_from_checkpoint=resume_from,
        )
        console.print(f"\n[bold green]✓ 训练完成: {config.training.output_dir}[/bold green]")
    except Exception as e:
        console.print(f"\n[red]✗ 训练失败: {e}[/red]")
        raise


if __name__ == "__main__":
    app()
