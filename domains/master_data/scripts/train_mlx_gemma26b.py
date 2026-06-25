#!/usr/bin/env python3
"""Gemma-4-26b MLX 4-bit LoRA 微调脚本。

用法:
    python domains/master_data/scripts/train_mlx_gemma26b.py
    python domains/master_data/scripts/train_mlx_gemma26b.py --model ~/.lmstudio/models/google/gemma-4-26b-a4b
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DOMAIN_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = str(DOMAIN_ROOT / "configs" / "mlx_lora_26b.yaml")
DEFAULT_MLX_DIR = str(DOMAIN_ROOT / "models" / "gemma-4-26b-mlx-4bit")


def convert_to_mlx(source_model: str, mlx_path: Path) -> None:
    """将 HF 格式模型转换为 MLX 4-bit 格式。"""
    if mlx_path.exists() and any(mlx_path.iterdir()):
        print(f"[跳过] MLX 4-bit 模型已存在: {mlx_path}")
        return

    print(f"[1/2] 正在转换 {source_model} -> MLX 4-bit...")
    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path", source_model,
        "--mlx-path", str(mlx_path),
        "-q",
        "--q-bits", "4",
        "--dtype", "bfloat16",
    ]
    subprocess.run(cmd, check=True)
    print("[1/2] 转换完成")


def run_lora_training(config_path: Path) -> None:
    """启动 LoRA 微调。"""
    print("[2/2] 开始 LoRA 微调...")
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]
    subprocess.run(cmd, check=True)
    print("[2/2] 训练完成")


def update_config_model_path(config_path: Path, model_path: str) -> None:
    """更新配置文件中的模型路径。"""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["model"] = model_path

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma-4-26b MLX 4-bit LoRA 微调")
    parser.add_argument(
        "--model",
        default="mlx-community/gemma-4-26b-a4b-it-4bit",
        help="源模型路径或 HuggingFace ID (默认: mlx-community/gemma-4-26b-a4b-it-4bit)",
    )
    parser.add_argument(
        "--mlx-path",
        default=DEFAULT_MLX_DIR,
        help=f"MLX 4-bit 模型保存路径 (默认: {DEFAULT_MLX_DIR})",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"训练配置文件路径 (默认: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    mlx_path = Path(args.mlx_path)
    config_path = Path(args.config)

    # 更新配置中的模型路径
    update_config_model_path(config_path, str(mlx_path))

    # 转换 + 训练
    convert_to_mlx(args.model, mlx_path)
    run_lora_training(config_path)


if __name__ == "__main__":
    main()
