"""Utility modules for QLoRA post-training."""

from src.utils.execution import (
    RemoteExecutor,
    check_remote_connection,
    execute_on_remote,
    sync_from_remote,
    train_on_remote,
)
from src.utils.logging import (
    console,
    log_gpu_memory,
    log_metrics,
    print_table,
    print_training_summary,
    setup_logging,
    setup_tensorboard,
    setup_wandb,
)
from src.utils.memory import (
    check_remote_gpu,
    clear_cache,
    estimate_model_vram,
    get_vram_usage,
    optimize_memory,
    print_vram_usage,
)
from src.utils.platform_utils import get_platform, detect_platform, recommend_settings
from src.utils.seed import get_seed, set_seed

__all__ = [
    # Platform
    "get_platform",
    "detect_platform",
    "recommend_settings",
    # Memory
    "get_vram_usage",
    "print_vram_usage",
    "clear_cache",
    "optimize_memory",
    "check_remote_gpu",
    "estimate_model_vram",
    # Seed
    "set_seed",
    "get_seed",
    # Logging
    "setup_logging",
    "setup_wandb",
    "setup_tensorboard",
    "log_metrics",
    "log_gpu_memory",
    "console",
    "print_table",
    "print_training_summary",
    # Remote execution
    "execute_on_remote",
    "train_on_remote",
    "sync_from_remote",
    "check_remote_connection",
    "RemoteExecutor",
]
