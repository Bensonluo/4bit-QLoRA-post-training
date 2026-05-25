"""医疗实体匹配 SFT 训练配置预设。

用法:
    python scripts/train_medical_entity.py --mac    # Mac Apple Silicon
    python scripts/train_medical_entity.py --poc    # 8GB GPU POC
    python scripts/train_medical_entity.py           # 24GB GPU 完整训练
"""

from config.base import DataConfig, LoRAConfig, LoggingConfig, ModelConfig, TrainingConfig
from config.sft import SFTConfig

DOMAIN_ROOT = "domains/medical_entity"

# ── Mac Apple Silicon (64GB) ── Qwen3-14B bf16 直加载，与 4B POC 对比
MEDICAL_ENTITY_MAC_CONFIG = SFTConfig(
    model=ModelConfig(
        name="Qwen/Qwen3-14B",
        quantization_bits=None,
        max_length=1024,
        torch_dtype="bfloat16",
        use_flash_attention=False,
    ),
    lora=LoRAConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    training=TrainingConfig(
        output_dir="./outputs/medical-entity-mac",
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        gradient_checkpointing=False,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        seed=42,
    ),
    data=DataConfig(
        dataset_name=f"{DOMAIN_ROOT}/data/train/train.json",
        format="alpaca",
        validation_split=0.0,
        train_file=f"{DOMAIN_ROOT}/data/train/train.json",
        validation_file=f"{DOMAIN_ROOT}/data/val/val.json",
    ),
    logging=LoggingConfig(
        use_wandb=False,
        use_tensorboard=True,
        log_memory=True,
        use_mlflow=True,
        mlflow_tracking_uri="./outputs/mlruns",
        mlflow_experiment_name="medical-entity-matching",
    ),
)


# ── POC (8GB GPU, Qwen3-4B 4-bit) ── 推荐
MEDICAL_ENTITY_POC_CONFIG = SFTConfig(
    model=ModelConfig(
        name="Qwen/Qwen3-4B-Instruct-2507",
        quantization_bits=4,
        max_length=1024,
        torch_dtype="bfloat16",
        use_flash_attention=False,
    ),
    lora=LoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    training=TrainingConfig(
        output_dir="./outputs/medical-entity-poc",
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        seed=42,
    ),
    data=DataConfig(
        dataset_name=f"{DOMAIN_ROOT}/data/train/train.json",
        format="alpaca",
        validation_split=0.0,
        train_file=f"{DOMAIN_ROOT}/data/train/train.json",
        validation_file=f"{DOMAIN_ROOT}/data/val/val.json",
    ),
    logging=LoggingConfig(use_wandb=False, use_tensorboard=True, log_memory=True),
)

# ── 完整训练 (24GB GPU, Qwen3-8B 4-bit) ──
MEDICAL_ENTITY_FULL_CONFIG = SFTConfig(
    model=ModelConfig(
        name="Qwen/Qwen3-8B",
        quantization_bits=4,
        max_length=2048,
        torch_dtype="bfloat16",
        use_flash_attention=False,
    ),
    lora=LoRAConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    training=TrainingConfig(
        output_dir="./outputs/medical-entity-full",
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        seed=42,
    ),
    data=DataConfig(
        dataset_name=f"{DOMAIN_ROOT}/data/train/train.json",
        format="alpaca",
        validation_split=0.0,
        train_file=f"{DOMAIN_ROOT}/data/train/train.json",
        validation_file=f"{DOMAIN_ROOT}/data/val/val.json",
    ),
    logging=LoggingConfig(use_wandb=False, use_tensorboard=True, log_memory=True),
)

# ── Mac 8B (64GB) ── Qwen3-8B bf16，快速验证全流程
MEDICAL_ENTITY_MAC_8B_CONFIG = SFTConfig(
    model=ModelConfig(
        name="Qwen/Qwen3-8B",
        quantization_bits=None,
        max_length=1024,
        torch_dtype="bfloat16",
        use_flash_attention=False,
    ),
    lora=LoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    training=TrainingConfig(
        output_dir="./outputs/medical-entity-mac-8b",
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        gradient_checkpointing=False,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        seed=42,
    ),
    data=DataConfig(
        dataset_name=f"{DOMAIN_ROOT}/data/train/train.json",
        format="alpaca",
        validation_split=0.0,
        train_file=f"{DOMAIN_ROOT}/data/train/train.json",
        validation_file=f"{DOMAIN_ROOT}/data/val/val.json",
    ),
    logging=LoggingConfig(
        use_wandb=False,
        use_tensorboard=True,
        log_memory=True,
        use_mlflow=True,
        mlflow_tracking_uri="./outputs/mlruns",
        mlflow_experiment_name="medical-entity-matching",
    ),
)

# ── Mac Small (64GB, 对比用) ── Qwen3-4B bf16，与 14B 对比
MEDICAL_ENTITY_MAC_SMALL_CONFIG = SFTConfig(
    model=ModelConfig(
        name="Qwen/Qwen3-4B-Instruct-2507",
        quantization_bits=None,
        max_length=1024,
        torch_dtype="bfloat16",
        use_flash_attention=False,
    ),
    lora=LoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
    training=TrainingConfig(
        output_dir="./outputs/medical-entity-mac-small",
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        seed=42,
    ),
    data=DataConfig(
        dataset_name=f"{DOMAIN_ROOT}/data/train/train.json",
        format="alpaca",
        validation_split=0.0,
        train_file=f"{DOMAIN_ROOT}/data/train/train.json",
        validation_file=f"{DOMAIN_ROOT}/data/val/val.json",
    ),
    logging=LoggingConfig(use_wandb=False, use_tensorboard=True, log_memory=True),
)

# 预设映射
PRESETS = {
    "mac": MEDICAL_ENTITY_MAC_CONFIG,
    "mac-8b": MEDICAL_ENTITY_MAC_8B_CONFIG,
    "mac-small": MEDICAL_ENTITY_MAC_SMALL_CONFIG,
    "poc": MEDICAL_ENTITY_POC_CONFIG,
    "full": MEDICAL_ENTITY_FULL_CONFIG,
}
