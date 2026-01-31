#!/usr/bin/env python3
"""Quick training test without Flash Attention."""

import sys
sys.path.insert(0, '.')

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from config.base import ModelConfig, TrainingConfig, LoRAConfig, DataConfig, LoggingConfig
from src.models import load_model_and_tokenizer
from src.data import AlpacaDataset
from src.training.sft_trainer import SFTTrainer

print('='*60)
print('   Quick Training Test')
print('='*60)
print()

# Model config (NO Flash Attention)
model_config = ModelConfig(
    name='Qwen/Qwen2.5-0.5B-Instruct',
    quantization_bits=4,
    use_flash_attention=False,  # Critical!
    max_length=512,
)

# Training config
training_config = TrainingConfig(
    output_dir='./outputs/test-quick',
    num_epochs=1,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    bf16=True,
    seed=42,
)

# LoRA config
lora_config = LoRAConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj'],
)

# Data config
data_config = DataConfig(
    dataset_name='yahma/alpaca-cleaned',
    max_samples=100,
    validation_split=0.1,
    format='alpaca',
)

# Logging
logging_config = LoggingConfig(
    use_wandb=False,
    use_tensorboard=False,
    console_level='INFO',
)

print(f'Model: {model_config.name}')
print(f'Quantization: {model_config.quantization_bits}-bit')
print(f'Dataset: {data_config.dataset_name}')
print(f'Max samples: {data_config.max_samples}')
print(f'Epochs: {training_config.num_epochs}')
print()

# Create trainer
trainer = SFTTrainer(
    model_config=model_config,
    training_config=training_config,
    lora_config=lora_config,
    data_config=data_config,
    logging_config=logging_config,
)

print('Starting training...')
print()

try:
    trainer.prepare_model()
    trainer.prepare_data()
    trainer.setup_trainer()
    trainer.train()
    trainer.evaluate()
    
    print()
    print('='*60)
    print('✅ TRAINING COMPLETE!')
    print('='*60)
    
except Exception as e:
    print()
    print('='*60)
    print(f'❌ Training failed: {e}')
    print('='*60)
    import traceback
    traceback.print_exc()
