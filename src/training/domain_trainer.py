"""Domain adaptation trainer (extends SFT with domain-specific features)."""

from typing import Optional

from config.base import ModelConfig, TrainingConfig, LoRAConfig, DataConfig, LoggingConfig
from src.training.sft_trainer import SFTTrainer
from src.utils import console


class DomainAdaptationTrainer(SFTTrainer):
    """Trainer for domain-specific fine-tuning.

    This extends SFTTrainer with domain-specific features:
    - Domain-specific prompts
    - Curriculum learning (start general, move to domain)
    - Domain-specific evaluation metrics
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        lora_config: LoRAConfig,
        data_config: DataConfig,
        logging_config: LoggingConfig,
        domain_name: str = "finance",
    ):
        """Initialize domain adaptation trainer.

        Args:
            model_config: Model configuration
            training_config: Training configuration
            lora_config: LoRA configuration
            data_config: Data configuration
            logging_config: Logging configuration
            domain_name: Name of the domain (e.g., "finance", "medical")
        """
        super().__init__(
            model_config=model_config,
            training_config=training_config,
            lora_config=lora_config,
            data_config=data_config,
            logging_config=logging_config,
        )
        self.domain_name = domain_name

    def prepare_data(self):
        """Load and prepare domain-specific dataset."""
        console.print(f"\n[bold cyan]=== Preparing {self.domain_name.title()} Domain Data ===[/bold cyan]\n")

        # Use FinanceDataset for finance domain
        if self.domain_name == "finance":
            from src.data import FinanceDataset

            dataset = FinanceDataset(
                data_path=self.data_config.dataset_name,
                max_samples=self.data_config.max_samples,
            )
        else:
            # Use base AlpacaDataset for other domains
            from src.data import AlpacaDataset

            dataset = AlpacaDataset(
                data_path=self.data_config.dataset_name,
                max_samples=self.data_config.max_samples,
            )

        # Load dataset (this will filter for domain-specific content)
        dataset.load()

        # Split
        self.train_dataset, self.eval_dataset = dataset.split_dataset(
            validation_split=self.data_config.validation_split,
            seed=self.training_config.seed,
        )

        console.print(f"[green]✓ {self.domain_name.title()} train samples: {len(self.train_dataset):,}[/green]")
        console.print(f"[green]✓ {self.domain_name.title()} val samples: {len(self.eval_dataset):,}[/green]\n")

        # Format for training
        self.train_dataset = dataset.format_for_training(
            self.tokenizer,
            max_length=self.model_config.max_length,
        )
        self.eval_dataset = dataset.format_for_training(
            self.tokenizer,
            max_length=self.model_config.max_length,
        )


def run_domain_adaptation(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    lora_config: LoRAConfig,
    data_config: DataConfig,
    logging_config: LoggingConfig,
    domain_name: str = "finance",
) -> None:
    """Run domain adaptation training.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        lora_config: LoRA configuration
        data_config: Data configuration
        logging_config: Logging configuration
        domain_name: Name of the domain
    """
    console.print(f"\n[bold magenta]Starting {domain_name.title()} Domain Adaptation[/bold magenta]\n")

    trainer = DomainAdaptationTrainer(
        model_config=model_config,
        training_config=training_config,
        lora_config=lora_config,
        data_config=data_config,
        logging_config=logging_config,
        domain_name=domain_name,
    )

    trainer.prepare_model()
    trainer.prepare_data()
    trainer.setup_trainer()
    trainer.train()
    trainer.evaluate()

    console.print(f"\n[bold green]{domain_name.title()} Domain Adaptation Complete![/bold green]\n")
