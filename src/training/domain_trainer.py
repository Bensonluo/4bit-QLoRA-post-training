"""Domain adaptation trainer (extends SFT with domain-specific features)."""

from pathlib import Path

from config.base import DataConfig, LoggingConfig, LoRAConfig, ModelConfig, TrainingConfig
from src.data.base import BaseDataset
from src.training.sft_trainer import SFTTrainer, _dataset_class_for
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
    ) -> None:
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

    def prepare_data(self) -> None:
        """Load and prepare domain-specific dataset."""
        console.print(
            f"\n[bold cyan]=== Preparing {self.domain_name.title()} Domain Data ===[/bold cyan]\n"
        )

        # Materialized data already has approved filtering and formatting rules.
        dataset: BaseDataset
        if self.data_config.dataset_loader is not None:
            dataset_class = _dataset_class_for(
                self.data_config.dataset_name, self.data_config.dataset_loader
            )
            dataset = dataset_class(
                data_path=self.data_config.dataset_name,
                max_samples=self.data_config.max_samples,
            )
        elif self.domain_name == "finance":
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

        # Match SFT's explicit validation-file path when automatic splitting is off.
        validation_loader: BaseDataset | None = None
        if self.eval_dataset is None and self.data_config.validation_file:
            if Path(self.data_config.validation_file).exists():
                validation_loader = type(dataset)(
                    data_path=self.data_config.validation_file,
                    max_samples=self.data_config.max_samples,
                )
                validation_loader.load()
                self.eval_dataset = validation_loader.dataset
            else:
                console.print("[yellow]⚠ No validation data found[/yellow]")

        console.print(
            f"[green]✓ {self.domain_name.title()} train samples: {len(self.train_dataset):,}[/green]"
        )
        val_count = len(self.eval_dataset) if self.eval_dataset is not None else 0
        console.print(f"[green]✓ {self.domain_name.title()} val samples: {val_count:,}[/green]\n")

        # The loader still holds the full source after split_dataset returns.
        dataset.dataset = self.train_dataset
        self.train_dataset = dataset.format_for_training(
            self.tokenizer,
            max_length=self.model_config.max_length,
        )
        if self.eval_dataset is not None:
            if validation_loader is None:
                validation_loader = dataset
            validation_loader.dataset = self.eval_dataset
            self.eval_dataset = validation_loader.format_for_training(
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
    console.print(
        f"\n[bold magenta]Starting {domain_name.title()} Domain Adaptation[/bold magenta]\n"
    )

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
