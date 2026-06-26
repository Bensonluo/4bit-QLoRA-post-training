"""CLI to run one iteration of the data flywheel."""

from __future__ import annotations

import json

import typer

from src.data_flywheel import (
    BadCaseMiner,
    DataFlywheelPipeline,
    DatasetItem,
    DataSynthesizer,
    LocalDatasetRegistry,
    PreferenceBuilder,
    RuleJudgeClient,
)
from src.utils import console

app = typer.Typer(help="Run data flywheel iteration")


class _DummyModelClient:
    """Minimal model client for synthesizer."""

    def __init__(self, response: str = "Synthetic prompt") -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


@app.command()
def run(
    seed_file: str = typer.Option(..., "--seed-file", help="Path to seed JSONL file"),
    completions_file: str = typer.Option(
        ..., "--completions-file", help="Path to completions JSONL file"
    ),
    output_dir: str = typer.Option("./data/registry", "--output-dir"),
    strategy: str = typer.Option("self_instruct", "--strategy"),
    n_synthetic: int = typer.Option(10, "--n-synthetic"),
    min_margin: float = typer.Option(0.0, "--min-margin"),
) -> None:
    """Run one data flywheel iteration from seed data and completions."""
    registry = LocalDatasetRegistry(base_dir=output_dir)

    seed_items = _load_seed_items(seed_file)
    prompt_completions = _load_completions(completions_file)

    synthesizer = DataSynthesizer(
        model_client=_DummyModelClient(),
        strategy=strategy,
    )
    builder = PreferenceBuilder(min_margin=min_margin)
    judge = RuleJudgeClient()
    miner = BadCaseMiner()

    pipeline = DataFlywheelPipeline(
        synthesizer=synthesizer,
        preference_builder=builder,
        registry=registry,
        judge=judge,
        miner=miner,
    )

    result = pipeline.run_iteration(
        seed_data=seed_items,
        prompt_completions=prompt_completions,
        n_synthetic=n_synthetic,
    )

    console.print("\n[bold green]=== Flywheel Iteration Complete ===[/bold green]")
    for key, value in result.items():
        console.print(f"  {key}: {value}")


def _load_seed_items(path: str) -> list[DatasetItem]:
    """Load seed items from JSONL."""
    items: list[DatasetItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            items.append(
                DatasetItem(
                    id=data.get("id", ""),
                    prompt=data["prompt"],
                    response=data.get("response"),
                    source=data.get("source", "seed"),
                )
            )
    return items


def _load_completions(path: str) -> list[dict]:
    """Load completions from JSONL."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


if __name__ == "__main__":
    app()
