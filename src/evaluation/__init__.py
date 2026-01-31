"""Evaluation modules for model assessment."""

from src.evaluation.comparisons import compare_models, side_by_side_generation
from src.evaluation.metrics import compute_accuracy, compute_perplexity
from src.evaluation.qualitative import generate_samples, interactive_generation

__all__ = [
    # Metrics
    "compute_perplexity",
    "compute_accuracy",
    # Qualitative
    "generate_samples",
    "interactive_generation",
    # Comparisons
    "compare_models",
    "side_by_side_generation",
]
