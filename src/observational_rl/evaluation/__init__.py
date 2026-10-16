"""Evaluation sub-package."""

from observational_rl.evaluation.metrics import (
    compute_rolling_stats,
    aggregate_across_seeds,
    build_learning_curves,
    summarize_experiment,
    METRIC_DESCRIPTIONS,
)

__all__ = [
    "compute_rolling_stats",
    "aggregate_across_seeds",
    "build_learning_curves",
    "summarize_experiment",
    "METRIC_DESCRIPTIONS",
]
