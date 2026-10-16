"""
Metrics and evaluation utilities for Observational RL experiments.

Provides functions to compute, aggregate, and compare performance metrics
across multiple seeds and agent types.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metric definitions
# ─────────────────────────────────────────────────────────────────────────────

METRIC_DESCRIPTIONS = {
    "success_rate": "Fraction of episodes where the agent reached the goal",
    "mean_reward": "Mean cumulative reward per episode",
    "std_reward": "Standard deviation of cumulative reward",
    "mean_steps": "Mean number of steps per episode",
    "episodes_to_threshold": (
        "First episode where 50-episode rolling success rate ≥ 0.7"
    ),
    "train_duration_s": "Wall-clock training time in seconds",
    "test_success_rate": "Success rate on held-out test environment",
}


def compute_rolling_stats(
    values: List[float],
    window: int = 50,
) -> Dict[str, List[float]]:
    """
    Compute rolling mean and std over a list of values.

    Args:
        values: List of scalar episode metrics.
        window: Rolling window size.

    Returns:
        Dict with keys "mean" and "std" (lists same length as values).
    """
    arr = np.array(values, dtype=float)
    means, stds = [], []
    for i in range(len(arr)):
        w_start = max(0, i - window + 1)
        window_vals = arr[w_start:i + 1]
        means.append(float(np.mean(window_vals)))
        stds.append(float(np.std(window_vals)))
    return {"mean": means, "std": stds}


def aggregate_across_seeds(
    results: List[Dict[str, Any]],
    metric_key: str,
) -> Dict[str, float]:
    """
    Aggregate a scalar metric across multiple seeds.

    Args:
        results:    List of result dicts (one per seed).
        metric_key: Key to look up in each result dict.

    Returns:
        Dict with: mean, std, min, max, n_seeds, n_nan (seeds where metric is None).
    """
    values = [r.get(metric_key) for r in results]
    valid = [v for v in values if v is not None]
    n_nan = len(values) - len(valid)

    if not valid:
        return {"mean": None, "std": None, "min": None, "max": None,
                "n_seeds": len(values), "n_nan": n_nan}

    arr = np.array(valid, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_seeds": len(values),
        "n_nan": n_nan,
    }


def build_learning_curves(
    histories: List[Any],  # List[TrainingHistory]
    metric: str = "total_reward",
    window: int = 50,
) -> Dict[str, Any]:
    """
    Build mean ± std learning curves from multiple training histories.

    Args:
        histories: List of TrainingHistory objects (one per seed).
        metric:    Attribute name of EpisodeRecord to plot.
                   One of: "total_reward", "n_steps", "success".
        window:    Rolling window for smoothing.

    Returns:
        Dict with:
            episodes:     List of episode indices (0-indexed).
            mean:         Mean values across seeds (after smoothing).
            std:          Std across seeds (after smoothing).
            raw_per_seed: Per-seed raw values (before smoothing).
    """
    # Determine the number of episodes (use minimum across seeds)
    min_len = min(len(h.records) for h in histories)
    raw_per_seed = []

    for h in histories:
        values = [getattr(r, metric) for r in h.records[:min_len]]
        raw_per_seed.append(values)

    raw_arr = np.array(raw_per_seed, dtype=float)  # (n_seeds, n_episodes)

    # Smooth each seed individually, then aggregate
    smoothed = np.zeros_like(raw_arr)
    for i in range(len(histories)):
        rolling = compute_rolling_stats(raw_arr[i].tolist(), window=window)
        smoothed[i] = rolling["mean"]

    return {
        "episodes": list(range(min_len)),
        "mean": smoothed.mean(axis=0).tolist(),
        "std": smoothed.std(axis=0).tolist(),
        "raw_per_seed": raw_per_seed,
    }


def summarize_experiment(
    results: List[Dict[str, Any]],
    agent_name: str,
    env_name: str,
) -> Dict[str, Any]:
    """
    Produce a full summary table for one (agent, env) combination.

    Args:
        results:    Result dicts from run_single_experiment.
        agent_name: Agent name for labeling.
        env_name:   Environment name for labeling.

    Returns:
        Summary dict with aggregated metrics (mean ± std across seeds).
    """
    filtered = [
        r for r in results
        if r.get("agent_name") == agent_name and r.get("env_name") == env_name
    ]
    if not filtered:
        return {}

    metrics = {
        "train_success_rate": [
            r["final_eval"].get("success_rate") for r in filtered
        ],
        "train_mean_reward": [
            r["final_eval"].get("mean_reward") for r in filtered
        ],
        "train_mean_steps": [
            r["final_eval"].get("mean_steps") for r in filtered
        ],
        "episodes_to_threshold": [
            r.get("episodes_to_threshold") for r in filtered
        ],
        "train_duration_s": [
            r.get("train_duration_s") for r in filtered
        ],
    }

    # Add test metrics if available
    if any(r.get("test_eval") for r in filtered):
        metrics["test_success_rate"] = [
            r["test_eval"].get("success_rate") if r.get("test_eval") else None
            for r in filtered
        ]
        metrics["test_mean_reward"] = [
            r["test_eval"].get("mean_reward") if r.get("test_eval") else None
            for r in filtered
        ]

    summary = {
        "agent_name": agent_name,
        "env_name": env_name,
        "n_seeds": len(filtered),
    }
    for metric_name, values in metrics.items():
        valid = [v for v in values if v is not None]
        if valid:
            summary[f"{metric_name}_mean"] = float(np.mean(valid))
            summary[f"{metric_name}_std"] = float(np.std(valid))
        else:
            summary[f"{metric_name}_mean"] = None
            summary[f"{metric_name}_std"] = None

    return summary
