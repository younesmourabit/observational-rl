"""
Experiment runner — orchestrates multi-seed, multi-environment experiments.

This module is the main scientific entry point. It:
1. Loads configuration from YAML.
2. Creates environments, agents, and demonstrations.
3. Runs training for each (agent, env, seed) combination.
4. Collects all results and exports to CSV.
5. Produces summary statistics (mean ± std across seeds).

Design principle:
    All randomness is controlled by explicit seeds derived from the base seed.
    This ensures reproducibility: same config → same results.
"""

import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from observational_rl.agents.behavioral_cloning import BehavioralCloningAgent
from observational_rl.agents.hybrid_orl import HybridORLAgent
from observational_rl.agents.q_learning import QLearningAgent
from observational_rl.demonstrations.expert import ExpertDemonstrationGenerator
from observational_rl.environments.gridworld import GridWorldEnv
from observational_rl.training.trainer import evaluate_agent, train_agent, TrainingHistory

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_environment(env_config: Dict[str, Any]) -> GridWorldEnv:
    """Create a GridWorldEnv from a configuration dict."""
    return GridWorldEnv.from_config(env_config)


def make_agent(
    agent_type: str,
    state_size: int,
    action_size: int,
    agent_config: Dict[str, Any],
    seed: int,
) -> Any:
    """
    Create an agent from a type string and configuration.

    Args:
        agent_type:   One of "q_learning", "behavioral_cloning", "hybrid_orl".
        state_size:   Number of environment states.
        action_size:  Number of actions.
        agent_config: Agent-specific configuration dict.
        seed:         Random seed.

    Returns:
        An instantiated agent.

    Raises:
        ValueError: If agent_type is unknown.
    """
    if agent_type == "q_learning":
        return QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=agent_config.get("learning_rate", 0.1),
            discount_factor=agent_config.get("discount_factor", 0.99),
            epsilon_start=agent_config.get("epsilon_start", 1.0),
            epsilon_end=agent_config.get("epsilon_end", 0.05),
            epsilon_decay=agent_config.get("epsilon_decay", 0.995),
            seed=seed,
            name=agent_config.get("name", "Q-Learning"),
        )
    elif agent_type == "behavioral_cloning":
        return BehavioralCloningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=agent_config.get("learning_rate", 0.001),
            hidden_dims=agent_config.get("hidden_dims", [64, 64]),
            batch_size=agent_config.get("batch_size", 32),
            n_epochs=agent_config.get("n_epochs", 50),
            seed=seed,
            name=agent_config.get("name", "Behavioral Cloning"),
        )
    elif agent_type == "hybrid_orl":
        return HybridORLAgent(
            state_size=state_size,
            action_size=action_size,
            bc_learning_rate=agent_config.get("bc_learning_rate", 0.001),
            bc_hidden_dims=agent_config.get("bc_hidden_dims", [64, 64]),
            bc_batch_size=agent_config.get("bc_batch_size", 32),
            bc_epochs=agent_config.get("bc_epochs", 30),
            rl_learning_rate=agent_config.get("rl_learning_rate", 0.1),
            rl_discount_factor=agent_config.get("rl_discount_factor", 0.99),
            rl_epsilon_start=agent_config.get("rl_epsilon_start", 0.3),
            rl_epsilon_end=agent_config.get("rl_epsilon_end", 0.05),
            rl_epsilon_decay=agent_config.get("rl_epsilon_decay", 0.998),
            q_init_from_bc=agent_config.get("q_init_from_bc", True),
            seed=seed,
            name=agent_config.get("name", "Hybrid ORL (BC→QL)"),
        )
    else:
        raise ValueError(f"Unknown agent_type: '{agent_type}'. "
                         "Must be 'q_learning', 'behavioral_cloning', or 'hybrid_orl'.")


def run_single_experiment(
    env_config: Dict[str, Any],
    agent_type: str,
    agent_config: Dict[str, Any],
    demo_config: Optional[Dict[str, Any]],
    n_episodes: int,
    seed: int,
    eval_env_config: Optional[Dict[str, Any]] = None,
    eval_interval: int = 50,
    eval_episodes: int = 20,
) -> Dict[str, Any]:
    """
    Run one full experiment: one agent, one env, one seed.

    Args:
        env_config:       Training environment configuration.
        agent_type:       Agent type string.
        agent_config:     Agent hyperparameters.
        demo_config:      Demonstration config (None for pure RL).
        n_episodes:       Training episode budget.
        seed:             Random seed for this run.
        eval_env_config:  Optional test environment config.
        eval_interval:    Evaluation frequency (episodes).
        eval_episodes:    Episodes per evaluation.

    Returns:
        Dict with training_history, final_eval, test_eval (if available),
        agent_config, env_config, demo_stats.
    """
    set_global_seed(seed)

    # Create environments
    env = make_environment(env_config)
    eval_env = make_environment(eval_env_config) if eval_env_config else env

    # Create agent
    agent = make_agent(
        agent_type=agent_type,
        state_size=env.state_space_size,
        action_size=env.action_space_size,
        agent_config=agent_config,
        seed=seed,
    )

    # Generate demonstrations (if needed)
    demo_stats: Dict[str, Any] = {}
    if demo_config is not None and agent_type in ("behavioral_cloning", "hybrid_orl"):
        gen = ExpertDemonstrationGenerator(env=env, seed=seed)
        demos = gen.generate(
            n_demos=demo_config.get("n_demos", 50),
            quality=demo_config.get("quality", "optimal"),
            noise_prob=demo_config.get("noise_prob", 0.0),
        )
        demo_stats = ExpertDemonstrationGenerator.get_statistics(demos)
        logger.info(f"Generated {len(demos)} demonstrations. Stats: {demo_stats}")

        # Phase 1: pre-train
        bc_train_stats = agent.pre_train(demos)
        demo_stats["bc_final_loss"] = bc_train_stats.get("final_loss")

    # Phase 2 (or full RL): training loop
    t_train_start = time.perf_counter()
    history = train_agent(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        eval_env=eval_env,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        verbose=True,
        log_interval=max(1, n_episodes // 10),
    )
    train_duration = time.perf_counter() - t_train_start

    # Final evaluation on training env
    final_eval = evaluate_agent(agent, env, n_episodes=eval_episodes)
    final_eval["phase"] = "train_env_final"

    # Final evaluation on test env (if different)
    test_eval: Optional[Dict] = None
    if eval_env_config is not None:
        test_env = make_environment(eval_env_config)
        test_eval = evaluate_agent(agent, test_env, n_episodes=eval_episodes)
        test_eval["phase"] = "test_env_final"

    return {
        "agent_type": agent_type,
        "agent_name": agent.name,
        "env_name": env.name,
        "seed": seed,
        "n_episodes": n_episodes,
        "train_duration_s": train_duration,
        "training_history": history,
        "final_eval": final_eval,
        "test_eval": test_eval,
        "demo_stats": demo_stats,
        "agent_config": agent.get_config(),
        "env_config": env.get_config(),
        "episodes_to_threshold": history.episodes_to_threshold(
            threshold=0.7, window=50
        ),
    }


def export_history_to_csv(
    histories: List[TrainingHistory],
    output_path: str,
) -> None:
    """
    Export multiple training histories to a single CSV file.

    Args:
        histories:    List of TrainingHistory objects.
        output_path:  Path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for h in histories:
        rows.extend(h.to_dict_list())

    if not rows:
        logger.warning("No rows to export.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported {len(rows)} rows to {output_path}")


def export_summary_to_csv(
    results: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Export per-seed summary statistics to CSV.

    Args:
        results:     List of result dicts from run_single_experiment.
        output_path: Output CSV path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for r in results:
        fe = r.get("final_eval", {})
        te = r.get("test_eval") or {}
        row = {
            "agent_name": r["agent_name"],
            "env_name": r["env_name"],
            "seed": r["seed"],
            "n_episodes": r["n_episodes"],
            "train_success_rate": fe.get("success_rate"),
            "train_mean_reward": fe.get("mean_reward"),
            "train_std_reward": fe.get("std_reward"),
            "train_mean_steps": fe.get("mean_steps"),
            "test_success_rate": te.get("success_rate"),
            "test_mean_reward": te.get("mean_reward"),
            "test_std_reward": te.get("std_reward"),
            "test_mean_steps": te.get("mean_steps"),
            "episodes_to_threshold": r.get("episodes_to_threshold"),
            "train_duration_s": r.get("train_duration_s"),
            "demo_n": r.get("demo_stats", {}).get("n_demos"),
            "demo_success_rate": r.get("demo_stats", {}).get("success_rate"),
        }
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Exported summary ({len(rows)} rows) to {output_path}")
