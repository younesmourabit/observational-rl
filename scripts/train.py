#!/usr/bin/env python3
"""
train.py — Train a single agent on a GridWorld environment.

Usage examples:
    # Train Q-learning on easy env with default config
    python scripts/train.py --agent q_learning --env easy

    # Train Hybrid ORL with optimal demonstrations
    python scripts/train.py --agent hybrid_orl --env medium --demo-quality optimal --n-demos 50

    # Train with custom seed and episode budget
    python scripts/train.py --agent q_learning --env hard --seed 123 --n-episodes 1000

    # Quick sanity check (fast mode)
    python scripts/train.py --agent q_learning --env easy --n-episodes 200 --seed 42 --render-final
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from project root without pip install
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import yaml

from observational_rl.training.experiment_runner import (
    run_single_experiment,
    export_history_to_csv,
    export_summary_to_csv,
    set_global_seed,
)
from observational_rl.visualization.plots import (
    plot_learning_curves,
    plot_gridworld,
    plot_final_comparison,
)
from observational_rl.evaluation.metrics import build_learning_curves
from observational_rl.environments.gridworld import GridWorldEnv


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single RL/IL/Hybrid agent on a GridWorld environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--agent",
        choices=["q_learning", "behavioral_cloning", "hybrid_orl"],
        default="q_learning",
        help="Agent type to train.",
    )
    parser.add_argument(
        "--env", type=str, default="easy",
        help="Environment name (must exist in config YAML).",
    )
    parser.add_argument(
        "--demo-quality",
        choices=["optimal", "suboptimal", "limited", "very_limited"],
        default="optimal",
        help="Demonstration quality (for BC and Hybrid ORL).",
    )
    parser.add_argument(
        "--n-demos", type=int, default=None,
        help="Override number of demonstrations (overrides config).",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Number of training episodes (overrides config).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory for output CSVs and figures.",
    )
    parser.add_argument(
        "--render-final", action="store_true",
        help="Render the GridWorld with learned policy after training.",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("train")

    # Load configuration
    config = load_config(args.config)

    env_configs = config.get("environments", {})
    if args.env not in env_configs:
        logger.error(f"Environment '{args.env}' not found in config. "
                     f"Available: {list(env_configs.keys())}")
        sys.exit(1)

    env_config = env_configs[args.env]

    agent_configs = config.get("agents", {})
    if args.agent not in agent_configs:
        logger.error(f"Agent '{args.agent}' not found in config.")
        sys.exit(1)
    agent_config = agent_configs[args.agent]

    demo_configs = config.get("demonstrations", {})
    demo_config = demo_configs.get(args.demo_quality, {})

    # Override n_demos if specified
    if args.n_demos is not None:
        demo_config = dict(demo_config)
        demo_config["n_demos"] = args.n_demos

    # Override n_episodes if specified
    n_episodes = args.n_episodes
    if n_episodes is None:
        n_episodes = config.get("training", {}).get("n_episodes", 500)

    # Determine test env (with _test suffix)
    test_env_name = f"{args.env}_test"
    test_env_config = env_configs.get(test_env_name)

    logger.info(
        f"Starting: agent={args.agent}, env={args.env}, "
        f"seed={args.seed}, n_episodes={n_episodes}"
    )

    # Run experiment
    needs_demos = args.agent in ("behavioral_cloning", "hybrid_orl")
    result = run_single_experiment(
        env_config=env_config,
        agent_type=args.agent,
        agent_config=agent_config,
        demo_config=demo_config if needs_demos else None,
        n_episodes=n_episodes,
        seed=args.seed,
        eval_env_config=test_env_config,
        eval_interval=config.get("training", {}).get("eval_interval", 50),
        eval_episodes=config.get("training", {}).get("eval_episodes", 20),
    )

    # Save training history CSV
    os.makedirs(args.output_dir, exist_ok=True)
    history = result["training_history"]
    csv_path = os.path.join(
        args.output_dir,
        f"history_{args.agent}_{args.env}_seed{args.seed}.csv",
    )
    export_history_to_csv([history], csv_path)

    # Save summary CSV
    summary_path = os.path.join(
        args.output_dir,
        f"summary_{args.agent}_{args.env}_seed{args.seed}.csv",
    )
    export_summary_to_csv([result], summary_path)

    # Print final results
    final_eval = result["final_eval"]
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Agent       : {result['agent_name']}")
    print(f"  Environment : {result['env_name']}")
    print(f"  Seed        : {result['seed']}")
    print(f"  Episodes    : {n_episodes}")
    print(f"  Train time  : {result['train_duration_s']:.1f}s")
    print(f"{'-'*60}")
    print(f"  FINAL EVALUATION (train env, {final_eval['n_episodes']} episodes):")
    print(f"  Success rate : {final_eval['success_rate']:.3f}")
    print(f"  Mean reward  : {final_eval['mean_reward']:.3f} ± {final_eval['std_reward']:.3f}")
    print(f"  Mean steps   : {final_eval['mean_steps']:.1f}")
    if result.get("test_eval"):
        te = result["test_eval"]
        print(f"{'-'*60}")
        print(f"  GENERALIZATION (test env, {te['n_episodes']} episodes):")
        print(f"  Success rate : {te['success_rate']:.3f}")
        print(f"  Mean reward  : {te['mean_reward']:.3f} ± {te['std_reward']:.3f}")
    eps_thresh = result.get("episodes_to_threshold")
    if eps_thresh is not None:
        print(f"{'-'*60}")
        print(f"  Convergence  : reached 70% success at episode {eps_thresh}")
    else:
        print(f"{'-'*60}")
        print(f"  Convergence  : 70% success rate NOT reached")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")

    # Generate figures
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Learning curve
    curves = {
        result["agent_name"]: build_learning_curves(
            [history], metric="success", window=50
        )
    }
    fig_path = os.path.join(
        figures_dir,
        f"learning_curve_{args.agent}_{args.env}_seed{args.seed}.png"
    )
    plot_learning_curves(
        curves=curves,
        metric="success",
        env_name=args.env,
        window=50,
        output_path=fig_path,
    )
    print(f"Learning curve saved: {fig_path}")

    # GridWorld visualization with policy
    if args.render_final:
        env = GridWorldEnv.from_config(env_config)
        # We need the trained agent — reconstruct for visualization
        # (simplified: just render env layout)
        gw_path = os.path.join(figures_dir, f"gridworld_{args.env}.png")
        plot_gridworld(
            env=env,
            title=f"GridWorld: {args.env}",
            output_path=gw_path,
        )
        print(f"GridWorld visualization saved: {gw_path}")


if __name__ == "__main__":
    main()
