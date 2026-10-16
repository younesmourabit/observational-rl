#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained concept by re-running an experiment
              and producing detailed evaluation reports.

Usage:
    # Evaluate all agents on medium env
    python scripts/evaluate.py --env medium --n-eval-episodes 100

    # Evaluate with generalization test
    python scripts/evaluate.py --env easy --test-env easy_test --seed 42
"""

import argparse
import logging
import os
import sys
from pathlib import Path

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
    export_summary_to_csv,
)
from observational_rl.visualization.plots import (
    plot_final_comparison,
    plot_gridworld,
)
from observational_rl.evaluation.metrics import summarize_experiment


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agents on a GridWorld environment.")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--env", type=str, default="easy",
                        help="Environment name.")
    parser.add_argument("--test-env", type=str, default=None,
                        help="Test environment name (for generalization evaluation).")
    parser.add_argument("--demo-quality", default="optimal",
                        choices=["optimal", "suboptimal", "limited", "very_limited"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--n-episodes", type=int, default=500,
                        help="Training episodes before evaluation.")
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--figures-dir", type=str,
                        default=str(PROJECT_ROOT / "figures"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"],
                        default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("evaluate")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env_configs = config["environments"]
    agent_cfgs = config["agents"]
    demo_cfgs = config["demonstrations"]

    if args.env not in env_configs:
        logger.error(f"Environment '{args.env}' not in config.")
        sys.exit(1)

    env_config = env_configs[args.env]
    test_env_config = env_configs.get(args.test_env) if args.test_env else None
    demo_config = demo_cfgs.get(args.demo_quality, {})

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    agent_types = [
        ("q_learning", False),
        ("behavioral_cloning", True),
        ("hybrid_orl", True),
    ]

    all_results = []
    for agent_type, needs_demo in agent_types:
        agent_config = agent_cfgs.get(agent_type, {})
        for seed in args.seeds:
            logger.info(f"Evaluating: {agent_type} | env={args.env} | seed={seed}")
            try:
                result = run_single_experiment(
                    env_config=env_config,
                    agent_type=agent_type,
                    agent_config=agent_config,
                    demo_config=demo_config if needs_demo else None,
                    n_episodes=args.n_episodes,
                    seed=seed,
                    eval_env_config=test_env_config,
                    eval_interval=50,
                    eval_episodes=args.n_eval_episodes,
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed: {e}", exc_info=True)

    # Summary CSV
    summary_path = os.path.join(
        args.output_dir, f"eval_{args.env}_{args.demo_quality}.csv"
    )
    export_summary_to_csv(all_results, summary_path)

    # Print table
    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS — env={args.env}, demo={args.demo_quality}")
    print(f"{'='*70}")
    print(f"  {'Agent':<30} {'Seed':>5} {'Train SR':>9} {'Test SR':>9} {'Conv.':>8}")
    print(f"  {'-'*30} {'-'*5} {'-'*9} {'-'*9} {'-'*8}")
    for r in all_results:
        print(
            f"  {r['agent_name']:<30} "
            f"{r['seed']:>5} "
            f"{r['final_eval'].get('success_rate', 0):>9.3f} "
            f"{(r['test_eval'] or {}).get('success_rate', 0):>9.3f} "
            f"{(r.get('episodes_to_threshold') or 'N/A'):>8}"
        )
    print(f"{'='*70}")

    # Comparison bar chart
    summaries = []
    agent_names_seen = set()
    for r in all_results:
        n = r["agent_name"]
        if n not in agent_names_seen:
            agent_names_seen.add(n)
            s = summarize_experiment(
                [x for x in all_results if x["agent_name"] == n],
                n, args.env
            )
            if s:
                summaries.append(s)

    if summaries:
        bar_path = os.path.join(args.figures_dir, f"eval_comparison_{args.env}.png")
        plot_final_comparison(
            summaries=summaries,
            metric="train_success_rate_mean",
            std_key="train_success_rate_std",
            env_name=args.env,
            output_path=bar_path,
        )
        logger.info(f"Comparison figure: {bar_path}")

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
