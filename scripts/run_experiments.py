#!/usr/bin/env python3
"""
run_experiments.py — Full scientific experiment runner.

Runs all (agent, environment, seed, demo_quality) combinations,
collects results, exports CSV, and generates all figures.

Usage:
    # Quick test (1 seed, easy env only, 200 episodes)
    python scripts/run_experiments.py --mode quick

    # Full experiment (all envs, all seeds, 2000 episodes)
    python scripts/run_experiments.py --mode full

    # Custom: specific env and demo quality
    python scripts/run_experiments.py --mode custom --envs easy medium --seeds 42 123

Options:
    --mode         quick | full | custom  (default: quick)
    --envs         Space-separated env names (for custom mode)
    --seeds        Space-separated seeds (for custom mode)
    --n-episodes   Override episode count
    --output-dir   Results directory (default: results/)
    --figures-dir  Figures directory (default: figures/)
    --no-figures   Skip figure generation
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from observational_rl.evaluation.metrics import (
    build_learning_curves,
    summarize_experiment,
)
from observational_rl.visualization.plots import (
    plot_learning_curves,
    plot_final_comparison,
    plot_convergence_comparison,
    plot_summary_heatmap,
    plot_gridworld,
)
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
        description="Run full Observational RL experiment suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument(
        "--mode", choices=["quick", "full", "custom"],
        default="quick",
        help="Experiment mode.",
    )
    parser.add_argument(
        "--envs", nargs="+", default=None,
        help="(Custom mode) Environment names.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="(Custom mode) Random seeds.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Override number of training episodes.",
    )
    parser.add_argument(
        "--demo-qualities", nargs="+",
        choices=["optimal", "suboptimal", "limited", "very_limited"],
        default=None,
        help="(Custom mode) Demonstration qualities to test.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "results"),
    )
    parser.add_argument(
        "--figures-dir", type=str,
        default=str(PROJECT_ROOT / "figures"),
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Agent configurations
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TYPES = [
    ("q_learning", None),              # No demonstrations
    ("behavioral_cloning", "demo"),    # Uses demonstrations
    ("hybrid_orl", "demo"),            # Uses demonstrations
]


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("run_experiments")

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Determine experiment parameters ──────────────────────────────────────
    exp_cfg = config.get("experiments", {})

    if args.mode == "quick":
        mode_cfg = exp_cfg.get("quick", {})
        env_names = mode_cfg.get("environments", ["easy"])
        seeds = mode_cfg.get("seeds", [42])
        demo_qualities = mode_cfg.get("demo_qualities", ["optimal"])
        n_episodes = args.n_episodes or mode_cfg.get("n_episodes", 200)
    elif args.mode == "full":
        mode_cfg = exp_cfg.get("full", {})
        env_names = mode_cfg.get("environments", ["easy", "medium", "hard"])
        seeds = mode_cfg.get("seeds", [42, 123, 456, 789, 1337])
        demo_qualities = mode_cfg.get("demo_qualities",
                                      ["optimal", "suboptimal", "limited", "very_limited"])
        n_episodes = args.n_episodes or mode_cfg.get("n_episodes", 2000)
    else:  # custom
        env_names = args.envs or ["easy"]
        seeds = args.seeds or [42]
        demo_qualities = args.demo_qualities or ["optimal"]
        n_episodes = args.n_episodes or 500

    logger.info(
        f"Experiment mode: {args.mode}\n"
        f"  Environments  : {env_names}\n"
        f"  Seeds         : {seeds}\n"
        f"  Demo qualities: {demo_qualities}\n"
        f"  N episodes    : {n_episodes}"
    )

    env_configs = config.get("environments", {})
    agent_cfgs = config.get("agents", {})
    demo_cfgs = config.get("demonstrations", {})

    # ── Run all experiments ──────────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    all_histories: Dict[str, List[Any]] = defaultdict(list)  # agent_name → histories

    total_runs = (
        len(env_names)
        * len(seeds)
        * len(demo_qualities)
        * 3  # 3 agent types
    )
    run_idx = 0
    t_total = time.perf_counter()

    for env_name in env_names:
        if env_name not in env_configs:
            logger.warning(f"Environment '{env_name}' not in config. Skipping.")
            continue

        env_config = env_configs[env_name]
        test_env_name = f"{env_name}_test"
        test_env_config = env_configs.get(test_env_name)

        for demo_quality in demo_qualities:
            demo_config = dict(demo_cfgs.get(demo_quality, {}))
            if not demo_config:
                logger.warning(f"Demo quality '{demo_quality}' not in config.")

            for agent_type, needs_demo in AGENT_TYPES:
                agent_config = agent_cfgs.get(agent_type, {})
                histories_this_run: List[Any] = []

                for seed in seeds:
                    run_idx += 1
                    logger.info(
                        f"\n[{run_idx}/{total_runs}] "
                        f"agent={agent_type}, env={env_name}, "
                        f"demo={demo_quality}, seed={seed}"
                    )

                    try:
                        result = run_single_experiment(
                            env_config=env_config,
                            agent_type=agent_type,
                            agent_config=agent_config,
                            demo_config=demo_config if needs_demo else None,
                            n_episodes=n_episodes,
                            seed=seed,
                            eval_env_config=test_env_config,
                            eval_interval=config.get("training", {}).get("eval_interval", 50),
                            eval_episodes=config.get("training", {}).get("eval_episodes", 20),
                        )

                        # Tag with demo quality
                        result["demo_quality"] = demo_quality
                        all_results.append(result)
                        histories_this_run.append(result["training_history"])
                        all_histories[result["agent_name"]].append(
                            result["training_history"]
                        )

                    except Exception as e:
                        logger.error(
                            f"Run failed (agent={agent_type}, env={env_name}, "
                            f"demo={demo_quality}, seed={seed}): {e}",
                            exc_info=True,
                        )
                        continue

    t_elapsed = time.perf_counter() - t_total
    logger.info(f"\nAll experiments complete in {t_elapsed:.1f}s.")

    # ── Export raw training histories ─────────────────────────────────────────
    all_hist_path = os.path.join(args.output_dir, "all_training_histories.csv")
    export_history_to_csv(
        [r["training_history"] for r in all_results],
        all_hist_path,
    )

    # ── Export per-seed summary ───────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "experiment_summary.csv")
    # Flatten eval dicts into top-level for CSV
    flat_results = []
    for r in all_results:
        flat = {
            "agent_name": r["agent_name"],
            "env_name": r["env_name"],
            "seed": r["seed"],
            "demo_quality": r.get("demo_quality", ""),
            "n_episodes": r["n_episodes"],
            "train_success_rate": r["final_eval"].get("success_rate"),
            "train_mean_reward": r["final_eval"].get("mean_reward"),
            "train_mean_steps": r["final_eval"].get("mean_steps"),
            "test_success_rate": (
                r["test_eval"].get("success_rate") if r.get("test_eval") else None
            ),
            "test_mean_reward": (
                r["test_eval"].get("mean_reward") if r.get("test_eval") else None
            ),
            "episodes_to_threshold": r.get("episodes_to_threshold"),
            "train_duration_s": r.get("train_duration_s"),
        }
        flat_results.append(flat)

    if flat_results:
        import csv
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_results[0].keys()))
            writer.writeheader()
            writer.writerows(flat_results)
        logger.info(f"Summary saved: {summary_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Agent':<30} {'Env':<12} {'Train SR':>9} {'Test SR':>9} {'Conv.':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*9} {'-'*9} {'-'*8}")

    for r in flat_results:
        print(
            f"  {r['agent_name']:<30} "
            f"{r['env_name']:<12} "
            f"{(r['train_success_rate'] or 0):>9.3f} "
            f"{(r['test_success_rate'] or 0):>9.3f} "
            f"{(r['episodes_to_threshold'] or 'N/A'):>8}"
        )
    print(f"{'='*70}")

    # ── Generate figures ──────────────────────────────────────────────────────
    if not args.no_figures and all_results:
        logger.info("Generating figures...")
        _generate_figures(
            all_results=all_results,
            all_histories=all_histories,
            env_names=env_names,
            demo_quality=demo_qualities[0],  # primary quality for learning curves
            figures_dir=args.figures_dir,
        )

    print(f"\nResults directory : {args.output_dir}")
    print(f"Figures directory : {args.figures_dir}")


def _generate_figures(
    all_results: List[Dict],
    all_histories: Dict[str, List],
    env_names: List[str],
    demo_quality: str,
    figures_dir: str,
) -> None:
    """Generate all standard experiment figures."""
    logger = logging.getLogger("run_experiments.figures")

    for env_name in env_names:
        # ── Learning curves per env ───────────────────────────────────────────
        env_results = [r for r in all_results if r["env_name"] == env_name
                       and r.get("demo_quality", "") == demo_quality]
        if not env_results:
            continue

        # Group by agent name
        agent_groups: Dict[str, List] = defaultdict(list)
        for r in env_results:
            agent_groups[r["agent_name"]].append(r["training_history"])

        curves = {}
        for agent_name, histories in agent_groups.items():
            if histories:
                curves[agent_name] = build_learning_curves(
                    histories, metric="success", window=50
                )

        if curves:
            path = os.path.join(figures_dir, f"learning_curves_{env_name}.png")
            plot_learning_curves(
                curves=curves,
                metric="success",
                env_name=env_name,
                window=50,
                output_path=path,
            )

        # ── Final performance bar chart ───────────────────────────────────────
        summaries = []
        for agent_name, histories in agent_groups.items():
            env_agent_results = [r for r in env_results if r["agent_name"] == agent_name]
            summary = summarize_experiment(env_agent_results, agent_name, env_name)
            if summary:
                summaries.append(summary)

        if summaries:
            bar_path = os.path.join(figures_dir, f"final_comparison_{env_name}.png")
            plot_final_comparison(
                summaries=summaries,
                metric="train_success_rate_mean",
                std_key="train_success_rate_std",
                env_name=env_name,
                output_path=bar_path,
            )

            conv_path = os.path.join(figures_dir, f"convergence_{env_name}.png")
            plot_convergence_comparison(
                summaries=summaries,
                env_name=env_name,
                output_path=conv_path,
            )

        # ── GridWorld visualization ───────────────────────────────────────────
        try:
            import yaml
            config_path = PROJECT_ROOT / "configs" / "default.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            env_cfg = cfg.get("environments", {}).get(env_name)
            if env_cfg:
                env = GridWorldEnv.from_config(env_cfg)
                gw_path = os.path.join(figures_dir, f"gridworld_{env_name}.png")
                plot_gridworld(env=env, title=f"GridWorld: {env_name}", output_path=gw_path)
        except Exception as e:
            logger.warning(f"Could not render GridWorld '{env_name}': {e}")

    # ── Cross-environment heatmap (if multiple envs) ───────────────────────
    if len(env_names) > 1:
        heatmap_data: Dict[str, Dict[str, float]] = defaultdict(dict)
        for r in all_results:
            sr = r["final_eval"].get("success_rate", 0)
            heatmap_data[r["agent_name"]][r["env_name"]] = sr

        if heatmap_data:
            hm_path = os.path.join(figures_dir, "heatmap_success_rate.png")
            plot_summary_heatmap(
                data=dict(heatmap_data),
                metric_label="Success Rate",
                output_path=hm_path,
            )

    logger.info(f"All figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
