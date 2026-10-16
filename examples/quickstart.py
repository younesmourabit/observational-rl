#!/usr/bin/env python3
"""
quickstart.py — Minimal end-to-end demonstration of Observational RL.

Runs:
1. Environment instantiation and visualization
2. Demonstration generation (optimal vs suboptimal)
3. Three-way comparison:
   - Pure RL (Q-Learning)
   - Pure Imitation (Behavioral Cloning)
   - Hybrid ORL (BC warm-start + Q-Learning)
4. Evaluation and printing comparative metrics

Run with:
    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Add src to python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from observational_rl.environments.gridworld import GridWorldEnv
from observational_rl.demonstrations.expert import ExpertDemonstrationGenerator
from observational_rl.agents.q_learning import QLearningAgent
from observational_rl.agents.behavioral_cloning import BehavioralCloningAgent
from observational_rl.agents.hybrid_orl import HybridORLAgent
from observational_rl.training.trainer import train_agent, evaluate_agent


def main():
    print("=" * 70)
    print("  OBSERVATIONAL REINFORCEMENT LEARNING — QUICKSTART DEMO")
    print("=" * 70)

    # 1. Environment
    print("\n[1] Creating GridWorld Environment (6x6 with obstacles)...")
    env = GridWorldEnv(
        grid_size=6,
        start_pos=(0, 0),
        goal_pos=(5, 5),
        n_obstacles=5,
        max_steps=50,
        obstacle_seed=42,
    )
    print("  State space size :", env.state_space_size)
    print("  Action space size:", env.action_space_size)
    path = env.find_shortest_path()
    print("  Shortest path len:", len(path) if path else "N/A")
    print("\nEnvironment ASCII render:")
    env.render()

    # 2. Expert Demonstrations
    print("\n[2] Generating Expert Demonstrations...")
    generator = ExpertDemonstrationGenerator(env=env, seed=42)
    demos = generator.generate(n_demos=30, quality="optimal")
    stats = generator.get_statistics(demos)
    print(f"  Generated {stats['n_demos']} optimal demonstrations.")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Mean reward : {stats['mean_reward']:.2f}")
    print(f"  Mean steps  : {stats['mean_steps']:.1f}")

    # 3. Instantiate Agents
    print("\n[3] Initializing Agents...")
    agents = {
        "Pure RL (Q-Learning)": QLearningAgent(
            state_size=env.state_space_size,
            action_size=env.action_space_size,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.98,
            epsilon_end=0.05,
            seed=42,
        ),
        "Pure Imitation (BC)": BehavioralCloningAgent(
            state_size=env.state_space_size,
            action_size=env.action_space_size,
            learning_rate=0.005,
            hidden_dims=[32, 32],
            batch_size=16,
            n_epochs=30,
            seed=42,
        ),
        "Hybrid ORL (BC->QL)": HybridORLAgent(
            state_size=env.state_space_size,
            action_size=env.action_space_size,
            rl_learning_rate=0.1,
            rl_discount_factor=0.99,
            rl_epsilon_start=0.3,
            rl_epsilon_decay=0.98,
            rl_epsilon_end=0.05,
            bc_learning_rate=0.005,
            bc_hidden_dims=[32, 32],
            bc_epochs=30,
            seed=42,
        ),
    }

    # Pre-train BC and Hybrid agents
    print("  Pre-training Behavioral Cloning on expert demonstrations...")
    agents["Pure Imitation (BC)"].pre_train(demos)

    print("  Pre-training Hybrid ORL (BC warm-start of Q-values)...")
    agents["Hybrid ORL (BC->QL)"].pre_train(demos)

    # 4. Training loop (online interaction)
    n_episodes = 150
    print(f"\n[4] Training Agents ({n_episodes} interaction episodes)...")
    histories = {}
    for name, agent in agents.items():
        print(f"  Training {name}...")
        hist = train_agent(
            agent=agent,
            env=env,
            n_episodes=n_episodes,
            eval_interval=50,
            verbose=False,
        )
        histories[name] = hist
        print(f"    -> Final rolling success rate: {hist.success_rate(window=30):.1%}")

    # 5. Final Evaluation (greedy, zero exploration)
    print("\n[5] Zero-Exploration Evaluation (50 test episodes each)...")
    print(f"  {'Agent':<25} {'Success Rate':>14} {'Mean Reward':>14} {'Mean Steps':>12}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*12}")

    for name, agent in agents.items():
        eval_metrics = evaluate_agent(agent=agent, env=env, n_episodes=50)
        print(
            f"  {name:<25} "
            f"{eval_metrics['success_rate']:>13.1%} "
            f"{eval_metrics['mean_reward']:>14.2f} "
            f"{eval_metrics['mean_steps']:>12.1f}"
        )

    print("\n" + "=" * 70)
    print("  QUICKSTART COMPLETED SUCCESSFULLY!")
    print("  To run full experiments with multiple seeds and generate figures:")
    print("    python scripts/run_experiments.py --mode quick")
    print("=" * 70)


if __name__ == "__main__":
    main()
