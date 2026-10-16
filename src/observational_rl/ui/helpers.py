"""
UI helper functions for the Streamlit web interface.

Provides:
- Presets and environment creation
- Agent initialization and training wrappers with progress callbacks
- Step-by-step interactive simulation logic
- Matplotlib rendering of GridWorld state and trajectory
- Result loading and figure discovery
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from observational_rl.agents.base import BaseAgent
from observational_rl.agents.behavioral_cloning import BehavioralCloningAgent
from observational_rl.agents.hybrid_orl import HybridORLAgent
from observational_rl.agents.q_learning import QLearningAgent
from observational_rl.demonstrations.expert import ExpertDemonstrationGenerator
from observational_rl.environments.gridworld import ACTION_NAMES, GridWorldEnv
from observational_rl.training.trainer import EpisodeRecord, TrainingHistory, run_episode


PRESET_ENVS: Dict[str, Dict[str, Any]] = {
    "Facile (5x5, 2 obstacles)": {
        "grid_size": 5,
        "n_obstacles": 2,
        "obstacle_seed": 42,
        "max_steps": 50,
        "name": "easy",
    },
    "Moyen (7x7, 6 obstacles)": {
        "grid_size": 7,
        "n_obstacles": 6,
        "obstacle_seed": 42,
        "max_steps": 100,
        "name": "medium",
    },
    "Difficile (9x9, 15 obstacles)": {
        "grid_size": 9,
        "n_obstacles": 15,
        "obstacle_seed": 42,
        "max_steps": 150,
        "name": "hard",
    },
}

ACTION_NAMES_FR: Dict[int, str] = {
    0: "Haut (↑)",
    1: "Droite (→)",
    2: "Bas (↓)",
    3: "Gauche (←)",
}


def create_ui_environment(
    preset_name: str = "Facile (5x5, 2 obstacles)",
    custom_size: Optional[int] = None,
    custom_obstacles: Optional[int] = None,
    obstacle_seed: int = 42,
    max_steps: Optional[int] = None,
) -> GridWorldEnv:
    """
    Create a GridWorld environment based on preset or custom parameters.
    """
    if preset_name in PRESET_ENVS:
        cfg = PRESET_ENVS[preset_name]
        grid_size = custom_size or cfg["grid_size"]
        n_obstacles = custom_obstacles if custom_obstacles is not None else cfg["n_obstacles"]
        env_max_steps = max_steps or cfg["max_steps"]
        name = cfg["name"]
    else:
        grid_size = custom_size or 5
        n_obstacles = custom_obstacles or 2
        env_max_steps = max_steps or 50
        name = "custom"

    return GridWorldEnv(
        grid_size=grid_size,
        n_obstacles=n_obstacles,
        start_pos=(0, 0),
        goal_pos=(grid_size - 1, grid_size - 1),
        max_steps=env_max_steps,
        obstacle_seed=obstacle_seed,
        name=name,
    )


def create_ui_agent(
    agent_type: str,
    env: GridWorldEnv,
    seed: int = 42,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_end: float = 0.05,
    bc_epochs: int = 30,
    bc_learning_rate: float = 0.005,
    bc_hidden_dims: Optional[List[int]] = None,
) -> BaseAgent:
    """
    Create an agent instance matching the specified type and parameters.
    """
    if bc_hidden_dims is None:
        bc_hidden_dims = [32, 32]

    state_size = env.state_space_size
    action_size = env.action_space_size

    if agent_type == "Q-Learning":
        return QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_start=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_end=epsilon_end,
            seed=seed,
            name="Q-Learning",
        )
    elif agent_type == "Behavioral Cloning":
        return BehavioralCloningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=bc_learning_rate,
            hidden_dims=bc_hidden_dims,
            batch_size=16,
            n_epochs=bc_epochs,
            seed=seed,
            name="Behavioral Cloning",
        )
    elif agent_type == "Hybrid ORL":
        return HybridORLAgent(
            state_size=state_size,
            action_size=action_size,
            rl_learning_rate=learning_rate,
            rl_discount_factor=discount_factor,
            rl_epsilon_start=0.3,
            rl_epsilon_decay=epsilon_decay,
            rl_epsilon_end=epsilon_end,
            bc_learning_rate=bc_learning_rate,
            bc_hidden_dims=bc_hidden_dims,
            bc_epochs=bc_epochs,
            seed=seed,
            name="Hybrid ORL (BC->QL)",
        )
    else:
        raise ValueError(f"Agent inconnu : {agent_type}")


def train_ui_agent(
    agent: BaseAgent,
    env: GridWorldEnv,
    n_episodes: int,
    n_demos: int = 30,
    demo_quality: str = "optimal",
    noise_prob: float = 0.0,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> TrainingHistory:
    """
    Train an agent with an optional callback for live UI updates.

    Callback signature: progress_callback(current_episode, total_episodes, rolling_sr, rolling_reward)
    """
    # Phase 1: Pre-training for BC and Hybrid agents
    if isinstance(agent, (BehavioralCloningAgent, HybridORLAgent)):
        generator = ExpertDemonstrationGenerator(env=env, seed=agent.seed)
        demos = generator.generate(
            n_demos=n_demos,
            quality=demo_quality,
            noise_prob=noise_prob if demo_quality == "suboptimal" else 0.0,
        )
        agent.pre_train(demos)

    # Phase 2: Online interaction loop
    history = TrainingHistory(
        agent_name=agent.name,
        env_name=env.name,
        seed=agent.seed,
    )

    for ep in range(n_episodes):
        record = run_episode(agent, env, training=True)
        record.episode = ep
        history.records.append(record)

        if progress_callback is not None and (ep + 1) % max(1, n_episodes // 100) == 0:
            window = min(30, len(history.records))
            sr = history.success_rate(window=window)
            r = history.mean_reward(window=window)
            progress_callback(ep + 1, n_episodes, sr, r)

    return history


def step_simulation(
    env: GridWorldEnv,
    agent: BaseAgent,
) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any], str]:
    """
    Advance the environment by a single step using the agent's policy.

    Returns:
        (new_pos, reward, done, info, action_name_fr)
    """
    curr_state = env._pos_to_state(env.agent_pos)
    action = agent.select_action(curr_state, training=False)
    next_state, reward, done, info = env.step(action)
    action_str = ACTION_NAMES_FR.get(action, f"Action {action}")
    return env.agent_pos, reward, done, info, action_str


def render_grid_figure(
    env: GridWorldEnv,
    current_pos: Optional[Tuple[int, int]] = None,
    trajectory: Optional[List[Tuple[int, int]]] = None,
    title: str = "GridWorld",
) -> plt.Figure:
    """
    Render a clean, high-resolution Matplotlib figure of GridWorld.
    Highlights start, goal, obstacles, trajectory trail, and current agent position.
    """
    gs = env.grid_size
    fig, ax = plt.subplots(figsize=(gs * 0.7 + 1.2, gs * 0.7 + 1.2), dpi=100)

    cell_colors = {
        "free": "#F8F9FA",
        "obstacle": "#343A40",
        "start": "#28A745",
        "goal": "#FFC107",
    }

    # Draw grid cells
    for r in range(gs):
        for c in range(gs):
            pos = (r, c)
            if pos in env.obstacles:
                fc = cell_colors["obstacle"]
                label = "X"
                tc = "white"
            elif pos == env.goal_pos:
                fc = cell_colors["goal"]
                label = "G"
                tc = "#212529"
            elif pos == env.start_pos:
                fc = cell_colors["start"]
                label = "S"
                tc = "white"
            else:
                fc = cell_colors["free"]
                label = ""
                tc = "#495057"

            rect = plt.Rectangle(
                (c, gs - r - 1), 1, 1,
                facecolor=fc, edgecolor="#CED4DA", linewidth=1.0
            )
            ax.add_patch(rect)
            if label:
                ax.text(
                    c + 0.5, gs - r - 0.5, label,
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=tc,
                )

    # Draw trajectory path
    if trajectory and len(trajectory) > 1:
        xs = [pos[1] + 0.5 for pos in trajectory]
        ys = [gs - pos[0] - 0.5 for pos in trajectory]
        ax.plot(
            xs, ys,
            color="#007BFF", linestyle="--", linewidth=2.0, alpha=0.7,
            marker="o", markersize=4, markerfacecolor="#0056B3"
        )

    # Highlight current agent position
    agent_p = current_pos if current_pos is not None else env.agent_pos
    ar, ac = agent_p
    agent_circle = plt.Circle(
        (ac + 0.5, gs - ar - 0.5), 0.35,
        facecolor="#6F42C1", edgecolor="#FFFFFF", linewidth=2.0, zorder=5
    )
    ax.add_patch(agent_circle)
    ax.text(
        ac + 0.5, gs - ar - 0.5, "@",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="white", zorder=6
    )

    ax.set_xlim(0, gs)
    ax.set_ylim(0, gs)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)

    legend_items = [
        mpatches.Patch(color=cell_colors["start"], label="Départ (S)"),
        mpatches.Patch(color=cell_colors["goal"], label="Objectif (G)"),
        mpatches.Patch(color=cell_colors["obstacle"], label="Obstacle (X)"),
        mpatches.Patch(color="#6F42C1", label="Agent (@)"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=9,
        frameon=True,
    )

    plt.tight_layout()
    return fig


def load_csv_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Read saved experimental CSV files from results directory.
    """
    data: Dict[str, pd.DataFrame] = {}
    if not results_dir.exists():
        return data

    for file_path in results_dir.glob("*.csv"):
        try:
            df = pd.read_csv(file_path)
            data[file_path.name] = df
        except Exception:
            continue
    return data


def list_available_figures(figures_dir: Path) -> List[Path]:
    """
    List image files available in figures directory.
    """
    if not figures_dir.exists():
        return []
    valid_exts = {".png", ".jpg", ".jpeg", ".svg"}
    return sorted([f for f in figures_dir.iterdir() if f.suffix.lower() in valid_exts])
