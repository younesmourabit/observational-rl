"""
Visualization module for Observational RL experiments.

Produces publication-quality figures using Matplotlib.
All figures are saved to the figures/ directory.

Figure types:
    1. learning_curves:  Training success rate / reward over episodes.
    2. comparison_bar:   Final performance bar chart across agents.
    3. convergence:      Episodes-to-threshold comparison.
    4. demo_ablation:    Performance vs. number of demonstrations.
    5. heatmap:          Q-table or visit frequency visualization.
    6. gridworld_render: Static visualization of a GridWorld layout.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Style configuration
# ─────────────────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "Q-Learning": "#E74C3C",           # Red
    "Behavioral Cloning": "#3498DB",   # Blue
    "Hybrid ORL (BC→QL)": "#2ECC71",   # Green
}

AGENT_LINESTYLES = {
    "Q-Learning": "-",
    "Behavioral Cloning": "--",
    "Hybrid ORL (BC→QL)": "-.",
}

FIGURE_DPI = 150
FIGURE_SIZE_WIDE = (12, 5)
FIGURE_SIZE_SQUARE = (7, 6)


def _setup_style() -> None:
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
    })


def _get_agent_color(agent_name: str) -> str:
    """Return color for an agent, falling back to a generic color."""
    for key, color in AGENT_COLORS.items():
        if key in agent_name:
            return color
    return "#7F8C8D"  # Default gray


def _get_agent_linestyle(agent_name: str) -> str:
    """Return linestyle for an agent."""
    for key, ls in AGENT_LINESTYLES.items():
        if key in agent_name:
            return ls
    return ":"


def save_figure(fig: plt.Figure, path: str, tight: bool = True) -> None:
    """Save figure and log the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Learning curves (success rate over episodes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    curves: Dict[str, Dict[str, Any]],
    metric: str = "success",
    env_name: str = "",
    window: int = 50,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot mean ± std learning curves for multiple agents.

    Args:
        curves:      Dict mapping agent_name → {episodes, mean, std}.
        metric:      Metric label for y-axis.
        env_name:    Environment name for title.
        window:      Smoothing window (for display only).
        output_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for agent_name, data in curves.items():
        episodes = data["episodes"]
        mean = np.array(data["mean"])
        std = np.array(data["std"])
        color = _get_agent_color(agent_name)
        ls = _get_agent_linestyle(agent_name)

        ax.plot(episodes, mean, color=color, linestyle=ls,
                label=agent_name, linewidth=2.2)
        ax.fill_between(episodes, mean - std, mean + std,
                        color=color, alpha=0.15)

    y_label = {
        "success": "Success Rate (rolling mean)",
        "total_reward": "Cumulative Reward (rolling mean)",
        "n_steps": "Steps per Episode (rolling mean)",
    }.get(metric, metric)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Learning Curves — {env_name}\n"
        f"(shaded area = ±1 std across seeds, window={window})"
    )
    ax.legend(loc="lower right")

    if metric == "success":
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.6,
                   label="Success threshold (0.7)")

    if output_path:
        save_figure(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Final performance comparison (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_comparison(
    summaries: List[Dict[str, Any]],
    metric: str = "train_success_rate_mean",
    std_key: str = "train_success_rate_std",
    env_name: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing final performance of agents.

    Args:
        summaries:   List of summary dicts (from summarize_experiment).
        metric:      Key for the bar height.
        std_key:     Key for error bars.
        env_name:    Environment name for title.
        output_path: Save path.

    Returns:
        Matplotlib Figure.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    agent_names = [s["agent_name"] for s in summaries]
    means = [s.get(metric) or 0 for s in summaries]
    stds = [s.get(std_key) or 0 for s in summaries]
    colors = [_get_agent_color(n) for n in agent_names]

    x = np.arange(len(agent_names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="white", linewidth=1.2, width=0.5)

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=10)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Final Performance Comparison — {env_name}")

    if "success_rate" in metric:
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.6)

    if output_path:
        save_figure(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Episodes-to-convergence comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence_comparison(
    summaries: List[Dict[str, Any]],
    env_name: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of episodes needed to reach success threshold.
    Agents that never reach the threshold are shown with a special marker.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 4))

    agent_names = [s["agent_name"] for s in summaries]
    values = [s.get("episodes_to_threshold_mean") for s in summaries]
    stds = [s.get("episodes_to_threshold_std") or 0 for s in summaries]
    colors = [_get_agent_color(n) for n in agent_names]

    max_eps = max((v for v in values if v is not None), default=500)

    y = np.arange(len(agent_names))
    for i, (name, val, std, color) in enumerate(zip(agent_names, values, stds, colors)):
        if val is not None:
            ax.barh(i, val, xerr=std, capsize=4, color=color,
                    edgecolor="white", linewidth=1.2, height=0.5)
            ax.text(val + std + 5, i, f"{val:.0f}", va="center", fontsize=10)
        else:
            ax.barh(i, max_eps * 1.1, color=color, alpha=0.3, height=0.5)
            ax.text(5, i, "Never reached threshold", va="center", fontsize=9,
                    color="gray", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(agent_names)
    ax.set_xlabel("Training Episodes to Reach 70% Success Rate")
    ax.set_title(f"Sample Efficiency (Convergence) — {env_name}")
    ax.invert_yaxis()

    if output_path:
        save_figure(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Demonstration quantity ablation
# ─────────────────────────────────────────────────────────────────────────────

def plot_demo_quantity_ablation(
    ablation_data: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "train_success_rate_mean",
    env_name: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot performance vs. number of demonstrations for each agent.

    Args:
        ablation_data: Dict mapping agent_name → {n_demos → {metric_mean, metric_std}}.
        metric:        Metric to plot.
        env_name:      Environment name for title.
        output_path:   Save path.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    for agent_name, data_by_n in ablation_data.items():
        n_demos_list = sorted(data_by_n.keys())
        means = [data_by_n[n].get(metric, 0) for n in n_demos_list]
        stds = [data_by_n[n].get(metric.replace("mean", "std"), 0) for n in n_demos_list]
        color = _get_agent_color(agent_name)

        ax.plot(n_demos_list, means, "o-", color=color, label=agent_name)
        ax.fill_between(
            n_demos_list,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Number of Demonstrations")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Effect of Demonstration Quantity — {env_name}")
    ax.legend()
    ax.set_xscale("log")

    if output_path:
        save_figure(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: GridWorld layout visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_gridworld(
    env: Any,  # GridWorldEnv
    agent: Optional[Any] = None,
    title: str = "GridWorld",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a GridWorld environment, optionally overlaid with a policy.

    Args:
        env:         GridWorldEnv instance.
        agent:       If provided, overlay greedy policy arrows.
        title:       Figure title.
        output_path: Save path.
    """
    _setup_style()
    gs = env.grid_size
    fig, ax = plt.subplots(figsize=(gs * 0.8 + 1, gs * 0.8 + 1))

    # Color map
    CELL_COLORS = {
        "free":     "#ECF0F1",
        "obstacle": "#2C3E50",
        "start":    "#27AE60",
        "goal":     "#F39C12",
    }

    for r in range(gs):
        for c in range(gs):
            pos = (r, c)
            if pos in env.obstacles:
                color = CELL_COLORS["obstacle"]
                label = "X"
                text_color = "white"
            elif pos == env.goal_pos:
                color = CELL_COLORS["goal"]
                label = "G"
                text_color = "black"
            elif pos == env.start_pos:
                color = CELL_COLORS["start"]
                label = "S"
                text_color = "white"
            else:
                color = CELL_COLORS["free"]
                label = ""
                text_color = "black"

            rect = plt.Rectangle(
                (c, gs - r - 1), 1, 1,
                color=color, linewidth=0.5, edgecolor="#BDC3C7"
            )
            ax.add_patch(rect)
            if label:
                ax.text(
                    c + 0.5, gs - r - 0.5, label,
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color,
                )

    # Overlay greedy policy arrows
    if agent is not None:
        policy = agent.get_greedy_policy()
        if policy is not None:
            ACTION_ARROWS = {0: (0, 0.35), 1: (0, -0.35), 2: (-0.35, 0), 3: (0.35, 0)}
            for s in range(env.state_space_size):
                r, c = env.state_to_pos(s)
                pos = (r, c)
                if pos in env.obstacles:
                    continue
                a = int(policy[s])
                dx, dy = ACTION_ARROWS[a]
                ax.annotate(
                    "", xy=(c + 0.5 + dx, gs - r - 0.5 + dy),
                    xytext=(c + 0.5, gs - r - 0.5),
                    arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.2),
                )

    ax.set_xlim(0, gs)
    ax.set_ylim(0, gs)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Legend
    legend_items = [
        mpatches.Patch(color=CELL_COLORS["start"], label="Start (S)"),
        mpatches.Patch(color=CELL_COLORS["goal"], label="Goal (G)"),
        mpatches.Patch(color=CELL_COLORS["obstacle"], label="Obstacle (X)"),
        mpatches.Patch(color=CELL_COLORS["free"], label="Free cell"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              bbox_to_anchor=(1.3, 1.0), fontsize=9)

    if output_path:
        save_figure(fig, output_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Multi-environment comparison heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_heatmap(
    data: Dict[str, Dict[str, float]],
    metric_label: str = "Success Rate",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of metric values across (agent × environment).

    Args:
        data:         Dict mapping agent_name → {env_name → value}.
        metric_label: Label for colorbar and title.
        output_path:  Save path.
    """
    _setup_style()
    agents = list(data.keys())
    envs = list({e for d in data.values() for e in d.keys()})
    envs.sort()

    matrix = np.full((len(agents), len(envs)), np.nan)
    for i, agent in enumerate(agents):
        for j, env in enumerate(envs):
            val = data[agent].get(env)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, len(envs) * 1.5), max(4, len(agents) * 1.2)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(envs, rotation=30, ha="right")
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    ax.set_title(f"{metric_label} — Agent × Environment Comparison")

    # Annotate cells
    for i in range(len(agents)):
        for j in range(len(envs)):
            val = matrix[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "N/A"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if not np.isnan(val) else "gray")

    plt.colorbar(im, ax=ax, label=metric_label, shrink=0.8)

    if output_path:
        save_figure(fig, output_path)
    return fig
