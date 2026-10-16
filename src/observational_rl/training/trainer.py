"""
Training module — episode-level training loops for all agent types.

Provides a unified training interface that:
- Runs episodes for a given number of steps.
- Collects per-episode metrics (reward, steps, success).
- Calls agent.update() after each step.
- Calls agent.on_episode_end() after each episode.
- Supports periodic evaluation during training.
- Is agnostic to agent type (QL, BC, Hybrid).

The key design principle is separation of training logic from agent internals:
agents receive transitions, training loops manage episodes.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from observational_rl.agents.base import BaseAgent
from observational_rl.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    """
    Record of a single training or evaluation episode.

    Attributes:
        episode:       Episode index (0-indexed).
        total_reward:  Sum of all rewards received.
        n_steps:       Number of environment steps taken.
        success:       True if goal was reached before timeout.
        duration_s:    Wall-clock time for this episode (seconds).
        losses:        List of loss values from agent.update() calls.
    """
    episode: int
    total_reward: float
    n_steps: int
    success: bool
    duration_s: float = 0.0
    losses: List[float] = field(default_factory=list)

    @property
    def mean_loss(self) -> Optional[float]:
        return float(np.mean(self.losses)) if self.losses else None


@dataclass
class TrainingHistory:
    """
    Complete training history for a single agent run.

    Attributes:
        agent_name:   Agent identifier string.
        env_name:     Environment identifier string.
        seed:         Random seed used.
        records:      List of EpisodeRecord for each training episode.
        eval_records: Periodic evaluation records (list of eval summaries).
    """
    agent_name: str
    env_name: str
    seed: int
    records: List[EpisodeRecord] = field(default_factory=list)
    eval_records: List[Dict[str, Any]] = field(default_factory=list)

    def success_rate(self, window: int = 50) -> float:
        """Rolling success rate over the last `window` episodes."""
        recent = self.records[-window:]
        if not recent:
            return 0.0
        return sum(r.success for r in recent) / len(recent)

    def mean_reward(self, window: int = 50) -> float:
        """Rolling mean reward over the last `window` episodes."""
        recent = self.records[-window:]
        if not recent:
            return 0.0
        return float(np.mean([r.total_reward for r in recent]))

    def episodes_to_threshold(
        self,
        threshold: float = 0.7,
        window: int = 50,
    ) -> Optional[int]:
        """
        First episode index where rolling success rate >= threshold.

        Args:
            threshold: Success rate to consider "solved" (default: 0.7).
            window:    Rolling window size.

        Returns:
            Episode index or None if threshold never reached.
        """
        for i in range(window, len(self.records) + 1):
            window_records = self.records[i - window:i]
            rate = sum(r.success for r in window_records) / window
            if rate >= threshold:
                return self.records[i - 1].episode
        return None

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert records to a list of dicts (for CSV export)."""
        return [
            {
                "agent": self.agent_name,
                "env": self.env_name,
                "seed": self.seed,
                "episode": r.episode,
                "total_reward": r.total_reward,
                "n_steps": r.n_steps,
                "success": int(r.success),
                "duration_s": r.duration_s,
                "mean_loss": r.mean_loss,
            }
            for r in self.records
        ]


def run_episode(
    agent: BaseAgent,
    env: BaseEnvironment,
    training: bool = True,
    render: bool = False,
) -> EpisodeRecord:
    """
    Run a single episode and return its record.

    Args:
        agent:    The learning agent.
        env:      The environment (already reset externally if needed).
        training: If True, agent.update() is called; exploration is enabled.
        render:   If True, print ASCII grid each step.

    Returns:
        EpisodeRecord with episode statistics.
    """
    state = env.reset()
    done = False
    total_reward = 0.0
    n_steps = 0
    losses: List[float] = []
    t_start = time.perf_counter()

    while not done:
        action = agent.select_action(state, training=training)
        next_state, reward, done, info = env.step(action)

        if training:
            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)

        total_reward += reward
        n_steps += 1
        state = next_state

        if render:
            env.render()

    duration = time.perf_counter() - t_start
    success = (info.get("event") == "goal")

    return EpisodeRecord(
        episode=agent.episode_count,
        total_reward=total_reward,
        n_steps=n_steps,
        success=success,
        duration_s=duration,
        losses=losses,
    )


def train_agent(
    agent: BaseAgent,
    env: BaseEnvironment,
    n_episodes: int,
    eval_env: Optional[BaseEnvironment] = None,
    eval_interval: int = 50,
    eval_episodes: int = 20,
    verbose: bool = True,
    log_interval: int = 100,
) -> TrainingHistory:
    """
    Full training loop for a single agent.

    Args:
        agent:         The agent to train.
        env:           Training environment.
        n_episodes:    Total number of training episodes.
        eval_env:      Optional separate evaluation environment.
        eval_interval: Evaluate every N training episodes.
        eval_episodes: Number of episodes per evaluation.
        verbose:       Print progress to logger.
        log_interval:  Logging interval in episodes.

    Returns:
        TrainingHistory with complete records.
    """
    history = TrainingHistory(
        agent_name=agent.name,
        env_name=env.name,
        seed=agent.seed,
    )

    logger.info(
        f"Training '{agent.name}' on '{env.name}' "
        f"for {n_episodes} episodes (seed={agent.seed})."
    )

    for ep in range(n_episodes):
        record = run_episode(agent, env, training=True)
        record.episode = ep
        history.records.append(record)
        agent.on_episode_end()

        # Periodic logging
        if verbose and (ep + 1) % log_interval == 0:
            sr = history.success_rate(window=50)
            mr = history.mean_reward(window=50)
            logger.info(
                f"  [{agent.name}] Ep {ep+1}/{n_episodes} | "
                f"SuccessRate(50): {sr:.2f} | AvgReward(50): {mr:.2f}"
            )

        # Periodic evaluation on eval_env
        if eval_env is not None and (ep + 1) % eval_interval == 0:
            eval_summary = evaluate_agent(agent, eval_env, eval_episodes)
            eval_summary["episode"] = ep
            history.eval_records.append(eval_summary)

    logger.info(
        f"Training complete. Final success rate: {history.success_rate():.2f}"
    )
    return history


def evaluate_agent(
    agent: BaseAgent,
    env: BaseEnvironment,
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate agent performance without exploration.

    Args:
        agent:       Agent to evaluate (exploration disabled).
        env:         Evaluation environment.
        n_episodes:  Number of evaluation episodes.

    Returns:
        Dict with: success_rate, mean_reward, std_reward,
        mean_steps, std_steps, n_episodes.
    """
    records: List[EpisodeRecord] = []
    for _ in range(n_episodes):
        record = run_episode(agent, env, training=False)
        records.append(record)

    successes = [r.success for r in records]
    rewards = [r.total_reward for r in records]
    steps = [r.n_steps for r in records]

    return {
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "env": env.name,
        "agent": agent.name,
    }
