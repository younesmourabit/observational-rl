"""
Base agent interface for Observational RL experiments.

All agents share a common interface enabling uniform training loops,
evaluation protocols, and comparison between methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all learning agents.

    Subclasses must implement: select_action, update, reset.
    Optional override: pre_train (for demonstration-based initialization).

    Args:
        state_size:   Number of states in the environment.
        action_size:  Number of available actions.
        seed:         Random seed for reproducibility.
        name:         Human-readable agent identifier.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 42,
        name: str = "BaseAgent",
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.name = name
        self._rng = np.random.default_rng(seed)
        self.episode_count: int = 0
        self.step_count: int = 0

    @abstractmethod
    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select an action given the current state.

        Args:
            state:    Current state index.
            training: If True, exploration may be used.

        Returns:
            Integer action index.
        """
        ...

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> Optional[float]:
        """
        Update the agent's internal model after a transition.

        Args:
            state, action, reward, next_state, done: Standard SARS'D tuple.

        Returns:
            Loss value (if applicable, else None).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset per-episode state (e.g., eligibility traces)."""
        ...

    def on_episode_end(self) -> None:
        """Called at the end of each episode. Override for custom behavior."""
        self.episode_count += 1

    def pre_train(self, demonstrations: Any) -> Dict[str, Any]:
        """
        Optional pre-training phase using demonstrations.

        Default implementation does nothing.
        Override in agents that support demonstration-based initialization.

        Args:
            demonstrations: List of Demonstration objects.

        Returns:
            Dict with training statistics.
        """
        return {}

    def get_config(self) -> Dict[str, Any]:
        """Return serializable agent configuration."""
        return {
            "name": self.name,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "seed": self.seed,
        }

    def get_greedy_policy(self) -> Optional[np.ndarray]:
        """
        Return the greedy policy as an array of shape (state_size,).
        Override in agents that maintain an explicit policy.
        """
        return None
