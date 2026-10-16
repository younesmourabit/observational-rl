"""
Base environment interface for Observational RL experiments.

Defines the abstract contract that all environments must satisfy,
enabling future integration with Gymnasium environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


# Type aliases
State = Any          # Environment-specific state representation
Action = int         # Discrete action index
Reward = float
Done = bool
Info = Dict[str, Any]

# Standard step return type
StepResult = Tuple[State, Reward, Done, Info]


class BaseEnvironment(ABC):
    """
    Abstract base class for all RL environments in this project.

    Follows the classic RL interface (similar to Gymnasium) to allow
    future extension with gymnasium.Env subclasses.

    Subclasses must implement: reset, step, action_space_size, state_space_size.
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> State:
        """
        Reset the environment to its initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial state observation.
        """
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """
        Execute one step in the environment.

        Args:
            action: Integer action from the action space.

        Returns:
            Tuple of (next_state, reward, done, info).
        """
        ...

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of discrete actions available."""
        ...

    @property
    @abstractmethod
    def state_space_size(self) -> int:
        """Total number of states (for tabular methods)."""
        ...

    @abstractmethod
    def render(self, mode: str = "ansi") -> Optional[str]:
        """Render the environment (for debugging)."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Return environment configuration as a dict."""
        return {}
