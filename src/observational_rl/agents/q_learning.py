"""
Q-Learning Agent — Tabular Reinforcement Learning Baseline.

Algorithm:
    Q-learning (Watkins, 1989; Watkins & Dayan, 1992) is a model-free,
    off-policy temporal-difference control algorithm. It maintains a table
    Q[s, a] and updates it using:

        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

    The convergence guarantee (to the optimal Q*) requires that:
    - all (state, action) pairs are visited infinitely often,
    - the learning rate satisfies standard stochastic approximation conditions.

    We use ε-greedy exploration with exponential decay.

Scientific context:
    This serves as the "pure RL" baseline — no demonstration data is used.
    Comparison with Behavioral Cloning and Hybrid ORL will reveal whether
    prior demonstration data helps (and under what conditions).

Advantages:
    - Simple and well-understood convergence theory.
    - No function approximation needed for small GridWorlds.
    - Tabular → fully interpretable.

Limitations:
    - Requires visiting all states; inefficient for sparse rewards.
    - Does not scale to large/continuous state spaces.
    - No demonstration guidance during initial random exploration.

References:
    Watkins, C.J.C.H. (1989). Learning from Delayed Rewards. PhD thesis.
    Watkins, C.J.C.H. & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279-292.
    Sutton, R.S. & Barto, A.G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
"""

from typing import Dict, Optional

import numpy as np

from observational_rl.agents.base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning agent with ε-greedy exploration.

    Args:
        state_size:       Number of states (grid cells).
        action_size:      Number of actions.
        learning_rate:    α — learning rate for Q-update (default: 0.1).
        discount_factor:  γ — discount factor for future rewards (default: 0.99).
        epsilon_start:    Initial exploration rate (default: 1.0).
        epsilon_end:      Minimum exploration rate (default: 0.05).
        epsilon_decay:    Multiplicative decay per episode end (default: 0.995).
        seed:             Random seed.
        name:             Agent name.

    Example:
        >>> agent = QLearningAgent(state_size=25, action_size=4, seed=42)
        >>> action = agent.select_action(state=0, training=True)
        >>> loss = agent.update(state=0, action=action, reward=-0.1,
        ...                    next_state=1, done=False)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
        name: str = "Q-Learning",
    ) -> None:
        super().__init__(state_size, action_size, seed, name)

        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not 0 < discount_factor <= 1:
            raise ValueError(f"discount_factor must be in (0, 1], got {discount_factor}")
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("Must have 0 <= epsilon_end <= epsilon_start <= 1")

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table to zeros (optimistic: use 0, pessimistic: use -inf)
        # Zero initialization is common and neutral.
        self.q_table: np.ndarray = np.zeros((state_size, action_size), dtype=np.float64)

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state:    Current state index.
            training: If True, apply ε-greedy exploration.

        Returns:
            Action index.
        """
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.action_size))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> Optional[float]:
        """
        Apply the Q-learning update rule.

        TD target = r + γ * max_a' Q(s', a')   (if not done)
        TD target = r                            (if done)
        TD error  = target - Q(s, a)
        Q(s, a)  += α * TD error

        Args:
            state, action, reward, next_state, done: SARS'D transition.

        Returns:
            Absolute TD error (proxy for loss).
        """
        current_q = self.q_table[state, action]

        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = td_target - current_q
        self.q_table[state, action] += self.lr * td_error
        self.step_count += 1

        return float(abs(td_error))

    def reset(self) -> None:
        """Reset per-episode state (Q-table persists across episodes)."""
        pass  # Q-table is persistent; nothing to reset per episode

    def on_episode_end(self) -> None:
        """Decay ε at the end of each episode."""
        super().on_episode_end()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_greedy_policy(self) -> np.ndarray:
        """Return greedy action for each state as array of shape (state_size,)."""
        return np.argmax(self.q_table, axis=1)

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg.update({
            "learning_rate": self.lr,
            "discount_factor": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        })
        return cfg

    def initialize_from_q_values(self, q_values: np.ndarray) -> None:
        """
        Warm-start the Q-table from externally computed Q-values.

        Used by the Hybrid ORL agent to transfer knowledge from
        a pre-trained Behavioral Cloning policy.

        Args:
            q_values: Array of shape (state_size, action_size).
        """
        if q_values.shape != self.q_table.shape:
            raise ValueError(
                f"Q-value shape mismatch: expected {self.q_table.shape}, "
                f"got {q_values.shape}"
            )
        self.q_table = q_values.copy()
