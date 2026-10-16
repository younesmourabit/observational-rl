"""
Hybrid Observational RL Agent — BC Pre-training + Q-learning Fine-tuning.

Algorithm:
    This agent implements a two-phase learning strategy:

    Phase 1 — Demonstration-based initialization (BC):
        Train a policy network on expert demonstrations using supervised learning
        (identical to BehavioralCloningAgent.pre_train). Extract the implied
        Q-values from the trained network and use them to warm-start the Q-table.

    Phase 2 — RL fine-tuning (Q-learning):
        Continue learning from environment interactions using Q-learning,
        starting from the warm-started Q-table rather than from zeros.
        ε-greedy exploration starts at a lower value than pure RL (since the
        policy is already reasonable), but ε decays identically.

Relationship to Observational Reinforcement Learning:
    The term "Observational RL" in the broader literature refers to agents that
    learn by watching other agents, without direct access to their internal
    reward signals or action labels. In this implementation, we provide full
    (s, a, r, s') tuples — a common simplification in the Learning from
    Demonstrations (LfD) literature.

    The scientific contribution of this framework is the empirical study of
    when and how much demonstration pre-training helps RL fine-tuning — the
    question of "observation utility" under varying conditions.

Scientific hypotheses tested:
    H1: Hybrid ORL converges faster than pure Q-learning (fewer episodes to
        reach a success rate threshold).
    H2: With suboptimal demonstrations, Hybrid ORL may still outperform pure RL
        early on but the gap narrows with more RL episodes.
    H3: With very few demonstrations, BC alone degrades but Hybrid ORL may still
        benefit from partial initialization.
    H4: Pure BC generalizes poorly to test environments; Hybrid ORL's RL phase
        enables adaptation.

Advantages over pure BC:
    - Can improve beyond demonstration quality via RL.
    - Adapts to test environments not seen during BC pre-training.

Advantages over pure Q-learning:
    - Potentially faster initial convergence.
    - Better handling of sparse-reward environments.

Limitations:
    - Requires demonstration data (unavailable in fully online settings).
    - BC pre-training may introduce policy bias if demonstrations are very suboptimal.
    - Hyperparameter sensitivity: BC/RL boundary and ε-start need tuning.

References:
    Hester, T. et al. (2018). Deep Q-learning from Demonstrations. AAAI.
    Nair, A. et al. (2018). Overcoming Exploration in RL with Demonstrations. ICRA.
    Schaal, S. (1996). Learning from Demonstration. NeurIPS.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from observational_rl.agents.base import BaseAgent
from observational_rl.agents.behavioral_cloning import BehavioralCloningAgent
from observational_rl.agents.q_learning import QLearningAgent
from observational_rl.demonstrations.expert import Demonstration

logger = logging.getLogger(__name__)


class HybridORLAgent(BaseAgent):
    """
    Hybrid Observational RL: BC warm-start → Q-learning fine-tuning.

    Internally maintains:
    - A BehavioralCloningAgent for Phase 1 pre-training.
    - A QLearningAgent for Phase 2 fine-tuning (warm-started from BC).

    After pre_train() is called, all subsequent select_action() and update()
    calls delegate to the Q-learning agent.

    Args:
        state_size:         Number of states.
        action_size:        Number of actions.
        bc_learning_rate:   BC neural network learning rate.
        bc_hidden_dims:     BC network hidden layer sizes.
        bc_batch_size:      BC training batch size.
        bc_epochs:          Number of BC training epochs.
        rl_learning_rate:   Q-learning learning rate (α).
        rl_discount_factor: Q-learning discount factor (γ).
        rl_epsilon_start:   Starting ε for RL phase (typically lower than pure RL).
        rl_epsilon_end:     Minimum ε.
        rl_epsilon_decay:   ε decay per episode.
        q_init_from_bc:     If True, warm-start Q-table from BC policy.
                            If False, use BC policy only during RL exploration.
        seed:               Random seed.
        name:               Agent identifier.

    Example:
        >>> agent = HybridORLAgent(state_size=25, action_size=4)
        >>> stats = agent.pre_train(demonstrations)   # Phase 1
        >>> # Phase 2: normal RL training loop
        >>> action = agent.select_action(state=0, training=True)
        >>> agent.update(state=0, action=action, reward=-0.1, next_state=1, done=False)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        bc_learning_rate: float = 0.001,
        bc_hidden_dims: Optional[List[int]] = None,
        bc_batch_size: int = 32,
        bc_epochs: int = 30,
        rl_learning_rate: float = 0.1,
        rl_discount_factor: float = 0.99,
        rl_epsilon_start: float = 0.3,
        rl_epsilon_end: float = 0.05,
        rl_epsilon_decay: float = 0.998,
        q_init_from_bc: bool = True,
        seed: int = 42,
        name: str = "Hybrid ORL (BC→QL)",
    ) -> None:
        super().__init__(state_size, action_size, seed, name)

        if bc_hidden_dims is None:
            bc_hidden_dims = [64, 64]

        self.q_init_from_bc = q_init_from_bc

        # Phase 1: BC pre-trainer
        self._bc_agent = BehavioralCloningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=bc_learning_rate,
            hidden_dims=bc_hidden_dims,
            batch_size=bc_batch_size,
            n_epochs=bc_epochs,
            seed=seed,
            name="BC (internal)",
        )

        # Phase 2: Q-learning fine-tuner
        self._ql_agent = QLearningAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=rl_learning_rate,
            discount_factor=rl_discount_factor,
            epsilon_start=rl_epsilon_start,
            epsilon_end=rl_epsilon_end,
            epsilon_decay=rl_epsilon_decay,
            seed=seed,
            name="QL (internal)",
        )

        self._bc_pretrained: bool = False
        self._bc_stats: Dict[str, Any] = {}

    def pre_train(self, demonstrations: List[Demonstration]) -> Dict[str, Any]:
        """
        Phase 1: Train BC policy and warm-start Q-table.

        Steps:
            1. Run BC training on demonstration (s, a) pairs.
            2. Extract implied Q-values from trained BC network.
            3. Warm-start the QL agent's Q-table with these values.

        Args:
            demonstrations: Expert demonstration episodes.

        Returns:
            BC training statistics dict.
        """
        logger.info("=== Hybrid ORL: Phase 1 — BC Pre-training ===")
        bc_stats = self._bc_agent.pre_train(demonstrations)
        self._bc_stats = bc_stats
        self._bc_pretrained = True

        if self.q_init_from_bc and bc_stats.get("n_samples", 0) > 0:
            logger.info("Warm-starting Q-table from BC policy logits...")
            implied_q = self._bc_agent.get_implied_q_values()
            self._ql_agent.initialize_from_q_values(implied_q)
            logger.info("Q-table warm-start complete.")
        else:
            logger.info("Skipping Q-table warm-start (q_init_from_bc=False or no data).")

        logger.info("=== Phase 1 complete. Phase 2 (RL) begins. ===")
        return bc_stats

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using the Q-learning policy (Phase 2).

        Args:
            state:    Current state.
            training: If True, apply ε-greedy exploration.

        Returns:
            Action index.
        """
        return self._ql_agent.select_action(state, training=training)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> Optional[float]:
        """
        Update Q-table via Q-learning (Phase 2).

        Args:
            state, action, reward, next_state, done: SARS'D transition.

        Returns:
            Absolute TD error.
        """
        loss = self._ql_agent.update(state, action, reward, next_state, done)
        self.step_count += 1
        return loss

    def reset(self) -> None:
        """Delegate per-episode reset to Q-learning agent."""
        self._ql_agent.reset()

    def on_episode_end(self) -> None:
        """Decay ε via Q-learning agent."""
        super().on_episode_end()
        self._ql_agent.on_episode_end()

    def get_greedy_policy(self) -> np.ndarray:
        """Return greedy policy from the Q-learning agent."""
        return self._ql_agent.get_greedy_policy()

    @property
    def q_table(self) -> np.ndarray:
        """Expose Q-table for inspection and visualization."""
        return self._ql_agent.q_table

    @property
    def epsilon(self) -> float:
        """Current exploration rate of the RL phase."""
        return self._ql_agent.epsilon

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg.update({
            "bc_epochs": self._bc_agent.n_epochs,
            "bc_pretrained": self._bc_pretrained,
            "q_init_from_bc": self.q_init_from_bc,
            "rl_lr": self._ql_agent.lr,
            "rl_gamma": self._ql_agent.gamma,
            "rl_epsilon_start": self._ql_agent.epsilon_start,
        })
        return cfg
