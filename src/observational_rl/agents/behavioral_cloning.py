"""
Behavioral Cloning Agent — Supervised Imitation Learning Baseline.

Algorithm:
    Behavioral Cloning (BC) treats imitation learning as supervised
    classification: given a dataset of (state, action) pairs from an expert,
    train a neural network π_θ(a|s) to maximize the likelihood of the
    expert's actions.

        Loss = CrossEntropy(π_θ(s), a_expert)

    BC is the simplest form of imitation learning. It uses a dataset of
    full (s, a, r, s') transitions, but during training only (s, a) pairs
    are used — rewards and next states are ignored.

Scientific limitations of BC:
    1. Distribution shift: The policy is trained on states visited by the
       expert. During execution, any deviation causes compounding errors
       (Ross & Bagnell, 2010 — addressed by DAgger).
    2. No RL: BC does not use the reward signal. If demonstrations are
       suboptimal, BC will imitate the suboptimality.
    3. Cannot improve beyond demonstrations: BC is fundamentally limited
       by the quality of the dataset.

This agent serves as an UPPER baseline for demonstration quality and as a
LOWER baseline for long-horizon performance (due to distribution shift).

References:
    Pomerleau, D.A. (1989). ALVINN: An Autonomous Land Vehicle in a Neural Network.
    Ross, S. & Bagnell, D. (2010). Efficient Reductions for Imitation Learning. AISTATS.
    Ross, S., Gordon, G., & Bagnell, D. (2011). A Reduction of Imitation Learning
        and Structured Prediction to No-Regret Online Learning. AISTATS.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from observational_rl.agents.base import BaseAgent
from observational_rl.demonstrations.expert import Demonstration, Transition

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Feedforward neural network for action classification.

    Maps integer state → one-hot → hidden layers → action logits.
    Using one-hot encoding keeps the model architecture independent
    of the state representation, matching the tabular Q-learning baseline.

    Args:
        state_size:   Dimension of one-hot state encoding.
        action_size:  Number of output classes.
        hidden_dims:  List of hidden layer dimensions.

    Example:
        >>> net = PolicyNetwork(state_size=25, action_size=4, hidden_dims=[64, 64])
        >>> logits = net(torch.tensor([[0.0]*25]))  # batch of 1
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_dims: List[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers = []
        in_dim = state_size
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_size))

        self.network = nn.Sequential(*layers)
        self.state_size = state_size
        self.action_size = action_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (pre-softmax)."""
        return self.network(x)

    def state_to_tensor(self, state: int) -> torch.Tensor:
        """Convert a scalar state index to a one-hot tensor."""
        one_hot = torch.zeros(self.state_size)
        one_hot[state] = 1.0
        return one_hot.unsqueeze(0)  # shape (1, state_size)


class BehavioralCloningAgent(BaseAgent):
    """
    Behavioral Cloning agent using a feedforward neural policy.

    Workflow:
        1. pre_train(demonstrations): Train the policy network on (s, a) pairs
           extracted from demonstration transitions.
        2. select_action(state): Use the trained policy greedily (or with
           ε-greedy during BC-RL experiments).
        3. update(...): BC does NOT update from environment interactions
           (pure imitation — no online RL). Returns None.

    Args:
        state_size:     Number of states.
        action_size:    Number of actions.
        learning_rate:  Adam optimizer learning rate.
        hidden_dims:    Hidden layer sizes for the policy network.
        batch_size:     Mini-batch size for training.
        n_epochs:       Number of training epochs over the demo dataset.
        seed:           Random seed.
        name:           Agent identifier.

    Example:
        >>> agent = BehavioralCloningAgent(state_size=25, action_size=4)
        >>> stats = agent.pre_train(demonstrations)
        >>> action = agent.select_action(state=0, training=False)
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        hidden_dims: Optional[List[int]] = None,
        batch_size: int = 32,
        n_epochs: int = 50,
        seed: int = 42,
        name: str = "Behavioral Cloning",
    ) -> None:
        super().__init__(state_size, action_size, seed, name)

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.lr = learning_rate
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Set torch seed for reproducibility
        torch.manual_seed(seed)

        self.policy_net = PolicyNetwork(state_size, action_size, hidden_dims)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self._is_trained: bool = False

    def pre_train(self, demonstrations: List[Demonstration]) -> Dict[str, Any]:
        """
        Train the policy network on demonstration data.

        Extracts (state, action) pairs from all transition tuples and
        trains the network as a supervised classification problem.

        Args:
            demonstrations: List of Demonstration episodes.

        Returns:
            Dict with training statistics: final_loss, n_samples,
            n_epochs, training_losses.
        """
        # Collect all transitions
        transitions: List[Transition] = [
            t for demo in demonstrations for t in demo.transitions
        ]

        if len(transitions) == 0:
            logger.warning("No transitions found in demonstrations. Skipping pre_train.")
            return {"final_loss": None, "n_samples": 0}

        logger.info(
            f"BC pre-training: {len(transitions)} transitions from "
            f"{len(demonstrations)} demonstrations."
        )

        # Build dataset
        states_list = [t.state for t in transitions]
        actions_list = [t.action for t in transitions]

        # One-hot encode states
        states_onehot = torch.zeros(len(states_list), self.state_size)
        for i, s in enumerate(states_list):
            states_onehot[i, s] = 1.0

        actions_tensor = torch.tensor(actions_list, dtype=torch.long)

        # Ensure deterministic training across instances with the same seed
        torch.manual_seed(self.seed)
        g = torch.Generator()
        g.manual_seed(self.seed)

        dataset = TensorDataset(states_onehot, actions_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=g)

        self.policy_net.train()
        training_losses: List[float] = []

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for states_batch, actions_batch in loader:
                self.optimizer.zero_grad()
                logits = self.policy_net(states_batch)
                loss = self.criterion(logits, actions_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            training_losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.debug(f"  Epoch {epoch+1}/{self.n_epochs} — Loss: {avg_loss:.4f}")

        self.policy_net.eval()
        self._is_trained = True

        final_loss = training_losses[-1] if training_losses else None
        logger.info(f"BC pre-training complete. Final loss: {final_loss:.4f}")

        return {
            "final_loss": final_loss,
            "n_samples": len(transitions),
            "n_epochs": self.n_epochs,
            "training_losses": training_losses,
        }

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using the trained policy network (greedy).

        If the network has not been pre-trained, falls back to random action.

        Args:
            state:    Current state index.
            training: Unused for pure BC (no exploration). Kept for API compatibility.

        Returns:
            Action index (greedy w.r.t. policy logits).
        """
        if not self._is_trained:
            return int(self._rng.integers(0, self.action_size))

        self.policy_net.eval()
        with torch.no_grad():
            x = self.policy_net.state_to_tensor(state)
            logits = self.policy_net(x)
            action = int(torch.argmax(logits, dim=1).item())
        return action

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> Optional[float]:
        """
        BC does NOT update from environment interactions.

        This method is a no-op for pure Behavioral Cloning.
        It exists to satisfy the BaseAgent interface and to allow the BC
        agent to be used in the same training loop as RL agents.
        """
        self.step_count += 1
        return None  # No online update

    def reset(self) -> None:
        """No per-episode state to reset for BC."""
        pass

    def get_action_probabilities(self, state: int) -> np.ndarray:
        """
        Return softmax action probabilities for a given state.

        Useful for visualizing the learned policy distribution.

        Args:
            state: State index.

        Returns:
            Array of shape (action_size,) with action probabilities.
        """
        self.policy_net.eval()
        with torch.no_grad():
            x = self.policy_net.state_to_tensor(state)
            logits = self.policy_net(x)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        return probs

    def get_implied_q_values(self) -> np.ndarray:
        """
        Compute implied Q-values from the BC policy for all states.

        Maps each state through the policy network to get action logits,
        which are used as a proxy for Q-values in the Hybrid ORL agent.

        Returns:
            Array of shape (state_size, action_size).
        """
        self.policy_net.eval()
        q_values = np.zeros((self.state_size, self.action_size))
        with torch.no_grad():
            for s in range(self.state_size):
                x = self.policy_net.state_to_tensor(s)
                logits = self.policy_net(x).squeeze().numpy()
                q_values[s] = logits
        return q_values

    def get_config(self) -> Dict:
        cfg = super().get_config()
        cfg.update({
            "learning_rate": self.lr,
            "hidden_dims": self.hidden_dims,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "is_trained": self._is_trained,
        })
        return cfg
