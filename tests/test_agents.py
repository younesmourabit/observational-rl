"""
Test suite for all agents.

Tests cover:
- Q-learning: creation, select_action, update, epsilon decay
- Behavioral Cloning: creation, pre_train, select_action
- Hybrid ORL: creation, pre_train + RL, Q-table warm-start
- Common: BaseAgent interface compliance
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from observational_rl.agents.q_learning import QLearningAgent
from observational_rl.agents.behavioral_cloning import BehavioralCloningAgent
from observational_rl.agents.hybrid_orl import HybridORLAgent
from observational_rl.demonstrations.expert import (
    ExpertDemonstrationGenerator,
    Demonstration,
    Transition,
)
from observational_rl.environments.gridworld import GridWorldEnv


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    return GridWorldEnv(grid_size=5, n_obstacles=0, start_pos=(0, 0), goal_pos=(4, 4))


@pytest.fixture
def demos(env):
    gen = ExpertDemonstrationGenerator(env=env, seed=42)
    return gen.generate_optimal(n_demos=10)


@pytest.fixture
def ql_agent():
    return QLearningAgent(state_size=25, action_size=4, seed=42)


@pytest.fixture
def bc_agent():
    return BehavioralCloningAgent(state_size=25, action_size=4, seed=42, n_epochs=5)


@pytest.fixture
def hybrid_agent():
    return HybridORLAgent(state_size=25, action_size=4, seed=42, bc_epochs=5)


# ─────────────────────────────────────────────────────────────────────────────
# Q-Learning
# ─────────────────────────────────────────────────────────────────────────────

class TestQLearningAgent:
    def test_q_table_shape(self, ql_agent):
        assert ql_agent.q_table.shape == (25, 4)

    def test_initial_q_table_zeros(self, ql_agent):
        assert np.allclose(ql_agent.q_table, 0)

    def test_select_action_range(self, ql_agent):
        for _ in range(20):
            a = ql_agent.select_action(state=0, training=True)
            assert 0 <= a < 4

    def test_greedy_action_valid(self, ql_agent):
        a = ql_agent.select_action(state=0, training=False)
        assert 0 <= a < 4

    def test_update_changes_q_table(self, ql_agent):
        q_before = ql_agent.q_table[0, 3].copy()
        ql_agent.update(state=0, action=3, reward=1.0, next_state=1, done=False)
        assert ql_agent.q_table[0, 3] != q_before

    def test_update_returns_loss(self, ql_agent):
        loss = ql_agent.update(state=0, action=3, reward=1.0, next_state=1, done=False)
        assert loss is not None
        assert loss >= 0

    def test_epsilon_decay_on_episode_end(self, ql_agent):
        eps_before = ql_agent.epsilon
        ql_agent.on_episode_end()
        assert ql_agent.epsilon <= eps_before

    def test_epsilon_does_not_go_below_minimum(self):
        agent = QLearningAgent(
            state_size=25, action_size=4, epsilon_end=0.05,
            epsilon_start=0.06, epsilon_decay=0.5
        )
        for _ in range(100):
            agent.on_episode_end()
        assert agent.epsilon >= 0.05

    def test_invalid_learning_rate(self):
        with pytest.raises(ValueError):
            QLearningAgent(state_size=25, action_size=4, learning_rate=1.5)

    def test_invalid_epsilon_order(self):
        with pytest.raises(ValueError):
            QLearningAgent(
                state_size=25, action_size=4,
                epsilon_start=0.1, epsilon_end=0.5
            )

    def test_td_update_terminal(self, ql_agent):
        """At terminal state, Q-target = reward only (no next state value)."""
        ql_agent.q_table[:] = 0
        ql_agent.update(state=0, action=3, reward=10.0, next_state=24, done=True)
        # Expected: Q(0,3) = 0 + lr * (10 - 0) = 1.0
        assert abs(ql_agent.q_table[0, 3] - 1.0) < 1e-6

    def test_greedy_policy_shape(self, ql_agent):
        policy = ql_agent.get_greedy_policy()
        assert policy is not None
        assert policy.shape == (25,)
        assert all(0 <= a < 4 for a in policy)

    def test_initialize_from_q_values(self, ql_agent):
        q_values = np.ones((25, 4)) * 5.0
        ql_agent.initialize_from_q_values(q_values)
        assert np.allclose(ql_agent.q_table, 5.0)

    def test_initialize_wrong_shape_raises(self, ql_agent):
        with pytest.raises(ValueError):
            ql_agent.initialize_from_q_values(np.ones((10, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral Cloning
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralCloningAgent:
    def test_initial_not_trained(self, bc_agent):
        assert not bc_agent._is_trained

    def test_random_action_before_training(self, bc_agent):
        # Should not crash and return valid action
        a = bc_agent.select_action(state=0, training=True)
        assert 0 <= a < 4

    def test_pre_train_sets_trained_flag(self, bc_agent, demos):
        bc_agent.pre_train(demos)
        assert bc_agent._is_trained

    def test_pre_train_returns_stats(self, bc_agent, demos):
        stats = bc_agent.pre_train(demos)
        assert "n_samples" in stats
        assert stats["n_samples"] > 0
        assert "final_loss" in stats

    def test_select_action_after_training(self, bc_agent, demos):
        bc_agent.pre_train(demos)
        a = bc_agent.select_action(state=0, training=False)
        assert 0 <= a < 4

    def test_update_returns_none(self, bc_agent):
        result = bc_agent.update(0, 3, 1.0, 1, False)
        assert result is None

    def test_pre_train_empty_demos(self, bc_agent):
        stats = bc_agent.pre_train([])
        assert stats["n_samples"] == 0

    def test_action_probabilities(self, bc_agent, demos):
        bc_agent.pre_train(demos)
        probs = bc_agent.get_action_probabilities(0)
        assert probs.shape == (4,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)

    def test_implied_q_values_shape(self, bc_agent, demos):
        bc_agent.pre_train(demos)
        q = bc_agent.get_implied_q_values()
        assert q.shape == (25, 4)

    def test_reproducibility(self, demos):
        """Same seed → same actions after training."""
        agent1 = BehavioralCloningAgent(state_size=25, action_size=4, seed=42, n_epochs=5)
        agent2 = BehavioralCloningAgent(state_size=25, action_size=4, seed=42, n_epochs=5)
        agent1.pre_train(demos)
        agent2.pre_train(demos)
        for s in range(25):
            assert agent1.select_action(s, training=False) == agent2.select_action(s, training=False)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid ORL
# ─────────────────────────────────────────────────────────────────────────────

class TestHybridORLAgent:
    def test_initial_q_table_zeros(self, hybrid_agent):
        """Before pre_train, Q-table should be initialized from BC (which is zero/random)."""
        # Before pre_train, BC is untrained → Q is at default init
        assert hybrid_agent.q_table is not None
        assert hybrid_agent.q_table.shape == (25, 4)

    def test_pre_train_changes_q_table(self, hybrid_agent, demos):
        q_before = hybrid_agent.q_table.copy()
        hybrid_agent.pre_train(demos)
        # After BC, Q-table should change from warm-start
        assert not np.allclose(hybrid_agent.q_table, q_before)

    def test_select_action_after_pretrain(self, hybrid_agent, demos):
        hybrid_agent.pre_train(demos)
        a = hybrid_agent.select_action(state=0, training=True)
        assert 0 <= a < 4

    def test_update_changes_q_table(self, hybrid_agent, demos):
        hybrid_agent.pre_train(demos)
        q_before = hybrid_agent.q_table.copy()
        hybrid_agent.update(state=0, action=3, reward=1.0, next_state=1, done=False)
        assert not np.allclose(hybrid_agent.q_table, q_before)

    def test_epsilon_exposed(self, hybrid_agent):
        eps = hybrid_agent.epsilon
        assert 0 <= eps <= 1

    def test_epsilon_decays_on_episode_end(self, hybrid_agent, demos):
        hybrid_agent.pre_train(demos)
        eps_before = hybrid_agent.epsilon
        hybrid_agent.on_episode_end()
        assert hybrid_agent.epsilon <= eps_before

    def test_greedy_policy_after_pretrain(self, hybrid_agent, demos):
        hybrid_agent.pre_train(demos)
        policy = hybrid_agent.get_greedy_policy()
        assert policy.shape == (25,)
        assert all(0 <= a < 4 for a in policy)


# ─────────────────────────────────────────────────────────────────────────────
# Common interface compliance
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("agent_fixture", ["ql_agent", "bc_agent", "hybrid_agent"])
class TestAgentInterface:
    def test_has_name(self, agent_fixture, request):
        agent = request.getfixturevalue(agent_fixture)
        assert isinstance(agent.name, str)
        assert len(agent.name) > 0

    def test_get_config_returns_dict(self, agent_fixture, request):
        agent = request.getfixturevalue(agent_fixture)
        cfg = agent.get_config()
        assert isinstance(cfg, dict)
        assert "name" in cfg
        assert "state_size" in cfg
        assert "action_size" in cfg

    def test_select_action_in_range(self, agent_fixture, request):
        agent = request.getfixturevalue(agent_fixture)
        a = agent.select_action(state=0, training=False)
        assert 0 <= a < agent.action_size

    def test_reset_does_not_crash(self, agent_fixture, request):
        agent = request.getfixturevalue(agent_fixture)
        agent.reset()  # Should not raise

    def test_on_episode_end_increments_count(self, agent_fixture, request):
        agent = request.getfixturevalue(agent_fixture)
        count_before = agent.episode_count
        agent.on_episode_end()
        assert agent.episode_count == count_before + 1
