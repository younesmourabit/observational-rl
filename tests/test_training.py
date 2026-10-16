"""
Integration tests — training loops and evaluation.

Tests cover:
- Single episode run
- Full training loop (short)
- TrainingHistory metrics
- Evaluation without exploration
- Episodes-to-threshold computation
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from observational_rl.environments.gridworld import GridWorldEnv
from observational_rl.agents.q_learning import QLearningAgent
from observational_rl.agents.behavioral_cloning import BehavioralCloningAgent
from observational_rl.agents.hybrid_orl import HybridORLAgent
from observational_rl.demonstrations.expert import ExpertDemonstrationGenerator
from observational_rl.training.trainer import (
    run_episode,
    train_agent,
    evaluate_agent,
    EpisodeRecord,
    TrainingHistory,
)


@pytest.fixture
def small_env():
    return GridWorldEnv(grid_size=5, n_obstacles=0, start_pos=(0, 0), goal_pos=(4, 4))


@pytest.fixture
def ql_agent_small():
    return QLearningAgent(state_size=25, action_size=4, seed=42)


class TestRunEpisode:
    def test_episode_record_fields(self, ql_agent_small, small_env):
        record = run_episode(ql_agent_small, small_env, training=True)
        assert isinstance(record, EpisodeRecord)
        assert record.n_steps >= 1
        assert record.duration_s >= 0

    def test_episode_increments_step_count(self, ql_agent_small, small_env):
        initial_steps = ql_agent_small.step_count
        run_episode(ql_agent_small, small_env, training=True)
        assert ql_agent_small.step_count > initial_steps

    def test_no_update_when_not_training(self, small_env):
        agent = BehavioralCloningAgent(state_size=25, action_size=4, n_epochs=5)
        gen = ExpertDemonstrationGenerator(env=small_env, seed=42)
        demos = gen.generate_optimal(n_demos=10)
        agent.pre_train(demos)
        record = run_episode(agent, small_env, training=False)
        assert record.losses == []  # BC returns None, so losses list stays empty


class TestTrainAgent:
    def test_training_returns_history(self, ql_agent_small, small_env):
        history = train_agent(ql_agent_small, small_env, n_episodes=20, verbose=False)
        assert isinstance(history, TrainingHistory)
        assert len(history.records) == 20

    def test_episode_indices_sequential(self, ql_agent_small, small_env):
        history = train_agent(ql_agent_small, small_env, n_episodes=10, verbose=False)
        for i, r in enumerate(history.records):
            assert r.episode == i

    def test_success_rate_computable(self, ql_agent_small, small_env):
        history = train_agent(ql_agent_small, small_env, n_episodes=20, verbose=False)
        sr = history.success_rate(window=10)
        assert 0.0 <= sr <= 1.0

    def test_mean_reward_computable(self, ql_agent_small, small_env):
        history = train_agent(ql_agent_small, small_env, n_episodes=20, verbose=False)
        mr = history.mean_reward(window=10)
        # Reward is in (-inf, 10], no specific bound check needed
        assert isinstance(mr, float)

    def test_q_learning_improves_on_easy_env(self):
        """Q-learning should achieve non-zero success on easy env after 500 episodes."""
        env = GridWorldEnv(grid_size=5, n_obstacles=0)
        agent = QLearningAgent(
            state_size=25, action_size=4,
            learning_rate=0.1, discount_factor=0.99,
            epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.99,
            seed=42,
        )
        history = train_agent(agent, env, n_episodes=300, verbose=False)
        # After 300 episodes on obstacle-free 5x5 grid, should solve it
        final_sr = history.success_rate(window=50)
        assert final_sr > 0.3, f"Expected success rate > 0.3, got {final_sr:.3f}"


class TestEvaluateAgent:
    def test_evaluate_returns_dict(self, ql_agent_small, small_env):
        result = evaluate_agent(ql_agent_small, small_env, n_episodes=10)
        assert isinstance(result, dict)

    def test_evaluate_keys(self, ql_agent_small, small_env):
        result = evaluate_agent(ql_agent_small, small_env, n_episodes=10)
        assert "success_rate" in result
        assert "mean_reward" in result
        assert "std_reward" in result
        assert "mean_steps" in result
        assert "n_episodes" in result

    def test_evaluate_success_rate_range(self, ql_agent_small, small_env):
        result = evaluate_agent(ql_agent_small, small_env, n_episodes=20)
        assert 0.0 <= result["success_rate"] <= 1.0

    def test_evaluate_n_episodes(self, ql_agent_small, small_env):
        result = evaluate_agent(ql_agent_small, small_env, n_episodes=15)
        assert result["n_episodes"] == 15


class TestTrainingHistory:
    def test_episodes_to_threshold_when_solved(self):
        """If all episodes succeed, should return first episode in the window."""
        history = TrainingHistory(agent_name="test", env_name="test", seed=42)
        # Add 100 successful episodes
        from observational_rl.training.trainer import EpisodeRecord
        for i in range(100):
            history.records.append(
                EpisodeRecord(episode=i, total_reward=10.0, n_steps=5, success=True)
            )
        result = history.episodes_to_threshold(threshold=0.7, window=50)
        assert result is not None
        assert result <= 100

    def test_episodes_to_threshold_when_never_solved(self):
        history = TrainingHistory(agent_name="test", env_name="test", seed=42)
        from observational_rl.training.trainer import EpisodeRecord
        for i in range(100):
            history.records.append(
                EpisodeRecord(episode=i, total_reward=-1.0, n_steps=50, success=False)
            )
        result = history.episodes_to_threshold(threshold=0.7, window=50)
        assert result is None

    def test_to_dict_list(self, ql_agent_small, small_env):
        history = train_agent(ql_agent_small, small_env, n_episodes=5, verbose=False)
        rows = history.to_dict_list()
        assert len(rows) == 5
        assert "episode" in rows[0]
        assert "total_reward" in rows[0]
        assert "success" in rows[0]
