"""
Unit tests for Streamlit UI helpers.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pytest

from observational_rl.environments.gridworld import GridWorldEnv
from observational_rl.ui.helpers import (
    PRESET_ENVS,
    create_ui_environment,
    create_ui_agent,
    train_ui_agent,
    step_simulation,
    render_grid_figure,
    load_csv_results,
    list_available_figures,
)


class TestUIHelpers:
    def test_create_ui_environment_presets(self):
        for preset_name in PRESET_ENVS:
            env = create_ui_environment(preset_name=preset_name)
            assert isinstance(env, GridWorldEnv)
            assert env.grid_size == PRESET_ENVS[preset_name]["grid_size"]
            assert len(env.obstacles) == PRESET_ENVS[preset_name]["n_obstacles"]

    def test_create_ui_environment_custom(self):
        env = create_ui_environment(
            preset_name="Personnalisé",
            custom_size=6,
            custom_obstacles=3,
            obstacle_seed=123,
            max_steps=60,
        )
        assert env.grid_size == 6
        assert len(env.obstacles) == 3
        assert env.max_steps == 60

    def test_create_ui_agents(self):
        env = create_ui_environment("Facile (5x5, 2 obstacles)")
        for agent_type in ["Q-Learning", "Behavioral Cloning", "Hybrid ORL"]:
            agent = create_ui_agent(agent_type, env, seed=42)
            assert agent.name is not None
            assert hasattr(agent, "select_action")

    def test_create_ui_agent_invalid(self):
        env = create_ui_environment("Facile (5x5, 2 obstacles)")
        with pytest.raises(ValueError):
            create_ui_agent("AgentInexistant", env)

    def test_step_simulation(self):
        env = create_ui_environment("Facile (5x5, 2 obstacles)")
        agent = create_ui_agent("Q-Learning", env, seed=42)
        initial_pos = env.agent_pos
        new_pos, reward, done, info, action_name = step_simulation(env, agent)

        assert isinstance(new_pos, tuple)
        assert len(new_pos) == 2
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(action_name, str)
        assert len(action_name) > 0

    def test_train_ui_agent_with_callback(self):
        env = create_ui_environment("Facile (5x5, 2 obstacles)")
        agent = create_ui_agent("Q-Learning", env, seed=42)

        callbacks_called = []

        def callback(ep, total, sr, r):
            callbacks_called.append((ep, total, sr, r))

        history = train_ui_agent(
            agent=agent,
            env=env,
            n_episodes=20,
            progress_callback=callback,
        )
        assert len(history.records) == 20
        assert len(callbacks_called) > 0

    def test_render_grid_figure(self):
        env = create_ui_environment("Facile (5x5, 2 obstacles)")
        traj = [(0, 0), (0, 1), (1, 1)]
        fig = render_grid_figure(env, current_pos=(1, 1), trajectory=traj, title="Test Grid")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_load_csv_results_and_figures(self, tmp_path):
        # Test empty
        results = load_csv_results(tmp_path)
        assert results == {}

        figs = list_available_figures(tmp_path)
        assert figs == []

        # Test with files
        csv_file = tmp_path / "summary.csv"
        csv_file.write_text("agent,reward\nQL,10.0\n", encoding="utf-8")
        img_file = tmp_path / "plot.png"
        img_file.write_bytes(b"dummy")

        results = load_csv_results(tmp_path)
        assert "summary.csv" in results
        assert len(results["summary.csv"]) == 1

        figs = list_available_figures(tmp_path)
        assert len(figs) == 1
        assert figs[0].name == "plot.png"
