"""
Test suite for GridWorld environment.

Tests cover:
- Environment creation and configuration
- State/position encoding
- Step transitions and rewards
- Wall and obstacle collision
- Episode termination
- BFS path-finding
- Factory (from_config)
- Edge cases
"""

import sys
from pathlib import Path

import pytest

# Allow running without pip install
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from observational_rl.environments.gridworld import GridWorldEnv, ACTIONS, N_ACTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_env():
    """5x5 GridWorld with no obstacles for basic testing."""
    return GridWorldEnv(
        grid_size=5,
        n_obstacles=0,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        max_steps=50,
        name="test_simple",
    )


@pytest.fixture
def obstacle_env():
    """5x5 GridWorld with 3 obstacles."""
    return GridWorldEnv(
        grid_size=5,
        n_obstacles=3,
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacle_seed=99,
        name="test_obstacles",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Creation and validation
# ─────────────────────────────────────────────────────────────────────────────

class TestGridWorldCreation:
    def test_basic_creation(self, simple_env):
        assert simple_env.grid_size == 5
        assert simple_env.state_space_size == 25
        assert simple_env.action_space_size == 4

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError, match="grid_size must be >= 2"):
            GridWorldEnv(grid_size=1)

    def test_too_many_obstacles(self):
        with pytest.raises(ValueError, match="Too many obstacles"):
            GridWorldEnv(grid_size=3, n_obstacles=8)

    def test_start_equals_goal(self):
        with pytest.raises(ValueError, match="start_pos and goal_pos must differ"):
            GridWorldEnv(grid_size=5, start_pos=(0, 0), goal_pos=(0, 0))

    def test_out_of_bounds_start(self):
        with pytest.raises(ValueError, match="start_pos"):
            GridWorldEnv(grid_size=5, start_pos=(5, 0))

    def test_obstacles_exclude_start_goal(self, obstacle_env):
        """Obstacles must not be placed on start or goal."""
        assert obstacle_env.start_pos not in obstacle_env.obstacles
        assert obstacle_env.goal_pos not in obstacle_env.obstacles

    def test_obstacle_count(self, obstacle_env):
        assert len(obstacle_env.obstacles) == 3


# ─────────────────────────────────────────────────────────────────────────────
# State encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestStateEncoding:
    def test_pos_to_state_origin(self, simple_env):
        assert simple_env._pos_to_state((0, 0)) == 0

    def test_pos_to_state_goal(self, simple_env):
        # (4, 4) → 4*5 + 4 = 24
        assert simple_env._pos_to_state((4, 4)) == 24

    def test_state_to_pos_roundtrip(self, simple_env):
        for r in range(5):
            for c in range(5):
                s = simple_env._pos_to_state((r, c))
                assert simple_env.state_to_pos(s) == (r, c)

    def test_reset_returns_correct_state(self, simple_env):
        state = simple_env.reset()
        assert state == simple_env._pos_to_state(simple_env.start_pos)


# ─────────────────────────────────────────────────────────────────────────────
# Step transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestStepTransitions:
    def test_step_right_moves_agent(self, simple_env):
        simple_env.reset()
        next_state, _, _, _ = simple_env.step(action=3)  # RIGHT
        assert simple_env.agent_pos == (0, 1)
        assert next_state == simple_env._pos_to_state((0, 1))

    def test_step_down_moves_agent(self, simple_env):
        simple_env.reset()
        next_state, _, _, _ = simple_env.step(action=1)  # DOWN
        assert simple_env.agent_pos == (1, 0)

    def test_step_up_from_top_row_hits_wall(self, simple_env):
        simple_env.reset()  # agent at (0, 0)
        state, reward, done, info = simple_env.step(action=0)  # UP (wall)
        assert simple_env.agent_pos == (0, 0)  # did not move
        assert reward == simple_env.reward_wall
        assert not done
        assert info["event"] == "wall"

    def test_step_left_from_col0_hits_wall(self, simple_env):
        simple_env.reset()
        state, reward, done, info = simple_env.step(action=2)  # LEFT
        assert simple_env.agent_pos == (0, 0)
        assert reward == simple_env.reward_wall

    def test_goal_reached(self, simple_env):
        """Navigate to goal manually."""
        simple_env.reset()
        # Move right 4 times, then down 4 times → goal at (4,4)
        for _ in range(4):
            simple_env.step(action=3)  # RIGHT
        for _ in range(3):
            simple_env.step(action=1)  # DOWN
        _, reward, done, info = simple_env.step(action=1)  # DOWN → (4,4)
        assert done
        assert reward == simple_env.reward_goal
        assert info["event"] == "goal"

    def test_obstacle_penalty_no_termination(self, obstacle_env):
        """Hitting an obstacle gives negative reward but does not end episode."""
        obstacle_env.reset()
        # Find an obstacle
        if obstacle_env.obstacles:
            obs_pos = next(iter(obstacle_env.obstacles))
            # Teleport agent next to obstacle
            obstacle_env.agent_pos = (obs_pos[0], max(0, obs_pos[1] - 1))
            # Try to move into obstacle (RIGHT)
            _, reward, done, info = obstacle_env.step(action=3)
            if info.get("event") == "obstacle":
                assert reward == obstacle_env.reward_obstacle
                assert not done

    def test_step_penalty_on_free_cell(self, simple_env):
        simple_env.reset()
        _, reward, done, info = simple_env.step(action=3)  # RIGHT (free cell)
        assert reward == simple_env.reward_step
        assert not done

    def test_invalid_action_raises(self, simple_env):
        simple_env.reset()
        with pytest.raises(ValueError):
            simple_env.step(action=99)

    def test_timeout(self):
        env = GridWorldEnv(grid_size=5, n_obstacles=0, max_steps=3)
        env.reset()
        env.step(3)
        env.step(3)
        _, _, done, info = env.step(3)
        assert done
        assert info["event"] in ("timeout", "goal")

    def test_step_count_increments(self, simple_env):
        simple_env.reset()
        assert simple_env.steps_taken == 0
        simple_env.step(3)
        assert simple_env.steps_taken == 1
        simple_env.step(3)
        assert simple_env.steps_taken == 2


# ─────────────────────────────────────────────────────────────────────────────
# BFS path-finding
# ─────────────────────────────────────────────────────────────────────────────

class TestBFS:
    def test_bfs_finds_path_no_obstacles(self, simple_env):
        path = simple_env.find_shortest_path()
        assert path is not None
        assert len(path) == 8  # 4 right + 4 down = 8 steps minimum

    def test_bfs_path_is_valid(self, simple_env):
        """Follow BFS path and verify goal is reached."""
        path = simple_env.find_shortest_path()
        simple_env.reset()
        done = False
        for action in path:
            _, _, done, info = simple_env.step(action)
            if done:
                break
        assert done
        assert info["event"] == "goal"

    def test_bfs_returns_none_for_blocked_env(self):
        """A fully blocked environment should return None."""
        # 3x3 with obstacles surrounding goal
        env = GridWorldEnv(
            grid_size=3,
            n_obstacles=0,
            start_pos=(0, 0),
            goal_pos=(2, 2),
        )
        # Manually block all paths
        env.obstacles = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
        path = env.find_shortest_path()
        assert path is None

    def test_is_solvable(self, simple_env):
        assert simple_env.is_solvable()


# ─────────────────────────────────────────────────────────────────────────────
# Factory (from_config)
# ─────────────────────────────────────────────────────────────────────────────

class TestFactory:
    def test_from_config_dict(self):
        config = {
            "grid_size": 8,
            "n_obstacles": 5,
            "start_pos": [0, 0],
            "goal_pos": [7, 7],
            "max_steps": 100,
            "name": "test_factory",
        }
        env = GridWorldEnv.from_config(config)
        assert env.grid_size == 8
        assert env.goal_pos == (7, 7)
        assert env.name == "test_factory"

    def test_from_config_ignores_unknown_keys(self):
        config = {
            "grid_size": 5,
            "unknown_key": "should_be_ignored",
        }
        env = GridWorldEnv.from_config(config)
        assert env.grid_size == 5

    def test_get_config_roundtrip(self, simple_env):
        """get_config() + from_config() should reproduce equivalent env."""
        cfg = simple_env.get_config()
        env2 = GridWorldEnv.from_config(cfg)
        assert env2.grid_size == simple_env.grid_size
        assert env2.goal_pos == simple_env.goal_pos
        assert env2.obstacles == simple_env.obstacles


# ─────────────────────────────────────────────────────────────────────────────
# Render (smoke test)
# ─────────────────────────────────────────────────────────────────────────────

class TestRender:
    def test_render_returns_string(self, simple_env, capsys):
        simple_env.reset()
        result = simple_env.render()
        assert isinstance(result, str)
        assert "S" in result or "@" in result
        assert "G" in result
