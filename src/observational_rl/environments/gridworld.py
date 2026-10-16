"""
Configurable GridWorld Environment for Observational RL experiments.

Design decisions:
- State is represented as a flat integer index (row * grid_size + col)
  enabling tabular Q-learning without additional encoding.
- Obstacles are fixed per configuration; different seeds produce different
  obstacle layouts, ensuring train/test environment separation.
- The BFS expert is placed in this module to keep path planning co-located
  with the environment it operates on.

Action space:
    0 = UP    (-row)
    1 = DOWN  (+row)
    2 = LEFT  (-col)
    3 = RIGHT (+col)
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from observational_rl.environments.base import (
    BaseEnvironment,
    State,
    Action,
    StepResult,
)


# Action definitions: (delta_row, delta_col)
ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}
ACTION_NAMES: Dict[int, str] = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
N_ACTIONS = 4


class GridWorldEnv(BaseEnvironment):
    """
    Configurable rectangular GridWorld environment.

    The grid contains:
    - One start cell (S)
    - One goal cell (G) — reaching it gives reward_goal
    - Zero or more obstacle cells (X) — stepping on one gives reward_obstacle
    - Free cells (.) — each step costs reward_step

    Hitting a wall (out-of-bounds) is penalized but the agent stays put.

    States are encoded as integers: state = row * grid_size + col.
    This flat encoding is directly compatible with tabular Q-tables.

    Args:
        grid_size:     Side length of the square grid.
        n_obstacles:   Number of obstacle cells to place.
        start_pos:     (row, col) of the starting cell.
        goal_pos:      (row, col) of the goal cell.
        max_steps:     Episode step limit.
        reward_goal:   Reward for reaching the goal.
        reward_step:   Per-step penalty (should be negative).
        reward_obstacle: Reward for hitting an obstacle (should be negative).
        reward_wall:   Reward for hitting a wall (should be negative).
        obstacle_seed: Seed for random obstacle placement.
        random_obstacles: If True, re-randomize obstacles each reset().
        name:          Identifier string for this environment variant.

    Example:
        >>> env = GridWorldEnv(grid_size=5, n_obstacles=3)
        >>> state = env.reset(seed=42)
        >>> next_state, reward, done, info = env.step(action=3)
    """

    def __init__(
        self,
        grid_size: int = 5,
        n_obstacles: int = 3,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Optional[Tuple[int, int]] = None,
        max_steps: int = 100,
        reward_goal: float = 10.0,
        reward_step: float = -0.1,
        reward_obstacle: float = -5.0,
        reward_wall: float = -0.5,
        obstacle_seed: int = 42,
        random_obstacles: bool = False,
        name: str = "gridworld",
    ) -> None:
        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")
        if n_obstacles >= grid_size * grid_size - 2:
            raise ValueError("Too many obstacles for grid size.")

        self.grid_size = grid_size
        self.n_obstacles = n_obstacles
        self.start_pos = tuple(start_pos)
        self.goal_pos = tuple(goal_pos) if goal_pos is not None else (grid_size - 1, grid_size - 1)
        self.max_steps = max_steps
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        self.reward_wall = reward_wall
        self.obstacle_seed = obstacle_seed
        self.random_obstacles = random_obstacles
        self.name = name

        # Validate start/goal
        self._validate_position(self.start_pos, "start_pos")
        self._validate_position(self.goal_pos, "goal_pos")
        if self.start_pos == self.goal_pos:
            raise ValueError("start_pos and goal_pos must differ.")

        # Generate fixed obstacles
        self.obstacles: Set[Tuple[int, int]] = set()
        self._place_obstacles(seed=obstacle_seed)

        # Runtime state
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.steps_taken: int = 0
        self._rng = random.Random(obstacle_seed)

    # ──────────────────────────────────────────────────────────────────────
    # BaseEnvironment interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> State:
        """
        Reset environment to initial state.

        Args:
            seed: If provided, used only for random obstacle re-placement
                  when random_obstacles=True.

        Returns:
            Flat integer state index of the start position.
        """
        if self.random_obstacles and seed is not None:
            self._place_obstacles(seed=seed)

        self.agent_pos = self.start_pos
        self.steps_taken = 0
        return self._pos_to_state(self.agent_pos)

    def step(self, action: Action) -> StepResult:
        """
        Execute action and return (next_state, reward, done, info).

        Args:
            action: Integer in [0, 3] representing direction.

        Returns:
            StepResult tuple.

        Raises:
            ValueError: If action is out of range.
        """
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be in {list(ACTIONS.keys())}.")

        dr, dc = ACTIONS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc

        # Check wall collision
        if not self._in_bounds(new_row, new_col):
            reward = self.reward_wall
            done = False
            info = {"event": "wall", "pos": self.agent_pos}
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                done = True
                info["event"] = "timeout"
            return self._pos_to_state(self.agent_pos), reward, done, info

        new_pos = (new_row, new_col)

        # Move agent
        self.agent_pos = new_pos
        self.steps_taken += 1

        # Determine reward and done
        if new_pos == self.goal_pos:
            reward = self.reward_goal
            done = True
            info = {"event": "goal", "steps": self.steps_taken}
        elif new_pos in self.obstacles:
            reward = self.reward_obstacle
            done = False  # Hitting obstacle does NOT end episode
            info = {"event": "obstacle", "pos": new_pos}
        else:
            reward = self.reward_step
            done = False
            info = {"event": "step", "pos": new_pos}

        # Timeout check
        if not done and self.steps_taken >= self.max_steps:
            done = True
            info["event"] = "timeout"

        return self._pos_to_state(self.agent_pos), reward, done, info

    @property
    def action_space_size(self) -> int:
        return N_ACTIONS

    @property
    def state_space_size(self) -> int:
        return self.grid_size * self.grid_size

    def render(self, mode: str = "ansi") -> Optional[str]:
        """
        Render the grid as ASCII art.

        Legend: S=start, G=goal, X=obstacle, @=agent, .=free cell
        """
        lines = []
        border = "+" + "-" * (self.grid_size * 2 - 1) + "+"
        lines.append(border)
        for r in range(self.grid_size):
            row_chars = []
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row_chars.append("@")
                elif pos == self.goal_pos:
                    row_chars.append("G")
                elif pos == self.start_pos:
                    row_chars.append("S")
                elif pos in self.obstacles:
                    row_chars.append("X")
                else:
                    row_chars.append(".")
            lines.append("|" + " ".join(row_chars) + "|")
        lines.append(border)
        result = "\n".join(lines)
        if mode == "ansi":
            print(result)
        return result

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary."""
        return {
            "name": self.name,
            "grid_size": self.grid_size,
            "n_obstacles": self.n_obstacles,
            "start_pos": list(self.start_pos),
            "goal_pos": list(self.goal_pos),
            "max_steps": self.max_steps,
            "reward_goal": self.reward_goal,
            "reward_step": self.reward_step,
            "reward_obstacle": self.reward_obstacle,
            "reward_wall": self.reward_wall,
            "obstacle_seed": self.obstacle_seed,
            "obstacles": [list(o) for o in sorted(self.obstacles)],
        }

    # ──────────────────────────────────────────────────────────────────────
    # BFS path-finding (used by expert demonstration generator)
    # ──────────────────────────────────────────────────────────────────────

    def find_shortest_path(self) -> Optional[List[Action]]:
        """
        Find shortest path from start to goal using BFS.

        Obstacles are impassable. Walls are not traversable.
        Returns None if no path exists.

        Returns:
            List of actions (integers) forming the shortest path,
            or None if unreachable.
        """
        start = self.start_pos
        goal = self.goal_pos
        queue: deque = deque([(start, [])])
        visited: Set[Tuple[int, int]] = {start}

        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path

            for action, (dr, dc) in ACTIONS.items():
                nr, nc = pos[0] + dr, pos[1] + dc
                new_pos = (nr, nc)
                if (
                    self._in_bounds(nr, nc)
                    and new_pos not in visited
                    and new_pos not in self.obstacles
                ):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [action]))

        return None  # No path found

    def is_solvable(self) -> bool:
        """Return True if there exists a path from start to goal."""
        return self.find_shortest_path() is not None

    # ──────────────────────────────────────────────────────────────────────
    # State / position utilities
    # ──────────────────────────────────────────────────────────────────────

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert (row, col) to flat integer state index."""
        return pos[0] * self.grid_size + pos[1]

    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flat integer state index to (row, col)."""
        return (state // self.grid_size, state % self.grid_size)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _validate_position(self, pos: Tuple[int, int], name: str) -> None:
        if not self._in_bounds(pos[0], pos[1]):
            raise ValueError(
                f"{name}={pos} is out of bounds for grid_size={self.grid_size}"
            )

    def _place_obstacles(self, seed: int) -> None:
        """
        Place obstacles randomly, avoiding start and goal positions.

        Args:
            seed: Random seed for reproducible placement.
        """
        rng = random.Random(seed)
        forbidden = {self.start_pos, self.goal_pos}
        candidates = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in forbidden
        ]
        n = min(self.n_obstacles, len(candidates))
        self.obstacles = set(rng.sample(candidates, n))

    # ──────────────────────────────────────────────────────────────────────
    # Factory class methods
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GridWorldEnv":
        """
        Instantiate GridWorldEnv from a configuration dictionary.

        Args:
            config: Dict with keys matching __init__ parameters.

        Returns:
            Configured GridWorldEnv instance.

        Example:
            >>> cfg = {"grid_size": 8, "n_obstacles": 5}
            >>> env = GridWorldEnv.from_config(cfg)
        """
        # Remove keys not in __init__
        known_keys = {
            "grid_size", "n_obstacles", "start_pos", "goal_pos",
            "max_steps", "reward_goal", "reward_step", "reward_obstacle",
            "reward_wall", "obstacle_seed", "random_obstacles", "name",
        }
        filtered = {k: v for k, v in config.items() if k in known_keys}

        # Convert list positions to tuples
        for key in ("start_pos", "goal_pos"):
            if key in filtered and isinstance(filtered[key], list):
                filtered[key] = tuple(filtered[key])

        return cls(**filtered)
