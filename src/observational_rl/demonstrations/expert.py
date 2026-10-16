"""
Expert Demonstration Generator for Observational RL.

Scientific clarification on demonstration types:
- "Optimal demonstrations": BFS finds the true shortest path → actions are
  directly provided to the learning agent. This is supervised demonstration
  data, NOT visual observational learning.
- "Suboptimal demonstrations": BFS path with random action substitutions at
  a given noise rate, simulating an imperfect teacher.
- "Limited demonstrations": Full BFS quality but very few episodes.

A Transition is the fundamental unit of experience:
    (state, action, reward, next_state, done)

This is the format used for Behavioral Cloning and Hybrid ORL pre-training.

References:
- BFS for shortest path: Dijkstra (1959), Hart et al. (1968)
- Behavioral Cloning from demonstrations: Pomerleau (1989), Ross et al. (2011)
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from observational_rl.environments.gridworld import GridWorldEnv


@dataclass
class Transition:
    """
    A single (s, a, r, s', done) transition from an episode.

    This is the standard unit of experience in RL and the fundamental
    building block for Behavioral Cloning training data.

    Attributes:
        state:       Integer state index before action.
        action:      Integer action taken.
        reward:      Reward received after action.
        next_state:  Integer state index after action.
        done:        Whether the episode ended.
    """
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


@dataclass
class Demonstration:
    """
    A complete episode demonstration from an expert.

    Attributes:
        transitions: Ordered list of transitions in the episode.
        total_reward: Sum of all rewards.
        n_steps:     Number of steps taken.
        success:     Whether the goal was reached.
        quality:     Label describing demonstration quality.
    """
    transitions: List[Transition]
    total_reward: float
    n_steps: int
    success: bool
    quality: str = "optimal"

    @property
    def states(self) -> List[int]:
        return [t.state for t in self.transitions]

    @property
    def actions(self) -> List[int]:
        return [t.action for t in self.transitions]

    @property
    def rewards(self) -> List[float]:
        return [t.reward for t in self.transitions]


class ExpertDemonstrationGenerator:
    """
    Generates expert demonstrations using BFS on GridWorld environments.

    The BFS expert computes the optimal (shortest) path in the current
    obstacle configuration. This is NOT a learning agent — it has full
    knowledge of the grid structure, which is why its demonstrations
    are called "supervised" rather than purely observational.

    Scientific note on terminology:
        True "observational RL" would have the agent observe state sequences
        without access to the expert's internal actions. Here we provide
        full (s, a, r, s') tuples — a standard assumption in Behavioral
        Cloning and Learning from Demonstrations (LfD) literature.

    Args:
        env:       GridWorld environment to generate demonstrations for.
        seed:      Random seed for noise injection reproducibility.

    Example:
        >>> env = GridWorldEnv(grid_size=5, n_obstacles=2)
        >>> gen = ExpertDemonstrationGenerator(env, seed=42)
        >>> demos = gen.generate(n_demos=20, quality="optimal")
    """

    def __init__(self, env: GridWorldEnv, seed: int = 42) -> None:
        self.env = env
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def generate(
        self,
        n_demos: int,
        quality: str = "optimal",
        noise_prob: float = 0.0,
    ) -> List[Demonstration]:
        """
        Generate a set of expert demonstrations.

        Args:
            n_demos:    Number of demonstration episodes to generate.
            quality:    Label for the quality type (used in logging/metrics).
            noise_prob: Probability of replacing each expert action with a
                        random action. 0.0 = optimal, 0.2 = 20% suboptimal.

        Returns:
            List of Demonstration objects. Only successful episodes are
            included; if BFS finds no path, an empty list is returned.

        Raises:
            RuntimeError: If the environment has no solvable path.
        """
        if not self.env.is_solvable():
            raise RuntimeError(
                f"Environment '{self.env.name}' has no path from "
                f"{self.env.start_pos} to {self.env.goal_pos}. "
                "Cannot generate demonstrations."
            )

        optimal_actions = self.env.find_shortest_path()
        if optimal_actions is None:
            raise RuntimeError("BFS returned None path despite is_solvable() check.")

        demonstrations: List[Demonstration] = []

        for demo_idx in range(n_demos):
            # Use per-demo seed for reproducibility
            demo_seed = self.seed + demo_idx * 1000
            state = self.env.reset(seed=demo_seed)
            transitions: List[Transition] = []
            total_reward = 0.0
            done = False
            success = False

            # Execute BFS path (with optional noise)
            for step_idx, expert_action in enumerate(optimal_actions):
                if done:
                    break

                # Inject noise: replace expert action with random action
                if noise_prob > 0.0 and self._rng.random() < noise_prob:
                    action = self._rng.randint(0, self.env.action_space_size - 1)
                else:
                    action = expert_action

                next_state, reward, done, info = self.env.step(action)

                transitions.append(Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                ))
                total_reward += reward
                state = next_state

                if info.get("event") == "goal":
                    success = True
                    break

            demonstrations.append(Demonstration(
                transitions=transitions,
                total_reward=total_reward,
                n_steps=len(transitions),
                success=success,
                quality=quality,
            ))

        return demonstrations

    def generate_optimal(self, n_demos: int) -> List[Demonstration]:
        """Generate optimal (noise-free) BFS demonstrations."""
        return self.generate(n_demos=n_demos, quality="optimal", noise_prob=0.0)

    def generate_suboptimal(self, n_demos: int, noise_prob: float = 0.2) -> List[Demonstration]:
        """Generate suboptimal demonstrations with noise injection."""
        return self.generate(n_demos=n_demos, quality="suboptimal", noise_prob=noise_prob)

    @staticmethod
    def flatten_transitions(demonstrations: List[Demonstration]) -> List[Transition]:
        """
        Flatten a list of demonstrations into a single transition list.

        Args:
            demonstrations: List of Demonstration objects.

        Returns:
            Flat list of all Transition objects across all demonstrations.
        """
        return [t for demo in demonstrations for t in demo.transitions]

    @staticmethod
    def get_statistics(demonstrations: List[Demonstration]) -> dict:
        """
        Compute summary statistics over a set of demonstrations.

        Returns:
            Dict with keys: n_demos, success_rate, mean_steps,
            mean_reward, std_reward, n_transitions.
        """
        if not demonstrations:
            return {}
        n = len(demonstrations)
        successes = sum(1 for d in demonstrations if d.success)
        rewards = [d.total_reward for d in demonstrations]
        steps = [d.n_steps for d in demonstrations]
        n_transitions = sum(len(d.transitions) for d in demonstrations)

        return {
            "n_demos": n,
            "success_rate": successes / n,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "n_transitions": n_transitions,
        }
