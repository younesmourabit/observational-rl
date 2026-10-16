"""
Observational Reinforcement Learning Research Package.

This package provides an experimental framework for comparing:
- Classical Q-Learning (RL baseline)
- Behavioral Cloning (imitation learning baseline)
- Hybrid Observational RL (BC pre-training + RL fine-tuning)

Research question:
    Under what conditions does learning from observed demonstrations improve
    learning efficiency, performance, and generalization of an RL agent
    compared to learning from experience alone?

Modules:
    environments  - Configurable GridWorld and base environment interface
    agents        - Q-Learning, Behavioral Cloning, Hybrid ORL agents
    demonstrations- Expert demonstration generation (BFS, suboptimal)
    training      - Training loops and experiment runners
    evaluation    - Metrics, evaluation protocols
    visualization - Plotting and figure generation
"""

__version__ = "0.1.0"
__author__ = "Research Author"

from observational_rl.environments.gridworld import GridWorldEnv
from observational_rl.agents.q_learning import QLearningAgent
from observational_rl.agents.behavioral_cloning import BehavioralCloningAgent
from observational_rl.agents.hybrid_orl import HybridORLAgent
from observational_rl.demonstrations.expert import ExpertDemonstrationGenerator

__all__ = [
    "GridWorldEnv",
    "QLearningAgent",
    "BehavioralCloningAgent",
    "HybridORLAgent",
    "ExpertDemonstrationGenerator",
]
