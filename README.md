# Observational Reinforcement Learning: A Python Project

## Overview

This project is a personal initiative developed to explore Reinforcement Learning (RL) and learning from expert demonstrations using Python.

It implements and compares three approaches in a GridWorld navigation environment:

- **Q-Learning:** learning through interaction with an environment and reward feedback.
- **Behavioral Cloning (BC):** learning from expert demonstrations through supervised learning.
- **Hybrid BC–Q-Learning:** an experimental approach combining imitation learning and reinforcement learning.

The project includes a custom GridWorld environment, expert demonstration generation, agent training, performance evaluation, result visualization, and an interactive Streamlit interface.

## Objectives

- Apply machine learning concepts to a practical programming project.
- Implement reinforcement learning and imitation learning algorithms.
- Compare different learning strategies in a controlled environment.
- Analyse performance using success rates, rewards, and learning curves.
- Develop reproducible experiments and an interactive application.

## Technologies

Python, PyTorch, NumPy, Matplotlib, Streamlit, and pytest.

## Installation

```bash
git clone https://github.com/younesmourabit/observational-rl.git
cd observational-rl
python -m pip install -e .
```

## Run the example

```bash
python examples/quickstart.py
```

## Run experiments

```bash
python scripts/run_experiments.py --mode quick
```

## Launch the interactive application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Project Status

This is a personal technical project developed to explore reinforcement learning, imitation learning, and experimental evaluation. It is an initial prototype intended for learning, experimentation, and further improvement.

The reported results apply to the tested configurations and should not be interpreted as evidence of general superiority over other methods.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
