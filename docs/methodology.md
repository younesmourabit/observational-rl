# Methodology Documentation

## Project: Observational Reinforcement Learning — Theory, Architectures and Social Learning Dynamics

---

## 1. Scientific Context and Motivation

This project investigates a fundamental question in the intersection of reinforcement learning (RL) and cognitive science:

> **Under what conditions does learning from observed demonstrations improve the learning efficiency, performance, and generalization of an RL agent compared to learning from experience alone?**

The motivation is grounded in developmental psychology and neuroscience: humans and many animals learn efficiently by observing others (social learning, observational learning). This project operationalizes this phenomenon in a computational framework.

---

## 2. Terminological Clarifications

It is essential to distinguish between related but distinct concepts:

### 2.1 Reinforcement Learning (RL)
An agent learns a policy π(a|s) by interacting with an environment and maximizing cumulative reward. No demonstration data is used. The agent must discover good behavior through trial and error.

### 2.2 Imitation Learning (IL) / Learning from Demonstrations (LfD)
The agent learns from a dataset of expert trajectories D = {(s_t, a_t, r_t, s_{t+1})}. The classical form is **Behavioral Cloning (BC)**, which treats imitation as supervised classification: minimize cross-entropy between the agent's policy and the expert's actions.

> ⚠️ **Important**: In this project, the agent has access to the expert's full (s, a, r, s') tuples, including the expert's actions. This is *supervised demonstration learning*, not purely observational learning in the cognitive science sense.

### 2.3 Observational Reinforcement Learning (ORL)
In the strict cognitive science sense, observational RL involves an agent that watches another agent's *behavior* without access to its internal reward signal or action labels — only state sequences are observed. This is an active research area and computationally harder.

### 2.4 Hybrid Approach (this project's contribution)
This project implements a pragmatic hybrid: **BC pre-training followed by RL fine-tuning**. This is well-established in the LfD literature (Hester et al., 2018; Nair et al., 2018) and allows empirical study of when demonstrations accelerate RL.

The term "Observational RL" in the project title refers to the broader research area, not to a claim that the agent learns purely from passive observation.

---

## 3. Algorithm Descriptions

### 3.1 Q-Learning (Baseline RL)

**Algorithm (Watkins & Dayan, 1992):**
```
Initialize Q(s, a) = 0 for all s, a
For each episode:
    s ← reset()
    While not done:
        a ← ε-greedy(Q, s)
        s', r, done ← step(a)
        Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]
        s ← s'
    ε ← max(ε_min, ε × ε_decay)
```

**Hyperparameters:**
- α (learning rate): 0.1
- γ (discount factor): 0.99
- ε_start: 1.0, ε_end: 0.05, ε_decay: 0.995

**Theoretical guarantees:**
Q-learning converges to Q* under mild conditions (GLIE policy, appropriate learning rate schedule). For our small GridWorlds, tabular Q-learning is guaranteed to converge.

**Limitations:**
- Requires exploration of all (s, a) pairs → slow on sparse rewards
- No prior knowledge incorporation
- Does not scale to large state spaces

---

### 3.2 Behavioral Cloning (Baseline IL)

**Algorithm (Pomerleau, 1989):**
```
Collect demonstrations D = {(s_i, a_i)}
Train policy π_θ to minimize: L = -Σ log π_θ(a_i | s_i)
At execution: a = argmax_a π_θ(a|s)
```

**Architecture:** Feedforward MLP (state_size → 64 → 64 → action_size)
**Input encoding:** One-hot vector over state indices

**Known failure modes (distribution shift, Ross & Bagnell 2010):**
- BC trains on states visited by the expert
- At test time, any deviation from the expert's path leads to out-of-distribution states
- Errors compound: the agent enters novel states it was never trained on

**Advantages:**
- Simple supervised learning
- Fast to train
- Immediately good performance near expert states

**Limitations:**
- Cannot improve beyond demonstration quality
- Degrades on states not covered by demonstrations
- No reward signal used → cannot adapt to new reward functions

---

### 3.3 Hybrid ORL: BC → Q-Learning

**Two-phase algorithm:**

**Phase 1 — Behavioral Cloning:**
```
Train π_θ on demonstrations D (as in BC above)
Extract Q-values: Q_BC(s, a) = logit_a(π_θ(s))
Initialize QL Q-table: Q ← Q_BC
```

**Phase 2 — Q-Learning fine-tuning:**
```
Start with Q = Q_BC, ε = ε_low (e.g., 0.3)
Run standard Q-learning training
```

**Rationale for Q-table warm-start:**
The BC policy logits represent the network's preference for each action. Using these as Q-value initializations biases exploration toward regions that the expert visited, while still allowing Q-learning to discover improvements.

**Hypotheses:**
- **H1 (Sample efficiency):** Hybrid ORL reaches a success threshold faster than pure Q-learning.
- **H2 (Suboptimal demonstrations):** With noisy demonstrations, Hybrid ORL still converges but the advantage over pure RL diminishes.
- **H3 (Few demonstrations):** With very limited data, BC alone degrades but Hybrid ORL's RL phase compensates.
- **H4 (Generalization):** BC generalizes poorly to new environments; Hybrid ORL adapts via RL.

---

## 4. Environment Design

### 4.1 GridWorld

A square grid of size G×G with:
- **Start cell S**: agent's initial position
- **Goal cell G**: terminal state (positive reward)
- **Obstacle cells X**: passable but penalized (NOT walls)
- **Free cells**: small negative reward per step (encourages efficiency)
- **Walls**: grid boundaries, penalized but agent stays in place

**State space:** Integer encoding s = row × G + col ∈ [0, G²-1]
**Action space:** {UP, DOWN, LEFT, RIGHT} (4 discrete actions)

### 4.2 Difficulty Levels

| Name   | Grid | Obstacles | Max Steps | Rationale |
|--------|------|-----------|-----------|-----------|
| Easy   | 5×5  | 2         | 50        | Sanity check; QL should solve easily |
| Medium | 8×8  | 8         | 100       | Meaningful challenge |
| Hard   | 10×10| 20        | 150       | Sparse reward; tests advantages of demonstrations |

### 4.3 Train/Test Split

Train environments and test environments use **different obstacle seeds**, ensuring the agent cannot simply memorize a path. This tests generalization to novel layouts.

---

## 5. Expert Demonstration Generation

### 5.1 BFS Expert

Breadth-First Search (BFS) finds the **optimal (shortest) path** from start to goal in the known grid. This is an algorithmic oracle, not a learning agent.

**Demonstration types:**
| Type         | n_demos | noise_prob | Success Rate |
|--------------|---------|------------|--------------|
| Optimal      | 50      | 0.00       | ~100%        |
| Suboptimal   | 50      | 0.20       | ~60-80%      |
| Limited      | 10      | 0.00       | ~100%        |
| Very Limited | 3       | 0.00       | ~100%        |

### 5.2 What the agent observes

In this implementation, the learning agent receives full **(s, a, r, s')** tuples — the expert's state, action, reward received, and resulting next state. This is the standard LfD assumption, not pure observational learning.

---

## 6. Experimental Design

### 6.1 Metrics

| Metric | Definition | Why |
|--------|-----------|-----|
| Success rate | Fraction of episodes where goal is reached | Primary performance metric |
| Cumulative reward | Sum of rewards per episode | Captures both success and efficiency |
| Steps per episode | Mean path length | Measures policy quality beyond success |
| Episodes to threshold | First episode where 50-ep rolling success ≥ 70% | Sample efficiency |
| Training time | Wall-clock seconds | Computational cost |
| Test success rate | Success on held-out environment | Generalization |

### 6.2 Controls

- **Multiple seeds**: 5 seeds [42, 123, 456, 789, 1337] per condition
- **Fixed episode budget**: same for all agents in a condition
- **No data leakage**: test environments have different obstacle configurations
- **Reporting**: mean ± std across seeds, not single-run results

### 6.3 Ablations

1. **Demo quantity**: n_demos ∈ {3, 5, 10, 20, 50, 100}
2. **Demo quality**: optimal, suboptimal (20% noise), limited (10 demos)
3. **Environment difficulty**: easy, medium, hard
4. **Generalization**: train on easy, test on easy_test (different layout)

---

## 7. Methodological Limitations

1. **Tabular Q-learning**: Does not scale beyond small GridWorlds. For larger state spaces, function approximation (DQN, PPO) would be needed.
2. **BC distribution shift**: The known BC failure mode (compounding errors) may favor Hybrid ORL even when demonstrations are only marginally helpful.
3. **BFS expert**: A perfect oracle. Real-world experts are never perfect. Testing with degraded experts (suboptimal demos) partially addresses this.
4. **Single environment type**: GridWorld is deterministic and fully observable. Results may not transfer to stochastic, partially observable, or continuous environments.
5. **Q-table warm-start heuristic**: Using BC logits as Q-values is a design choice, not a theoretically motivated transfer mechanism.
6. **Action access**: We provide action labels to the learning agent. True observational RL (state sequences only) remains as future work.

---

## 8. Future Research Directions

1. **State-only observation**: Remove action labels from demonstrations; force the agent to infer actions via inverse RL or model-based planning.
2. **DQN/PPO integration**: Scale to larger environments using function approximation.
3. **DAgger (Ross et al., 2011)**: Interactive imitation learning that addresses distribution shift.
4. **Multi-agent observational learning**: Multiple observers watching a single expert simultaneously.
5. **Partial observability**: Extend GridWorld with limited agent field-of-view.
6. **Transfer learning**: Pre-train on easy env, transfer to hard env (curriculum learning).
7. **Online demonstration**: Expert provides demonstrations interactively during RL training.

---

## 9. References

1. **Watkins, C.J.C.H. (1989)**. *Learning from Delayed Rewards*. PhD thesis, Cambridge.
2. **Watkins, C.J.C.H. & Dayan, P. (1992)**. Q-learning. *Machine Learning*, 8, 279-292. DOI: 10.1007/BF00992698
3. **Sutton, R.S. & Barto, A.G. (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
4. **Pomerleau, D.A. (1989)**. ALVINN: An Autonomous Land Vehicle in a Neural Network. *NeurIPS 1989*.
5. **Ross, S. & Bagnell, D. (2010)**. Efficient Reductions for Imitation Learning. *AISTATS 2010*.
6. **Ross, S., Gordon, G., & Bagnell, D. (2011)**. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *AISTATS 2011*.
7. **Hester, T. et al. (2018)**. Deep Q-learning from Demonstrations. *AAAI 2018*. arXiv:1704.03732
8. **Nair, A. et al. (2018)**. Overcoming Exploration in RL with Demonstrations. *ICRA 2018*. arXiv:1709.10089
9. **Schaal, S. (1996)**. Learning from Demonstration. *NeurIPS 1996*.
10. **Bandura, A. (1977)**. *Social Learning Theory*. Prentice Hall. (Cognitive science foundation)
11. **Galef, B.G. & Laland, K.N. (2005)**. Social Learning in Animals: Empirical Studies and Theoretical Models. *BioScience*, 55(6), 489-499.
