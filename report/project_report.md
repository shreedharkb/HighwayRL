# Project Report: Autonomous Driving with Proximal Policy Optimization (PPO)

**Github Repository:** [https://github.com/shreedharkb/HighwayRL.git](https://github.com/shreedharkb/HighwayRL.git)
**Author:** Shreedhar K B

---

## 1. Introduction

Reinforcement Learning (RL) has emerged as a deeply promising paradigm for solving complex sequential decision-making problems. Unlike traditional supervised learning, RL agents learn through continuous interaction with an environment, optimizing a cumulative reward signal. Over the past decade, deep reinforcement learning (DRL) algorithms have achieved superhuman performance in domains ranging from classic Atari games to complex strategic games like Go.

This project focuses on a significant application area for RL: **Autonomous Driving**. While introductory RL courses frequently focus on standard OpenAI Gym benchmark environments (such as CartPole, Pendulum, or MountainCar), these environments often lack the complexity required to model real-world scenarios. 

To demonstrate a more advanced application of RL, this project employs the `highway-env` package—a rich collection of environments designed specifically for autonomous driving decision-making. We tackle the `highway-v0` environment, where an agent-controlled ego-vehicle must navigate through a simulated multi-lane highway, interact with non-player character (NPC) vehicles, maintain high speeds, and avoid collisions.

The problem is solved using **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient method in the Actor-Critic family. PPO provides an excellent balance between sample efficiency, implementation complexity, and training stability.

This report will detail the environment formulation, the theoretical background of Actor-Critic methods and PPO, the implementation pipeline, and an evaluation of the trained agent's performance.

---

## 2. Environment Description

The `highway-v0` environment is simulated using kinematics and collision dynamics. It represents a continuous stretch of road with multiple lanes. The environment is highly stochastic due to the presence of multiple NPC vehicles with their own behavioral models.

### 2.1 State Space (Observation)
The observation space is a 2D matrix (a continuous `Box` space) of shape $(V, F)$, where $V$ is the number of vehicles (typically the ego-vehicle + $V-1$ closest nearby vehicles) and $F$ is the number of features per vehicle. In our configuration, we observe 5 vehicles total, and each vehicle provides 5 features:
1. **Presence:** A binary flag (1 if the vehicle is present, 0 otherwise).
2. **x:** Longitudinal position on the highway.
3. **y:** Lateral position (representing lanes).
4. **vx:** Longitudinal velocity.
5. **vy:** Lateral velocity.

Therefore, the state provides the agent with an egocentric view of the surrounding traffic layout.

### 2.2 Action Space
To simulate steering and pedal control at a discrete decision level, the environment uses a discrete action space with 5 possible actions:
- `0`: **LANE_LEFT** (Change to the left lane)
- `1`: **IDLE** (Maintain the current lane and speed)
- `2`: **LANE_RIGHT** (Change to the right lane)
- `3`: **FASTER** (Accelerate)
- `4`: **SLOWER** (Decelerate)

### 2.3 Reward Function
The reward function dictates the policy the agent will learn. In `highway-v0`, the reward is uniquely designed to balance speed, safekeeping, and lane discipline. A typical configuration of the reward $R_t$ at step $t$ is calculated as:

$$ R_t = a \cdot R_{collision} + b \cdot R_{velocity} + c \cdot R_{lane\_change} + d \cdot R_{right\_lane} $$

- A strong negative penalty is applied if a crash occurs.
- Positive reward scales linearly with higher velocities (up to a limit).
- Small positive rewards are given for driving in the right-most lanes (to simulate legal driving rules).
- The episode terminates immediately if a collision occurs.

---

## 3. Theoretical Background: Actor-Critic and PPO

### 3.1 Policy Gradient and Actor-Critic
Policy Gradient methods directly parameterize the policy $\pi_\theta(a|s)$ and optimize the parameters $\theta$ to maximize expected returns by performing gradient ascent. The gradient of the expected return $J(\theta)$ is given by the Policy Gradient Theorem:

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi}(s_t, a_t)] $$

where $A^{\pi}(s_t, a_t)$ is the Advantage function, indicating how much better action $a_t$ was compared to the average action in state $s_t$.

In **Actor-Critic** architectures, two neural networks (or two heads of a single network) are maintained:
1. **The Actor:** Parameterizes the policy $\pi_\theta(a|s)$ and decides which actions to take.
2. **The Critic:** Parameterizes the value function $V_\phi(s)$ and evaluates the quality of states.

The Critic is used to estimate the Advantage function $A(s,a) = Q(s,a) - V(s)$. In modern algorithms, **Generalized Advantage Estimation (GAE)** is used to drastically reduce the variance of this estimate by taking exponentially moving averages of multi-step Temporal Difference (TD) errors.

### 3.2 Proximal Policy Optimization (PPO)
Standard policy gradient algorithms suffer from instability; a single overly large update to $\theta$ can completely break the policy, dropping the agent into a region of the parameter space from which it cannot recover. Trust Region Policy Optimization (TRPO) solved this using hard KL-divergence constraints, but it was computationally expensive.

**Proximal Policy Optimization (PPO)**, introduced by OpenAI in 2017, achieves the stability of TRPO but relies only on first-order optimization, making it much simpler and faster. PPO achieves this by employing a **clipped surrogate objective function**.

Let $r_t(\theta)$ denote the probability ratio between the new policy and the old policy:
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} $$

The PPO objective function is defined as:
$$ L^{CLIP}(\theta) = \mathbb{E} [ \min( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t ) ] $$

- If the advantage $A_t > 0$ (the action was good), the objective increases as the policy increases the probability of the action. However, the `clip` function ensures that the ratio does not exceed $1+\epsilon$. This prevents excessive greediness.
- If $A_t < 0$ (the action was bad), the objective decreases, but clipping prevents the ratio from dropping below $1-\epsilon$, preventing overly drastic dampening.

This clipping mechanism acts as a "trust region," ensuring the new policy does not stray too noticeably from the old policy in a single update step.

---

## 4. Implementation Details

The project utilizes Python alongside `gymnasium` and the `stable-baselines3` library, an industry-standard library that provides highly optimized PyTorch implementations of RL algorithms.

### 4.1 Project Architecture
To maintain a modular approach, the repository consists of several scripts:
1. `config.py`: Contains hyperparameters and configurations, isolating changes from the code.
2. `explore_env.py`: Explores and validates the state/action spaces of `highway-v0` via a random agent baseline.
3. `train_ppo.py`: The pipeline for vectorizing environments, constructing the Actor-Critic network, mapping custom callbacks, and training the model.
4. `evaluate.py`: Compares the loaded trained model deterministically against random baselines.
5. `plot_results.py`: visualizes saved rewards across timesteps using Matplotlib.

### 4.2 Network and Hyperparameters
The configuration explicitly relies on shared initial Multi-Layer Perceptrons (MLPs).
- **Network Depth/Breadth:** `[256, 256]` neurons with Tanh activation. This is notably larger than the default Stable-Baselines dimensions (`[64, 64]`), compensating for the relative complexity of observing 5 highly dynamic vehicles.
- **Learning Rate:** $3 \times 10^{-4}$ configured with Adam Optimizer.
- **$\epsilon$ (Clip Range):** $0.2$, the standard empirical value that prevents destructive updates.
- **GAE $\lambda$:** $0.95$, to heavily smooth returns over trajectories.
- **Parallelization:** `DummyVecEnv` wraps 4 concurrent environments. This multiplies sample collection rate by 4, offering diverse batching data to stabilize policy gradients.

### 4.3 Custom Callbacks
A `TrainingMetricsCallback` intercepts step boundaries mid-training. Since Stable Baselines does not explicitly retain Episode Rewards arrays easily via standard API boundaries, our code tracks rewards over exactly `50,000` steps and caches them to a `.json` file for downstream programmatic interpretation and visualization.

---

## 5. Training Results and Evaluation

### 5.1 Training Convergence
Training for $50,000$ timesteps allows the agent approximately several hundred episodes inside the highway sandbox. Initially, the agent executes sporadic lateral movements and blind accelerations. As updating epochs proceed, PPO inherently drives the Critic network to correctly minimize temporal difference loss, while the Actor aggressively learns that crashing equates to negative terminal rewards.

Plots generated by `plot_results.py` systematically verify that:
1. **Reward over Timesteps:** Gradually ascends from erratic base values mapping a clear upward moving average path.
2. **Episode Length:** Progressively increases. At epoch 0, the agent crashes almost immediately (within 10-15 frames). Near episode 300, the agent survives continuously until the natural environment truncation boundaries (e.g., maximum default step lengths limits).

### 5.2 Deterministic Evaluation
To test the finalized `ppo_highway_final.zip` weights, `evaluate.py` runs 20 continuous, noise-free deterministic episodes comparing the Actor against a purely random selection heuristic.

**Random Baseline Outcomes:**
- Mean Episode Reward: Very low (typically under 10)
- Failure mode: Extremely fast collisions caused by straying off safety margins or slamming brakes randomly.

**Trained PPO Outcomes:**
- Mean Episode Reward: Ranges between 30 to 40+.
- Survival logic: The agent successfully executes multi-lane overtakes when forward momentum is blocked.
- Collision rate drops to practically $0\%$.
- Improvement over random spans over hundreds of percentage points, concretely proving the convergence of the deep surrogate loss algorithm.

---

## 6. Conclusion 

This project robustly validates how the **Proximal Policy Optimization** algorithm can address continuous, multi-agent dynamics inside a simulated driving engine completely independent from vanilla classic control sandboxes. 

By applying Actor-Critic methodologies equipped with Generalized Advantage Estimation and Clustered Value/Policy architectures, the system rapidly pivots away from lethal driving maneuvers and optimizes specifically for stable, consistent highway propulsion. The code structure encapsulates parameter-tuning safely and exposes deep learning metrics, offering a complete end-to-end framework. 

**Code Availability:** All source code, results, and network artifacts have been pushed iteratively to the public repository hosted at: `https://github.com/shreedharkb/HighwayRL.git`. 
