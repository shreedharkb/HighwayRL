# PPO Highway RL — Complete Technical Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Environment](#2-environment-highway-fast-v0)
3. [Algorithms](#3-algorithms)
4. [Codebase Reference](#4-codebase-reference)
5. [Configuration Guide](#5-configuration-guide)
6. [Running the Project](#6-running-the-project)
7. [Output Files](#7-output-files)
8. [Understanding Results](#8-understanding-results)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Project Overview

This project trains a Reinforcement Learning agent to drive autonomously on a multi-lane highway using **Proximal Policy Optimization (PPO)** with an **Actor-Critic** architecture and **Generalized Advantage Estimation (GAE)**.

A second agent, **A2C (Advantage Actor-Critic)**, is trained identically except without PPO's clipping — this serves as a controlled comparison to demonstrate PPO's advantages.

**What the agent learns:**
- Maintain high speed on the highway
- Avoid collisions with other vehicles
- Change lanes (left/right) to overtake slower vehicles
- Return to a safe lane after overtaking

**Key result:** PPO achieves 21.1 avg reward vs 9.2 for random (2.3× better), with a perfect 0% crash rate.

---

## 2. Environment: Highway-fast-v0

### What it is
`highway-fast-v0` is from the [highway-env](https://github.com/eleurent/highway-env) package — a suite of environments for autonomous driving research. It is **not** part of standard OpenAI Gym.

### State Space
The agent observes a **5 × 5 matrix** at each timestep:
```
Row 0: Ego vehicle (your car)
Row 1: Nearest vehicle 1
Row 2: Nearest vehicle 2
Row 3: Nearest vehicle 3
Row 4: Nearest vehicle 4

Each row has 5 features: [x, y, vx, vy, cos(heading)]
```
Total: **25 continuous values** fed into the neural network.

### Action Space
**5 discrete actions:**

| Index | Action | Effect |
|-------|--------|--------|
| 0 | LANE LEFT | Change to left lane |
| 1 | IDLE | Stay in current lane at current speed |
| 2 | LANE RIGHT | Change to right lane |
| 3 | FASTER | Increase target speed |
| 4 | SLOWER | Decrease target speed |

### Reward Function
```
R = (v / v_max) - collision_penalty × crashed
```
- `v/v_max`: Reward proportional to speed (max ~0.83 per step when at top speed)
- `collision_penalty`: Large negative reward when crashed
- Episode ends on collision OR after 30 steps (timeout)

### Why this is beyond standard Gym
1. **Multi-agent**: 4 other vehicles make independent decisions
2. **Continuous physics**: Vehicles have realistic acceleration/braking
3. **Multi-objective reward**: Must balance speed AND safety
4. **Partial observability**: Can only see 4 nearest vehicles, not the whole road

---

## 3. Algorithms

### 3.1 Actor-Critic Architecture

```
Input State (25 values)
        │
   ┌────▼────┐
   │ FC(256) │  ← Shared Layer 1
   │  + ReLU │
   └────┬────┘
        │
   ┌────▼────┐
   │ FC(256) │  ← Shared Layer 2
   │  + ReLU │
   └──┬─────┬┘
      │     │
  ┌───▼──┐ ┌▼──────┐
  │ACTOR │ │CRITIC │
  │FC(5) │ │ FC(1) │
  │softmax│ │scalar │
  └──────┘ └───────┘
```

- **Actor**: Outputs 5 probabilities (one per action). During training, samples from this distribution. During evaluation, picks the highest probability action.
- **Critic**: Outputs a single scalar — the estimated value V(s) of the current state.
- Total parameters: ~135,000

### 3.2 Generalized Advantage Estimation (GAE)

**Why we need it:**
Raw rewards are noisy. If you get a high reward, was it because of THIS action or an action 10 steps ago? GAE solves this.

**How it works:**
```
δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ)     ← TD residual at step t
Âₜ = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + ...

Parameters:
  γ = 0.99   (discount: cares about rewards 99 steps in future)
  λ = 0.95   (smoothing: high = low variance, low = low bias)
```

**Intuition:** 
- `γλ = 0.9405` means each additional look-ahead step counts 94% as much
- This blends short-term (TD) and long-term (Monte Carlo) estimates

### 3.3 PPO: Proximal Policy Optimization

**Core Problem:** Policy gradient methods can take destructively large steps.

**PPO's Solution — Clipped Objective:**
```
ratio = π_new(a|s) / π_old(a|s)

L = min(
    ratio × Advantage,
    clip(ratio, 1-ε, 1+ε) × Advantage
)
```

With `ε = 0.2`:
- If ratio > 1.2: gradient is zeroed out (don't update further)
- If ratio < 0.8: gradient is zeroed out (don't update further)
- Otherwise: normal gradient update

**Total PPO loss:**
```
L_total = L_CLIP − c₁ × L_value + c₂ × H[π]

Where:
  L_value = (V_predicted − V_target)²    ← Critic loss
  H[π]    = entropy of action distribution  ← Exploration bonus
  c₁ = 0.5  (value loss weight)
  c₂ = 0.01 (entropy weight)
```

**Training loop:**
```
For each iteration:
  1. Collect 256 × 4 = 1024 steps (4 parallel envs)
  2. Compute advantages using GAE
  3. For 10 epochs:
       Shuffle data into batches of 64
       Compute clipped loss
       Update network via Adam
```

### 3.4 A2C: Advantage Actor-Critic

**Identical to PPO except:**
- No clipping (ratio is used directly, can be any value)
- Single epoch per update (no multi-epoch sampling)
- Higher learning rate (1e-3 vs 3e-4)

**Why A2C is less stable:**
Without clipping, if the advantage estimate is inaccurate (due to noisy sampling), the policy can change drastically → training collapses → reward drops → repeat. This creates the "jagged" learning curve we see in results.

---

## 4. Codebase Reference

### `config.py`
Central configuration file. **Change hyperparameters here only.**

```python
PPO_CONFIG = {
    "learning_rate": 3e-4,   # Adam LR — lower = more stable, slower
    "n_steps": 256,          # Steps collected per env before update
    "batch_size": 64,        # Mini-batch size for gradient descent
    "n_epochs": 10,          # Times we reuse each batch of data
    "gamma": 0.99,           # Discount factor for future rewards
    "gae_lambda": 0.95,      # GAE smoothing parameter
    "clip_range": 0.2,       # PPO epsilon — max policy change ±20%
    "ent_coef": 0.01,        # Entropy bonus weight
    "vf_coef": 0.5,          # Value function loss weight
    "max_grad_norm": 0.5,    # Gradient clipping (prevents exploding)
}

TRAINING_CONFIG = {
    "total_timesteps": 50_000,  # Total env steps for training
}

ENV_CONFIG = {
    "env_id": "highway-fast-v0",
    "n_envs": 4,              # Parallel environments
}
```

### `train_ppo.py`
**What it does:**
1. Prints GPU/CPU info
2. Creates 4 parallel `highway-fast-v0` environments
3. Initializes PPO model with config hyperparameters
4. Attaches two callbacks:
   - `EvalCallback`: Evaluates on 5 episodes every 2500 steps, saves best model
   - `TrainingMetricsCallback`: Logs episode rewards/lengths to JSON
5. Trains for 50,000 steps
6. Saves final model to `models/ppo_highway_final.zip`

**Custom Callback Logic:**
```python
def _on_step(self):
    for info in self.locals["infos"]:
        if "episode" in info:          # Episode just ended
            self.episode_rewards.append(info["episode"]["r"])
            self.episode_lengths.append(info["episode"]["l"])
```

### `train_a2c.py`
Identical to `train_ppo.py` but:
- Uses `A2C` class instead of `PPO`
- Uses `A2C_CONFIG` (higher LR, no clip range)
- Saves to `models/a2c_highway_final.zip`
- Logs to `results/training_metrics_a2c.json`

### `compare_agents.py`
**What it does:**
- Loads PPO, A2C, and Random agents
- Runs 50 test episodes for each (deterministic evaluation)
- Prints a high-resolution comparison table
- Generates `plot_evaluation.png` for the report
- Shows stability (std dev), safety (crash rate), and performance (mean reward) side-by-side

### `plot_results.py`
**Two modes:**
```bash
python plot_results.py          # PPO only dashboard
python plot_results.py compare  # PPO vs A2C comparison
```

**PPO Dashboard (4 plots):**
1. Reward per episode (raw + 10-episode moving average)
2. Episode length per episode
3. Reward vs total timesteps (sample efficiency)
4. Histogram of reward distribution

**PPO vs A2C Comparison (4 plots):**
1. Smoothed reward curves for both
2. Episode length comparison
3. Sample efficiency comparison
4. Statistics table (mean, std, max, min, collision rate)

### `record_video.py`
**What it does:**
1. Loads `ppo_highway_final.zip`
2. Creates high-res `highway-fast-v0` environment (1920×600)
3. Enables trajectory visualization (`show_trajectories: True`)
4. Runs multiple episodes until 900 frames collected (~60s at 15fps)
5. Draws compact HUD on each frame:
   - Top bar: title, episode number, step count
   - Bottom bar: velocity gauge, action badge, reward counter
6. Prepends 60-frame cinematic intro card
7. Appends 45-frame outro with stats
8. Encodes to `results/videos/highway_ppo_annotated.mp4`

---

## 5. Configuration Guide

### Increase training quality (at cost of time):
```python
TRAINING_CONFIG = {
    "total_timesteps": 200_000,  # 4× more training
}
```

### Make training faster (less quality):
```python
ENV_CONFIG = {
    "n_envs": 8,  # Double parallel environments
}
PPO_CONFIG = {
    "n_steps": 128,  # Shorter rollouts
}
```

### Make video longer:
```python
# In record_video.py, line ~120:
target_frames = 1800  # ~2 minutes at 15fps
```

---

## 6. Running the Project

### Full pipeline (recommended order):
```bash
# Step 1: Train PPO (5-8 minutes)
python train_ppo.py

# Step 2: Train A2C (4-6 minutes)
python train_a2c.py

# Step 3: Compare agents (1 minute)
python compare_agents.py

# Step 4: Generate plots (30 seconds)
python plot_results.py compare

# Step 5: Record cinematic video (2 minutes)
python record_video.py
```

### Expected console output during training:
```
🚀 TRAINING PPO ON HIGHWAY-V0

🖥️  PyTorch version: 2.6.0+cu124
🎮 CUDA available: True
🔥 GPU detected: NVIDIA GeForce RTX 3050 Laptop GPU
⚡ Using device: cpu (optimal for MLP policy)

Creating 4 parallel environments...
Environment: highway-fast-v0

  Training...
  Total Timesteps: 50,000

Training progress: [=========>] 50000/50000 [05:32<00:00]
Model saved: ./models/ppo_highway_final.zip
Episodes: 180 | Mean: 33.4 | Max: 42.1
```

---

## 7. Output Files

| File | Generated by | Content |
|------|-------------|---------|
| `models/ppo_highway_final.zip` | train_ppo.py | Final PPO model weights |
| `models/a2c_highway_final.zip` | train_a2c.py | Final A2C model weights |
| `results/training_metrics.json` | train_ppo.py | Per-episode rewards, lengths, timesteps |
| `results/training_metrics_a2c.json` | train_a2c.py | Same for A2C |
| `results/training_rewards_high_res.png` | plot_results.py | PPO 4-panel dashboard (300 DPI) |
| `results/ppo_vs_a2c_comparison.png` | plot_results.py | PPO vs A2C comparison (300 DPI) |
| `results/plot_evaluation.png` | compare_agents.py | Final evaluation bar chart (300 DPI) |
| `results/videos/highway_ppo_annotated.mp4` | record_video.py | Cinematic annotated demo video |

---

## 8. Understanding Results

### What good training looks like:
- **Reward trend:** Should increase from ~5-8 (random level) to ~25-40 over 200 episodes
- **Episode length:** Should increase as agent survives longer
- **Std deviation:** Should decrease as policy becomes more consistent

### What bad training looks like:
- Reward oscillates wildly → Try lower learning rate
- Reward flatlines at low values → Try more timesteps or higher entropy
- Reward collapses late in training → Try lower clip_range (0.1)

### Interpreting the comparison plot:
- **PPO curve smoother** = clipping prevents destructive updates
- **A2C curve jagged** = no clipping causes erratic updates
- **Standard deviation** is the most important metric: lower = more reliable agent

### Will the car overtake?
**Yes.** Here's why and how:
1. Reward = speed/max_speed. Higher speed = higher reward.
2. If a slow car is ahead, agent can't speed up without collision risk.
3. Agent discovers: switch lanes → no obstacle → FASTER action → higher reward.
4. After many episodes, this becomes a consistent strategy.
5. You will see this in the video: agent weaves between lanes to maintain speed.

---

## 9. Troubleshooting

### "No module named highway_env"
```bash
pip install highway-env
```

### "FileNotFoundError: models/ppo_highway_final.zip"
You haven't trained yet. Run `python train_ppo.py` first.

### "Training is very slow (4+ hours)"
You're accidentally using GPU for a small MLP. Check `train_ppo.py` line that says `device = "cpu"`. Make sure it's CPU.

### "Video frames look wrong / too small"
The environment is configured for 1920×600 in `record_video.py`. Make sure you have enough RAM.

### "Comparison plot won't generate"
Both `results/training_metrics.json` AND `results/training_metrics_a2c.json` must exist. Run both training scripts first.

### "Images in LaTeX are blank"
The PNG files must exist in `results/` before compiling LaTeX. Run `plot_results.py compare` first, then compile the report.

### "imageio can't encode video"
```bash
pip install imageio[ffmpeg]
```
