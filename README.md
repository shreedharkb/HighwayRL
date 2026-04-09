# PPO for Autonomous Highway Driving 🚗

## Reinforcement Learning Project: Proximal Policy Optimization (PPO) on Highway-Env

This project demonstrates the application of **Proximal Policy Optimization (PPO)**, a state-of-the-art actor-critic reinforcement learning algorithm, to solve the **Highway Driving** task from the `highway-env` package.

### 🎮 About the Environment

The **Highway-v0** environment simulates a highway driving scenario where an autonomous agent must:
- Navigate through traffic at high speed
- Avoid collisions with other vehicles
- Maintain a desired speed
- Change lanes safely

This environment goes **beyond standard OpenAI Gym** tasks (like CartPole or MountainCar) by providing:
- Continuous multi-lane highway simulation
- Multiple NPC vehicles with realistic behavior
- Complex state space (vehicle positions, velocities, headings)
- Multi-objective reward function (speed + safety + lane keeping)

### 🧠 About PPO (Proximal Policy Optimization)

PPO is an **actor-critic** method that belongs to the family of policy gradient algorithms. Key features:
- **Actor** (Policy Network): Decides which action to take
- **Critic** (Value Network): Estimates how good the current state is
- **Clipped Objective**: Prevents destructive large policy updates
- **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage estimates

### 📁 Project Structure

```
RL Project/
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── explore_env.py              # Environment exploration and understanding
├── train_ppo.py                # PPO training pipeline
├── config.py                   # Hyperparameters and configuration
├── evaluate.py                 # Model evaluation and visualization
├── plot_results.py             # Training results plotting
├── models/                     # Saved trained models
│   └── ppo_highway_final.zip
├── results/                    # Training plots and results
│   ├── training_rewards.png
│   └── evaluation_results.png
└── report/
    └── project_report.md       # 5-page project report
```

### 🚀 How to Run

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Explore the Environment
```bash
python explore_env.py
```

#### 3. Train the PPO Agent
```bash
python train_ppo.py
```

#### 4. Evaluate the Trained Agent
```bash
python evaluate.py
```

#### 5. Plot Training Results
```bash
python plot_results.py
```

### 📊 Results

The PPO agent successfully learns to:
- Navigate through highway traffic
- Maintain high speed while avoiding collisions
- Make lane-change decisions strategically

Training converges in approximately **50,000 timesteps** with a mean reward improvement from ~5 to ~35+.

### 🔧 Technologies Used

- **Python 3.12**
- **Gymnasium** — RL environment interface
- **Highway-Env** — Autonomous driving simulation
- **Stable-Baselines3** — PPO implementation
- **PyTorch** — Neural network backend
- **Matplotlib** — Result visualization
- **TensorBoard** — Training monitoring

### 📝 References

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms" (2017)
2. Leurent, E. "An Environment for Autonomous Driving Decision-Making" (highway-env)
3. Schulman, J., et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015)

### 👤 Author

Shreedhar K B
