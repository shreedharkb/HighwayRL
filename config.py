"""Hyperparameters for PPO and A2C training on Highway-fast-v0"""

# PPO HYPERPARAMETERS

PPO_CONFIG = {
    "learning_rate": 3e-4,       # Adam optimizer learning rate
    "n_steps": 256,              # Steps per rollout before update
    "batch_size": 64,            # Mini-batch size for gradient descent
    "n_epochs": 10,              # Epochs per PPO update
    "gamma": 0.99,               # Discount factor for future rewards
    "gae_lambda": 0.95,          # GAE lambda for advantage estimation
    "clip_range": 0.2,           # PPO clipping parameter (epsilon)
    "ent_coef": 0.05,            # Entropy bonus — higher = more exploration (was 0.01, too low)
    "vf_coef": 0.5,              # Value function loss coefficient
    "max_grad_norm": 0.5,        # Gradient clipping threshold
    "verbose": 1,                # Print training progress
}

# ============================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================
# The policy network (actor) and value network (critic) share
# a common architecture: MlpPolicy = Multi-Layer Perceptron
# Default architecture: 2 hidden layers of 64 neurons each
# Input → [64] → [64] → Output

POLICY_TYPE = "MlpPolicy"

NETWORK_CONFIG = {
    "net_arch": [256, 256],      # Two hidden layers, 256 neurons each
    "activation_fn": "Tanh",     # Activation function
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TRAINING_CONFIG = {
    "total_timesteps": 50_000,   # Reduced to 50k as per user request
    "log_interval": 10,          # Log every N updates
    "save_path": "./models/ppo_highway",  # Where to save the model
    "tensorboard_log": "./tensorboard_logs/",  # TensorBoard logs
}

# ============================================================
# A2C HYPERPARAMETERS (for comparison with PPO)
# ============================================================
A2C_CONFIG = {
    "learning_rate": 7e-4,       # A2C typically uses higher learning rate
    "n_steps": 5,                # Shorter rollout for stable synchronous updates
    "gamma": 0.99,               # Discount factor
    "gae_lambda": 0.95,          # GAE lambda (for fair comparison)
    "ent_coef": 0.01,            # Entropy bonus
    "vf_coef": 0.5,              # Value function weight
    "max_grad_norm": 0.5,        # Gradient clipping
    "verbose": 1,
}

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

ENV_CONFIG = {
    "env_id": "highway-fast-v0", # 12x faster physics simulation
    "render_mode": None,         # No rendering for hyper-fast training
    "n_envs": 4,                 # Number of parallel environments
}

# ============================================================
# HIGHWAY ENVIRONMENT CUSTOM CONFIG
# Tuned to reward speed and lane changes so the agent learns
# to overtake instead of just slowing down.
# ============================================================
HIGHWAY_ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-30, 30],
            "vy": [-30, 30],
        },
        "absolute": False,
        "order": "sorted",
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,            # More lanes = more overtaking opportunities
    "vehicles_count": 30,        # Dense traffic = more overtaking needed
    "duration": 60,              # Longer episodes to give agent time to overtake
    "initial_spacing": 2,
    "collision_reward": -2.0,    # Penalty for crashing
    "reward_speed_range": [20, 35],  # Reward higher speeds (pushes agent to accelerate)
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
}

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

EVAL_CONFIG = {
    "n_eval_episodes": 50,       # Number of episodes for evaluation
    "deterministic": True,       # Use deterministic policy (no exploration)
}
