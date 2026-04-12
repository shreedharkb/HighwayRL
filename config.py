"""Hyperparameters for PPO and A2C training on Highway-v0"""

# PPO HYPERPARAMETERS

PPO_CONFIG = {
    "learning_rate": 3e-4,       # Adam optimizer learning rate
    "n_steps": 256,              # Steps per rollout before update
    "batch_size": 64,            # Mini-batch size for gradient descent
    "n_epochs": 10,              # Epochs per PPO update
    "gamma": 0.99,               # Discount factor for future rewards
    "gae_lambda": 0.95,          # GAE lambda for advantage estimation
    "clip_range": 0.2,           # PPO clipping parameter (epsilon)
    "ent_coef": 0.01,            # Entropy bonus coefficient
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
    "total_timesteps": 50_000,   # Total training steps (matches report)
    "log_interval": 10,          # Log every N updates
    "save_path": "./models/ppo_highway",  # Where to save the model
    "tensorboard_log": "./tensorboard_logs/",  # TensorBoard logs
}

# ============================================================
# A2C HYPERPARAMETERS (for comparison with PPO)
# ============================================================
A2C_CONFIG = {
    "learning_rate": 1e-3,       # A2C typically uses higher learning rate
    "n_steps": 256,              # Steps per rollout (same as PPO for fair comparison)
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
    "env_id": "highway-v0",      # Highway driving environment
    "render_mode": None,         # No rendering for hyper-fast training
    "n_envs": 4,                 # Number of parallel environments
}

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

EVAL_CONFIG = {
    "n_eval_episodes": 50,       # Number of episodes for evaluation
    "deterministic": True,       # Use deterministic policy (no exploration)
}
