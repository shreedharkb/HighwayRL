"""
config.py - Hyperparameters and Configuration
===============================================
This file contains all the hyperparameters used for PPO training.
Keeping them in a separate file makes it easy to tune and experiment.

PPO HYPERPARAMETERS EXPLAINED:
------------------------------
1. learning_rate: How fast the neural network updates its weights.
   - Too high → unstable training, policy oscillates
   - Too low  → very slow learning
   - 3e-4 is a good default for PPO

2. n_steps: Number of steps to collect before each policy update.
   - More steps = more data per update = more stable but slower
   - 256 is a good balance for discrete action spaces

3. batch_size: Number of samples used in each gradient descent step.
   - Must divide n_steps evenly
   - 64 is a common choice

4. n_epochs: Number of times to iterate over the collected data.
   - More epochs = more learning from same data
   - Too many epochs → overfitting to old data
   - 10 is the standard for PPO

5. gamma (discount factor): How much future rewards matter.
   - 0.99 means future rewards are almost as important as immediate
   - Lower gamma = more short-sighted agent

6. gae_lambda: Lambda for Generalized Advantage Estimation (GAE).
   - Controls bias-variance tradeoff in advantage estimation
   - 0.95 is the standard value
   - Higher = lower bias, higher variance

7. clip_range: The epsilon for PPO's clipped objective.
   - Limits how much the policy can change in one update
   - 0.2 is the standard value from the PPO paper
   - This is THE key innovation of PPO!

8. ent_coef: Entropy coefficient — encourages exploration.
   - Higher = more random exploration
   - 0.01 is a small but helpful amount

9. vf_coef: Value function coefficient in the loss.
   - Balances policy loss vs value loss
   - 0.5 is standard
"""

# ============================================================
# PPO HYPERPARAMETERS
# ============================================================

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
    "total_timesteps": 3_000,   # Total training steps
    "log_interval": 10,          # Log every N updates
    "save_path": "./models/ppo_highway",  # Where to save the model
    "tensorboard_log": "./tensorboard_logs/",  # TensorBoard logs
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
