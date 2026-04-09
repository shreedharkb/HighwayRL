"""
train_ppo.py - Train PPO Agent on Highway-v0
==============================================
This is the main training script that:
    1. Sets up the Highway environment
    2. Configures the PPO algorithm
    3. Trains the agent
    4. Saves the trained model
    5. Logs training metrics

HOW PPO WORKS (Step-by-Step):
=============================

Step 1: COLLECT EXPERIENCE
    - The agent interacts with the environment for 'n_steps' timesteps
    - At each step, the ACTOR network picks an action
    - We store: (state, action, reward, next_state, done)

Step 2: COMPUTE ADVANTAGES using GAE
    - The CRITIC network estimates V(s) for each state
    - We compute advantages: A(s,a) = how much better was this action
      compared to what we expected?
    - GAE smooths these estimates: A_t = sum(gamma*lambda)^l * delta_{t+l}
    - where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)

Step 3: UPDATE POLICY (Actor) with CLIPPING
    - We compute the probability ratio: r(θ) = π_new(a|s) / π_old(a|s)
    - PPO's clipped objective:
        L = min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)
    - This prevents the policy from changing too drastically
    - If advantage is positive (good action), ratio is capped at 1+ε
    - If advantage is negative (bad action), ratio is capped at 1-ε

Step 4: UPDATE VALUE FUNCTION (Critic)
    - Minimize: L_VF = (V(s) - V_target)^2
    - The critic learns to better predict future rewards

Step 5: REPEAT for 'total_timesteps'
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import json

# Import our configuration
from config import PPO_CONFIG, TRAINING_CONFIG, ENV_CONFIG, POLICY_TYPE


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to log training metrics for plotting later.
    
    WHY WE NEED THIS:
    - Stable-Baselines3 logs internally, but we want to save
      metrics to a JSON file for our own plotting
    - This records: episode rewards, lengths, and timesteps
    """
    
    def __init__(self, save_path="./results/training_metrics.json", verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_log = []
        
    def _on_step(self) -> bool:
        """Called at every step. We check if any episode finished."""
        # Check if any episode in the vectorized env has completed
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_log.append(self.num_timesteps)
        return True  # Return True to continue training
    
    def _on_training_end(self) -> None:
        """Save all metrics when training ends."""
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "timesteps": self.timesteps_log,
        }
        with open(self.save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n📊 Training metrics saved to {self.save_path}")


def make_env():
    """
    Create and wrap the Highway environment.
    
    WHY Monitor WRAPPER?
    - Monitor tracks episode rewards and lengths automatically
    - This data is used by our callback and evaluation
    """
    env = gym.make(
        ENV_CONFIG["env_id"],
        render_mode=ENV_CONFIG["render_mode"]
    )
    env = Monitor(env)  # Wrap with Monitor for tracking
    return env


def train():
    """
    Main training function.
    
    This function:
    1. Creates parallel environments (faster training)
    2. Initializes PPO with our hyperparameters
    3. Sets up evaluation callback (test during training)
    4. Trains the agent
    5. Saves the final model
    """
    
    print("=" * 60)
    print("🚗 PPO TRAINING ON HIGHWAY-V0")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # STEP 1: Create Vectorized Environments
    # ----------------------------------------------------------
    # DummyVecEnv runs multiple environments in the same process
    # This gives us more diverse experience per update
    print(f"\n📦 Creating {ENV_CONFIG['n_envs']} parallel environments...")
    
    train_env = DummyVecEnv([make_env for _ in range(ENV_CONFIG["n_envs"])])
    eval_env = DummyVecEnv([make_env])  # Separate env for evaluation
    
    print(f"   Environment: {ENV_CONFIG['env_id']}")
    print(f"   Observation shape: {train_env.observation_space.shape}")
    print(f"   Action space: {train_env.action_space}")
    
    # ----------------------------------------------------------
    # STEP 2: Initialize PPO Agent
    # ----------------------------------------------------------
    # PPO combines:
    #   - Actor (policy network): outputs action probabilities
    #   - Critic (value network): outputs state value estimate
    # Both networks share the MlpPolicy architecture
    
    print(f"\n🧠 Initializing PPO Agent...")
    print(f"   Policy: {POLICY_TYPE} (Multi-Layer Perceptron)")
    print(f"   Learning Rate: {PPO_CONFIG['learning_rate']}")
    print(f"   Clip Range: {PPO_CONFIG['clip_range']}")
    print(f"   GAE Lambda: {PPO_CONFIG['gae_lambda']}")
    
    model = PPO(
        policy=POLICY_TYPE,
        env=train_env,
        learning_rate=PPO_CONFIG["learning_rate"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        ent_coef=PPO_CONFIG["ent_coef"],
        vf_coef=PPO_CONFIG["vf_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        tensorboard_log=TRAINING_CONFIG["tensorboard_log"],
        verbose=PPO_CONFIG["verbose"],
        seed=42,  # For reproducibility
    )
    
    # Print network architecture
    print(f"\n🏗️  Network Architecture:")
    print(f"   {model.policy}")
    
    # ----------------------------------------------------------
    # STEP 3: Setup Callbacks
    # ----------------------------------------------------------
    # Callbacks are functions that run during training
    # EvalCallback: periodically tests the agent's performance
    # TrainingMetricsCallback: logs metrics for our plotting
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./results/",
        eval_freq=2500,           # Evaluate every 2500 steps
        n_eval_episodes=5,        # Use 5 episodes per evaluation
        deterministic=True,       # Use deterministic policy for eval
        render=False,
    )
    
    metrics_callback = TrainingMetricsCallback(
        save_path="./results/training_metrics.json"
    )
    
    # ----------------------------------------------------------
    # STEP 4: TRAIN!
    # ----------------------------------------------------------
    print(f"\n🏋️ Starting Training...")
    print(f"   Total Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"   This may take a few minutes...\n")
    
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=[eval_callback, metrics_callback],
        progress_bar=True,
    )
    
    # ----------------------------------------------------------
    # STEP 5: Save the Final Model
    # ----------------------------------------------------------
    save_path = TRAINING_CONFIG["save_path"] + "_final"
    model.save(save_path)
    print(f"\n✅ Model saved to: {save_path}.zip")
    
    # ----------------------------------------------------------
    # STEP 6: Print Training Summary
    # ----------------------------------------------------------
    if metrics_callback.episode_rewards:
        rewards = metrics_callback.episode_rewards
        last_10 = rewards[-10:] if len(rewards) >= 10 else rewards
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"   Total Episodes: {len(rewards)}")
        print(f"   Final 10 Episodes Avg Reward: {np.mean(last_10):.2f}")
        print(f"   Best Episode Reward: {max(rewards):.2f}")
        print(f"   Worst Episode Reward: {min(rewards):.2f}")
    
    print(f"\n🎉 Training Complete!")
    print(f"   Run 'python evaluate.py' to test the trained agent")
    print(f"   Run 'python plot_results.py' to visualize training curves")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    train()
