"""
train_ppo.py - Train PPO Agent on Highway-v0
"""

import os
import gymnasium as gym
import highway_env  # registers highway environments with gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import json

from config import PPO_CONFIG, TRAINING_CONFIG, ENV_CONFIG, POLICY_TYPE


class TrainingMetricsCallback(BaseCallback):
    """Custom callback to log training metrics for plotting."""
    
    def __init__(self, save_path="./results/training_metrics.json", verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_log = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_log.append(self.num_timesteps)
        return True
    
    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "timesteps": self.timesteps_log,
        }
        with open(self.save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nTraining metrics saved to {self.save_path}")


def make_env():
    """Create and wrap the Highway environment."""
    env = gym.make(ENV_CONFIG["env_id"], render_mode=ENV_CONFIG["render_mode"])
    env = Monitor(env)
    return env


def train():
    print("=" * 60)
    print("PPO TRAINING ON HIGHWAY-V0")
    print("=" * 60)
    
    # Create parallel environments for faster training
    print(f"\nCreating {ENV_CONFIG['n_envs']} parallel environments...")
    train_env = DummyVecEnv([make_env for _ in range(ENV_CONFIG["n_envs"])])
    eval_env = DummyVecEnv([make_env])
    
    print(f"  Environment: {ENV_CONFIG['env_id']}")
    print(f"  Observation shape: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space}")
    
    # Initialize PPO with Actor-Critic architecture
    print(f"\nInitializing PPO Agent...")
    print(f"  Policy: {POLICY_TYPE}")
    print(f"  Learning Rate: {PPO_CONFIG['learning_rate']}")
    print(f"  Clip Range: {PPO_CONFIG['clip_range']}")
    print(f"  GAE Lambda: {PPO_CONFIG['gae_lambda']}")
    
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
        seed=42,
    )
    
    print(f"\nNetwork Architecture:")
    print(f"  {model.policy}")
    
    # Setup callbacks
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./results/",
        eval_freq=2500,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    
    metrics_callback = TrainingMetricsCallback(
        save_path="./results/training_metrics.json"
    )
    
    # Train
    print(f"\nStarting Training...")
    print(f"  Total Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  This may take a few minutes...\n")
    
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=[eval_callback, metrics_callback],
        progress_bar=True,
    )
    
    # Save final model
    save_path = TRAINING_CONFIG["save_path"] + "_final"
    model.save(save_path)
    print(f"\nModel saved to: {save_path}.zip")
    
    # Print summary
    if metrics_callback.episode_rewards:
        rewards = metrics_callback.episode_rewards
        last_10 = rewards[-10:] if len(rewards) >= 10 else rewards
        print(f"\nTRAINING SUMMARY:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Final 10 Avg Reward: {np.mean(last_10):.2f}")
        print(f"  Best Reward: {max(rewards):.2f}")
        print(f"  Worst Reward: {min(rewards):.2f}")
    
    print(f"\nTraining Complete!")
    
    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
