"""
train_ppo.py - Train Custom PPO Agent on Highway-v0
"""

import os
import gymnasium as gym
import highway_env
import numpy as np
import json
import torch

from models.custom_ppo import CustomPPO
from config import PPO_CONFIG, TRAINING_CONFIG, ENV_CONFIG

class TrainingMetricsCallback:
    """Logs training metrics to JSON file."""
    
    def __init__(self, save_path="./results/training_metrics.json"):
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_log = []
        
    def on_episode_end(self, reward, length, timestep):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.timesteps_log.append(timestep)
        
    def save_metrics(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "timesteps": self.timesteps_log,
        }
        with open(self.save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nTraining metrics saved to {self.save_path}")

def make_env(env_id, render_mode):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        from config import HIGHWAY_ENV_CONFIG
        env.unwrapped.configure(HIGHWAY_ENV_CONFIG)
        env.reset()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init

def train():
    print("\n🚀 TRAINING CUSTOM PPO ON HIGHWAY-V0\n")
    
    # Use Gymnasium's built-in vectorization
    env_fns = [make_env(ENV_CONFIG["env_id"], ENV_CONFIG["render_mode"]) for _ in range(ENV_CONFIG["n_envs"])]
    envs = gym.vector.SyncVectorEnv(env_fns)
    
    print(f"Creating {ENV_CONFIG['n_envs']} parallel environments...")
    print(f"Environment: {ENV_CONFIG['env_id']}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = CustomPPO(
        envs=envs,
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
        device=device,
        verbose=PPO_CONFIG["verbose"],
    )
    
    print(f"\nNetwork Architecture:")
    print(model.network)
    
    # Setup callbacks
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    metrics_callback = TrainingMetricsCallback(
        save_path="./results/training_metrics.json"
    )
    
    # Train
    print(f"\nStarting Training...")
    print(f"  Total Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  This may take a few minutes...\n")
    
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        metrics_callback=metrics_callback,
    )
    
    save_path = TRAINING_CONFIG["save_path"] + "_final"
    model.save(save_path)
    print(f"Model saved: {save_path}.pt")
    
    metrics_callback.save_metrics()
    
    if metrics_callback.episode_rewards:
        rewards = metrics_callback.episode_rewards
        print(f"Episodes: {len(rewards)} | Mean: {np.mean(rewards[-10:]):.2f} | Max: {max(rewards):.2f}")
    
    envs.close()
    return model

if __name__ == "__main__":
    train()
