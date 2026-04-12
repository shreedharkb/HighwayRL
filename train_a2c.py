"""
train_a2c.py - Train A2C Agent for Comparison with PPO
Demonstrates how different algorithms perform on the same task
"""

import os
import gymnasium as gym
import highway_env
import numpy as np
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import json

from config import A2C_CONFIG, TRAINING_CONFIG, ENV_CONFIG, POLICY_TYPE


class TrainingMetricsCallback(BaseCallback):
    """Logs training metrics to JSON file."""
    
    def __init__(self, save_path="./results/training_metrics_a2c.json", verbose=0):
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
    """Create Highway environment with monitoring."""
    env = gym.make(ENV_CONFIG["env_id"], render_mode=ENV_CONFIG["render_mode"])
    env = Monitor(env)
    return env


def train_a2c():
    print("=" * 60)
    print("A2C TRAINING ON HIGHWAY-V0 (For Comparison with PPO)")
    print("=" * 60)
    
    # Device Selection
    # Note: For small MLP policies (256x256), CPU is faster than GPU
    # because the CPU↔GPU data transfer overhead exceeds the computation time.
    device = "cpu"
    print(f"\n🖥️  PyTorch version: {torch.__version__}")
    print(f"🎮 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"📦 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"⚡ Using device: {device} (optimal for MLP policy)\n")
    
    # Create parallel environments
    print(f"Creating {ENV_CONFIG['n_envs']} parallel environments...")
    train_env = DummyVecEnv([make_env for _ in range(ENV_CONFIG["n_envs"])])
    eval_env = DummyVecEnv([make_env])
    
    print(f"  Environment: {ENV_CONFIG['env_id']}")
    print(f"  Observation shape: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space}")
    
    # Initialize A2C Agent
    print(f"\nInitializing A2C Agent...")
    print(f"  Policy: {POLICY_TYPE}")
    print(f"  Learning Rate: {A2C_CONFIG['learning_rate']}")
    print(f"  GAE Lambda: {A2C_CONFIG['gae_lambda']}")
    print(f"  Note: A2C uses NO clipping (unlike PPO)")
    
    model = A2C(
        policy=POLICY_TYPE,
        env=train_env,
        learning_rate=A2C_CONFIG["learning_rate"],
        n_steps=A2C_CONFIG["n_steps"],
        gamma=A2C_CONFIG["gamma"],
        gae_lambda=A2C_CONFIG["gae_lambda"],
        ent_coef=A2C_CONFIG["ent_coef"],
        vf_coef=A2C_CONFIG["vf_coef"],
        max_grad_norm=A2C_CONFIG["max_grad_norm"],
        tensorboard_log=TRAINING_CONFIG["tensorboard_log"],
        verbose=A2C_CONFIG["verbose"],
        device=device,
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
        save_path="./results/training_metrics_a2c.json"
    )
    
    # Train
    print(f"\nStarting A2C Training...")
    print(f"  Total Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  (Same as PPO for fair comparison)")
    print(f"  This will take a few minutes...\n")
    
    model.learn(
        total_timesteps=TRAINING_CONFIG["total_timesteps"],
        callback=[eval_callback, metrics_callback],
        progress_bar=True,
    )
    
    # Save final model
    save_path = "./models/a2c_highway_final"
    model.save(save_path)
    print(f"\nModel saved to: {save_path}.zip")
    
    # Print summary
    if metrics_callback.episode_rewards:
        rewards = metrics_callback.episode_rewards
        last_10 = rewards[-10:] if len(rewards) >= 10 else rewards
        print(f"\nA2C TRAINING SUMMARY:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Final 10 Avg Reward: {np.mean(last_10):.2f}")
        print(f"  Best Reward: {max(rewards):.2f}")
        print(f"  Worst Reward: {min(rewards):.2f}")
        print(f"  Std Dev: {np.std(rewards):.2f}")
    
    print(f"\nA2C Training Complete!")
    
    train_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train_a2c()
