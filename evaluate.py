"""
evaluate.py - Evaluate the Trained PPO Agent
"""

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO
from config import ENV_CONFIG, EVAL_CONFIG


def evaluate_trained_agent():
    print("=" * 60)
    print("EVALUATING TRAINED PPO AGENT")
    print("=" * 60)
    
    # Load trained model
    model_path = "./models/ppo_highway_final"
    try:
        model = PPO.load(model_path)
        print(f"\nModel loaded from: {model_path}")
    except FileNotFoundError:
        try:
            model = PPO.load("./models/best_model")
            print(f"\nBest model loaded from: ./models/best_model")
        except FileNotFoundError:
            print("\nNo trained model found! Run train_ppo.py first.")
            return
    
    env = gym.make(ENV_CONFIG["env_id"], render_mode="rgb_array")
    n_episodes = EVAL_CONFIG["n_eval_episodes"]
    deterministic = EVAL_CONFIG["deterministic"]
    
    # Evaluate trained agent
    print(f"\nRunning TRAINED agent for {n_episodes} episodes...")
    trained_rewards = []
    trained_lengths = []
    collision_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        trained_rewards.append(total_reward)
        trained_lengths.append(steps)
        if done and not truncated:
            collision_count += 1
        
        status = "CRASH" if (done and not truncated) else "SURVIVED"
        print(f"  Episode {episode + 1:2d}: Reward = {total_reward:7.2f}, "
              f"Steps = {steps:3d}, {status}")
    
    # Evaluate random agent for comparison
    print(f"\nRunning RANDOM agent for {n_episodes} episodes...")
    random_rewards = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        random_rewards.append(total_reward)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Trained PPO':>15} {'Random Agent':>15}")
    print("-" * 55)
    print(f"{'Mean Reward':<25} {np.mean(trained_rewards):>15.2f} {np.mean(random_rewards):>15.2f}")
    print(f"{'Std Reward':<25} {np.std(trained_rewards):>15.2f} {np.std(random_rewards):>15.2f}")
    print(f"{'Min Reward':<25} {np.min(trained_rewards):>15.2f} {np.min(random_rewards):>15.2f}")
    print(f"{'Max Reward':<25} {np.max(trained_rewards):>15.2f} {np.max(random_rewards):>15.2f}")
    
    improvement = ((np.mean(trained_rewards) - np.mean(random_rewards)) 
                   / max(abs(np.mean(random_rewards)), 0.01) * 100)
    
    print(f"\nPPO Improvement over Random: {improvement:+.1f}%")
    print(f"Collision Rate (Trained): {collision_count}/{n_episodes}")
    
    env.close()
    return trained_rewards, random_rewards


if __name__ == "__main__":
    evaluate_trained_agent()
