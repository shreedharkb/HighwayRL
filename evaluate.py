"""
evaluate.py - Evaluate the Trained PPO Agent
=============================================
This script:
    1. Loads the trained PPO model
    2. Runs it on the Highway environment
    3. Compares trained agent vs random agent
    4. Prints detailed performance statistics
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from config import ENV_CONFIG, EVAL_CONFIG


def evaluate_trained_agent():
    """
    Evaluate the trained PPO agent and compare with random baseline.
    """
    
    print("=" * 60)
    print("🧪 EVALUATING TRAINED PPO AGENT")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # STEP 1: Load the Trained Model
    # ----------------------------------------------------------
    model_path = "./models/ppo_highway_final"
    
    try:
        model = PPO.load(model_path)
        print(f"\n✅ Model loaded from: {model_path}")
    except FileNotFoundError:
        # Try loading the best model saved by EvalCallback
        try:
            model = PPO.load("./models/best_model")
            print(f"\n✅ Best model loaded from: ./models/best_model")
        except FileNotFoundError:
            print("\n❌ No trained model found!")
            print("   Run 'python train_ppo.py' first to train the agent.")
            return
    
    # ----------------------------------------------------------
    # STEP 2: Create Evaluation Environment
    # ----------------------------------------------------------
    env = gym.make(ENV_CONFIG["env_id"], render_mode="rgb_array")
    
    n_episodes = EVAL_CONFIG["n_eval_episodes"]
    deterministic = EVAL_CONFIG["deterministic"]
    
    # ----------------------------------------------------------
    # STEP 3: Evaluate Trained Agent
    # ----------------------------------------------------------
    print(f"\n🤖 Running TRAINED agent for {n_episodes} episodes...")
    print(f"   Deterministic: {deterministic}")
    
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
            # The trained agent uses the learned policy
            # deterministic=True means we always pick the best action
            # (no exploration noise)
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        trained_rewards.append(total_reward)
        trained_lengths.append(steps)
        
        # Check if episode ended due to collision
        if done and not truncated:
            collision_count += 1
        
        print(f"  Episode {episode + 1:2d}: Reward = {total_reward:7.2f}, "
              f"Steps = {steps:3d}, "
              f"{'💥 CRASH' if (done and not truncated) else '✅ SURVIVED'}")
    
    # ----------------------------------------------------------
    # STEP 4: Evaluate Random Agent (Baseline Comparison)
    # ----------------------------------------------------------
    print(f"\n🎲 Running RANDOM agent for {n_episodes} episodes...")
    
    random_rewards = []
    random_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        random_rewards.append(total_reward)
        random_lengths.append(steps)
    
    # ----------------------------------------------------------
    # STEP 5: Print Comparison Results
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Trained PPO':>15} {'Random Agent':>15}")
    print("-" * 55)
    print(f"{'Mean Reward':<25} {np.mean(trained_rewards):>15.2f} {np.mean(random_rewards):>15.2f}")
    print(f"{'Std Reward':<25} {np.std(trained_rewards):>15.2f} {np.std(random_rewards):>15.2f}")
    print(f"{'Min Reward':<25} {np.min(trained_rewards):>15.2f} {np.min(random_rewards):>15.2f}")
    print(f"{'Max Reward':<25} {np.max(trained_rewards):>15.2f} {np.max(random_rewards):>15.2f}")
    print(f"{'Mean Episode Length':<25} {np.mean(trained_lengths):>15.1f} {np.mean(random_lengths):>15.1f}")
    
    # Calculate improvement
    improvement = ((np.mean(trained_rewards) - np.mean(random_rewards)) 
                   / max(abs(np.mean(random_rewards)), 0.01) * 100)
    
    print(f"\n🏆 PPO Improvement over Random: {improvement:+.1f}%")
    print(f"💥 Collision Rate (Trained): {collision_count}/{n_episodes} "
          f"({collision_count/n_episodes*100:.0f}%)")
    
    print("\n" + "=" * 60)
    if np.mean(trained_rewards) > np.mean(random_rewards):
        print("✅ PPO agent significantly outperforms random agent!")
    else:
        print("⚠️  Training may need more timesteps for better results.")
    print("=" * 60)
    
    env.close()
    
    return trained_rewards, random_rewards


if __name__ == "__main__":
    evaluate_trained_agent()
