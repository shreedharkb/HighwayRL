"""
compare_agents.py - Direct evaluation comparison of Custom PPO, Custom A2C, and Random Baseline
"""

import gymnasium as gym
import highway_env
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from config import ENV_CONFIG, HIGHWAY_ENV_CONFIG, EVAL_CONFIG
from models.custom_ppo import CustomPPO
from models.custom_a2c import CustomA2C

def evaluate_agents():
    print("=" * 70)
    print("PPO vs A2C vs RANDOM DIRECT COMPARISON on Highway-fast-v0")
    print("=" * 70)
    
    env = gym.make(ENV_CONFIG["env_id"], render_mode="rgb_array")
    env.unwrapped.configure(HIGHWAY_ENV_CONFIG)
    env.reset()  # Resets observation space shape before loading weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Models
    ppo_path = "./models/ppo_highway_final"
    a2c_path = "./models/a2c_highway_final"
    
    agents_to_evaluate = [("Random", None)]
    
    try:
        ppo_model = CustomPPO.load(ppo_path, envs=env, device=device)
        print(f"Custom PPO model loaded from: {ppo_path}.pt")
        agents_to_evaluate.insert(0, ("PPO", ppo_model))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Custom PPO model could not be loaded: {e}")

    try:
        a2c_model = CustomA2C.load(a2c_path, envs=env, device=device)
        print(f"Custom A2C model loaded from: {a2c_path}.pt")
        agents_to_evaluate.insert(1, ("A2C", a2c_model))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Custom A2C model could not be loaded: {e}")

    n_episodes = EVAL_CONFIG["n_eval_episodes"]
    
    print(f"\n{'=' * 70}")
    print(f"Running evaluation on {n_episodes} episodes each...")
    print(f"{'=' * 70}\n")
    
    results = {}

    # Evaluate each agent type
    for agent_name, model in agents_to_evaluate:
        print(f"{'EVALUATING ' + agent_name:^70}")
        print("-" * 70)
        
        rewards = []
        lengths = []
        collisions = 0
        
        for episode in range(n_episodes):
            obs, info = env.reset(seed=episode)
            total_reward = 0
            steps = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                if agent_name == "Random":
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    if isinstance(action, np.ndarray):
                        action = action.item()
                
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            
            rewards.append(total_reward)
            lengths.append(steps)
            if done and not truncated:
                collisions += 1
                status = "CRASHED"
            else:
                status = "SURVIVED"
            
            print(f"  Episode {episode + 1:2d}: Reward = {total_reward:7.2f} | Steps = {steps:3d} | {status}")
        
        results[agent_name] = {
            "rewards": rewards,
            "lengths": lengths,
            "collisions": collisions,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "crash_rate": (collisions / n_episodes) * 100
        }
        print(f"\n{agent_name} Done.\n")

    env.close()
    
    # Print comparison table
    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE COMPARISON")
    print(f"{'=' * 70}\n")
    
    agents = list(results.keys())
    headers = ["Metric"] + agents
    print(f"{headers[0]:<20} " + " ".join([f"{a:>12}" for a in agents]))
    print("-" * 70)
    
    mean_str = " ".join([f"{results[a]['mean_reward']:>12.2f}" for a in agents])
    print(f"{'Mean Reward':<20} {mean_str}")
    
    std_str = " ".join([f"{results[a]['std_reward']:>12.2f}" for a in agents])
    print(f"{'Std Dev':<20} {std_str}")
    
    crash_str = " ".join([f"{results[a]['crash_rate']:>11.1f}%" for a in agents])
    print(f"{'Crash Rate %':<20} {crash_str}")
    
    # Generate Evaluation Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    agents = list(results.keys())
    means = [results[a]['mean_reward'] for a in agents]
    stds = [results[a]['std_reward'] for a in agents]
    crash_rates = [results[a]['crash_rate'] for a in agents]
    
    colors = ['#2980b9', '#c0392b', '#7f8c8d'][:len(agents)]
    
    # Mean Rewards
    bars1 = ax1.bar(agents, means, yerr=stds, capsize=8, color=colors, alpha=0.9, edgecolor='black')
    ax1.set_ylabel('Mean Total Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Average Reward', fontsize=14, fontweight='bold')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center', fontweight='bold')

    # Crash Rates
    bars2 = ax2.bar(agents, crash_rates, color=colors, alpha=0.9, hatch='//', edgecolor='black')
    ax2.set_ylabel('Crash Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Safety (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 115)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, f'{bar.get_height():.0f}%', ha='center', fontweight='bold')

    plt.tight_layout()
    os.makedirs("./results", exist_ok=True)
    plt.savefig('./results/plot_evaluation.png', dpi=300)
    print("Evaluation plot saved to ./results/plot_evaluation.png\n")

if __name__ == "__main__":
    evaluate_agents()
