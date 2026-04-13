import os
import json
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import highway_env

# Import the actual models
from train_ppo import MyPPOAgent
from train_a2c import MyA2CAgent

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def evaluate_agent(agent_class, model_path, env_name, total_steps=50000):
    print(f"Evaluating {model_path} for {total_steps} steps...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make(env_name, render_mode=None)
    env.unwrapped.configure({
        "observation": {"type": "Kinematics", "vehicles_count": 10},
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4, "vehicles_count": 15, "duration": 60, # Reduced vehicles_count from 30 to 15 for faster physics
        "reward_speed_range": [20, 35]
    })
    
    sample_obs, _ = env.reset()
    state_dim = np.array(sample_obs).flatten().shape[0]
    action_dim = env.action_space.n
    agent = agent_class(state_dim, action_dim).to(device)
    
    # Only try to load if the model exists (so it doesn't crash if you haven't trained yet)
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"Warning: {model_path} not found. Operating with random initialization.")
    agent.eval()
    
    returns = []
    lengths = []
    
    global_step = 0
    
    # Display Progress Bar dynamically on the same line
    import time
    start_time = time.time()
    
    while global_step < total_steps:
        state, _ = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        
        while not done and global_step < total_steps:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            state_t = state_t.view(1, -1)
            with torch.no_grad():
                if isinstance(agent, MyPPOAgent):
                    logits = agent.actor(state_t)
                else:
                    logits, _ = agent(state_t)
                    
                action = torch.argmax(logits, dim=-1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_length += 1
            global_step += 1
            
            # Print evaluation progress
            if global_step % 100 == 0 or global_step == total_steps:
                elapsed_time = time.time() - start_time
                fps = global_step / elapsed_time if elapsed_time > 0 else 0
                remaining_steps = total_steps - global_step
                eta_seconds = remaining_steps / fps if fps > 0 else 0
                
                progress = global_step / total_steps
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"\r[{bar}] {progress*100:.1f}% | Steps: {global_step}/{total_steps} | ETA: {eta_seconds:.0f}s", end="", flush=True)

        returns.append(ep_return)
        lengths.append(ep_length)
        
    print("\nDone Evaluation.")
    env.close()
    return returns, lengths

def main():
    os.makedirs("./results", exist_ok=True)
    print("--- Running Analytics and Evaluating Models ---")
    
    # 1. Evaluate both trained models
    ppo_returns, ppo_lengths = evaluate_agent(MyPPOAgent, "./models/my_custom_ppo.pth", "highway-fast-v0", total_steps=50000)
    a2c_returns, a2c_lengths = evaluate_agent(MyA2CAgent, "./models/my_custom_a2c.pth", "highway-fast-v0", total_steps=50000)
    
    print("\n--- Analytics Report ---")
    print(f"PPO Avg Return: {np.mean(ppo_returns):.2f} +/- {np.std(ppo_returns):.2f}")
    print(f"A2C Avg Return: {np.mean(a2c_returns):.2f} +/- {np.std(a2c_returns):.2f}")
    print(f"PPO Avg Length: {np.mean(ppo_lengths):.2f}")
    print(f"A2C Avg Length: {np.mean(a2c_lengths):.2f}")
    
    # 2. Load Training Logs
    ppo_train_r, a2c_train_r = [], []
    ppo_train_t, a2c_train_t = [], []
    
    if os.path.exists("./results/ppo_metrics.json"):
        with open("./results/ppo_metrics.json", "r") as f:
            ppo_data = json.load(f)
            ppo_train_r = smooth_curve(ppo_data["rewards"])
            ppo_train_t = ppo_data["timesteps"]
            
    if os.path.exists("./results/a2c_metrics.json"):
        with open("./results/a2c_metrics.json", "r") as f:
            a2c_data = json.load(f)
            a2c_train_r = smooth_curve(a2c_data["rewards"])
            a2c_train_t = a2c_data["timesteps"]

    # 3. Create Analytics Plots
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Training Curve (Rewards)
    plt.subplot(1, 3, 1)
    if ppo_train_r: plt.plot(ppo_train_t, ppo_train_r, label="PPO (Smoothed)", alpha=0.8)
    if a2c_train_r: plt.plot(a2c_train_t, a2c_train_r, label="A2C (Smoothed)", alpha=0.8)
    plt.title("Training Learning Curve")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Evaluation Returns Boxplot
    plt.subplot(1, 3, 2)
    plt.boxplot([ppo_returns, a2c_returns], tick_labels=["PPO", "A2C"])
    plt.title("Evaluation Returns (50,000 Steps)")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Evaluation Lengths Bar Chart
    plt.subplot(1, 3, 3)
    means = [np.mean(ppo_lengths), np.mean(a2c_lengths)]
    stds = [np.std(ppo_lengths), np.std(a2c_lengths)]
    plt.bar(["PPO", "A2C"], means, yerr=stds, capsize=10, color=["blue", "orange"], alpha=0.7)
    plt.title("Average Episode Length (Survival)")
    plt.ylabel("Steps")
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("./results/agent_comparison.png")
    print("\nVisual analytics saved successfully to ./results/agent_comparison.png")

if __name__ == "__main__":
    main()