"""Plot training results from PPO training metrics."""

import json
import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(data, window=10):
    """Compute moving average for smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_results():
    metrics_path = "./results/training_metrics.json"
    
    if not os.path.exists(metrics_path):
        print("No training metrics found. Run train_ppo.py first.")
        return
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    rewards = metrics["episode_rewards"]
    lengths = metrics["episode_lengths"]
    timesteps = metrics["timesteps"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Training Results on Highway-v0", fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')
    if len(rewards) > 10:
        smoothed = moving_average(rewards, window=10)
        ax1.plot(range(9, 9+len(smoothed)), smoothed, color='darkblue', 
                linewidth=2, label='Moving Avg (10)')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Rewards Over Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    ax2 = axes[0, 1]
    ax2.plot(lengths, alpha=0.3, color='coral')
    if len(lengths) > 10:
        smoothed_len = moving_average(lengths, window=10)
        ax2.plot(range(9, 9+len(smoothed_len)), smoothed_len, color='darkred', linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Episode Length Over Training")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward vs Timesteps
    ax3 = axes[1, 0]
    ax3.scatter(timesteps, rewards, alpha=0.3, s=10, color='green')
    if len(rewards) > 10:
        smoothed = moving_average(rewards, window=10)
        smoothed_ts = timesteps[9:9+len(smoothed)] if len(timesteps) > 9 else timesteps
        if len(smoothed_ts) == len(smoothed):
            ax3.plot(smoothed_ts, smoothed, color='darkgreen', linewidth=2)
    ax3.set_xlabel("Timesteps")
    ax3.set_ylabel("Reward")
    ax3.set_title("Reward vs Training Timesteps")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution
    ax4 = axes[1, 1]
    ax4.hist(rewards, bins=20, color='mediumpurple', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.2f}')
    ax4.set_xlabel("Reward")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Reward Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/training_rewards.png", dpi=150, bbox_inches='tight')
    print("Plot saved to ./results/training_rewards.png")
    plt.show()


if __name__ == "__main__":
    plot_training_results()
