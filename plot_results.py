"""Plot training results from PPO training metrics."""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_training_results():
    metrics_path = "./results/training_metrics.json"
    if not os.path.exists(metrics_path):
        print("No training metrics found. Train the model first.")
        return
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    rewards = metrics["episode_rewards"]
    lengths = metrics["episode_lengths"]
    timesteps = metrics["timesteps"]
    
    # Enable modern aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("PPO Training Dashboard - Highway-v0", fontsize=20, fontweight='heavy', y=0.98, color='#2c3e50')
    
    # Modern color palette
    c_raw = '#AEC6CF'     # light blue for raw data
    c_smooth = '#2980b9'  # deep blue for smoothed data
    c_len_raw = '#FFB347' # pastel orange
    c_len_sm = '#d35400'  # deep orange
    c_ts_raw = '#77DD77'  # pastel green
    c_ts_sm = '#27ae60'   # deep green
    c_hist = '#9b59b6'    # amethyst purple
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.4, color=c_raw, label='Raw Reward')
    if len(rewards) > 10:
        smoothed = moving_average(rewards, window=10)
        ax1.plot(range(9, 9+len(smoothed)), smoothed, color=c_smooth, linewidth=2.5, label='10-Ep Moving Avg')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Reward Progression", fontsize=14, color='#34495e')
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    
    # 2. Episode Lengths
    ax2 = axes[0, 1]
    ax2.plot(lengths, alpha=0.4, color=c_len_raw)
    if len(lengths) > 10:
        smoothed_len = moving_average(lengths, window=10)
        ax2.plot(range(9, 9+len(smoothed_len)), smoothed_len, color=c_len_sm, linewidth=2.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps Survived")
    ax2.set_title("Episode Duration", fontsize=14, color='#34495e')
    
    # 3. Reward vs Timesteps
    ax3 = axes[1, 0]
    ax3.scatter(timesteps, rewards, alpha=0.3, s=15, color=c_ts_raw, edgecolors='none')
    if len(rewards) > 10:
        smoothed = moving_average(rewards, window=10)
        smoothed_ts = timesteps[9:9+len(smoothed)] if len(timesteps) > 9 else timesteps
        ax3.plot(smoothed_ts, smoothed, color=c_ts_sm, linewidth=2.5)
    ax3.set_xlabel("Total Timesteps")
    ax3.set_ylabel("Total Reward")
    ax3.set_title("Sample Efficiency", fontsize=14, color='#34495e')
    
    # 4. Reward Distribution
    ax4 = axes[1, 1]
    n, bins, patches = ax4.hist(rewards, bins=25, color=c_hist, edgecolor='white', alpha=0.85)
    mean_val = np.mean(rewards)
    ax4.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.1f}')
    ax4.set_xlabel("Reward")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Reward Distribution", fontsize=14, color='#34495e')
    ax4.legend(loc='upper right', frameon=True, shadow=True)
    
    # Clean up axes (despine)
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/training_rewards_high_res.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved in high-resolution to {save_path}")

if __name__ == "__main__":
    plot_training_results()
