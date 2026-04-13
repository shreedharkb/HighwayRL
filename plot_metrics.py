import os
import json
import numpy as np
import matplotlib.pyplot as plt

def get_rolling_stats(data, window=100):
    """Calculates a moving average to create smooth, readable curves."""
    data = np.array(data)
    if len(data) < window:
        window = len(data) // 2 or 1
    means = np.convolve(data, np.ones(window)/window, mode='valid')
    return means

def main():
    print("--- Generating 4 Separate Professional Plots ---")
    
    ppo_data, a2c_data = None, None
    metrics_dir = "./results"
    
    if os.path.exists(f"{metrics_dir}/ppo_metrics.json"):
        with open(f"{metrics_dir}/ppo_metrics.json", "r") as f:
            ppo_data = json.load(f)
            
    if os.path.exists(f"{metrics_dir}/a2c_metrics.json"):
        with open(f"{metrics_dir}/a2c_metrics.json", "r") as f:
            a2c_data = json.load(f)

    if not ppo_data and not a2c_data:
        print("No metrics JSON files found.")
        return

    # We use the last 100 episodes as our exact "Final Evaluation" stand-in
    # This prevents the boxplot from being messy and unreadable with early training fails.
    eval_window = 100
    roll_window = 100

    # ---------------------------------------------------------
    # Plot 1: Learning Curve (Rewards)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    if ppo_data:
        means = get_rolling_stats(ppo_data["rewards"], window=roll_window)
        plt.plot(ppo_data["timesteps"][:len(means)], means, label="PPO", color="#1f77b4", linewidth=2.5)
    if a2c_data:
        means = get_rolling_stats(a2c_data["rewards"], window=roll_window)
        plt.plot(a2c_data["timesteps"][:len(means)], means, label="A2C", color="#ff7f0e", linewidth=2.5)
    plt.title("Training Learning Curve (Rewards)", fontsize=16, fontweight='bold')
    plt.xlabel("Timesteps", fontsize=14)
    plt.ylabel(f"Average Reward (Rolling Window={roll_window})", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/plot1_learning_curve.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Survival Time (Episode Lengths)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    if ppo_data:
        means = get_rolling_stats(ppo_data["lengths"], window=roll_window)
        plt.plot(ppo_data["timesteps"][:len(means)], means, label="PPO", color="#1f77b4", linewidth=2.5, linestyle="--")
    if a2c_data:
        means = get_rolling_stats(a2c_data["lengths"], window=roll_window)
        plt.plot(a2c_data["timesteps"][:len(means)], means, label="A2C", color="#ff7f0e", linewidth=2.5, linestyle="--")
    plt.title("Agent Survival Time (Episode Lengths)", fontsize=16, fontweight='bold')
    plt.xlabel("Timesteps", fontsize=14)
    plt.ylabel(f"Steps Survived (Rolling Window={roll_window})", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/plot2_survival_time.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Final Evaluation Reward Distribution (Histogram)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    if ppo_data and len(ppo_data["rewards"]) >= eval_window:
        plt.hist(ppo_data["rewards"][-eval_window:], bins=15, alpha=0.7, label="PPO", color="#1f77b4", edgecolor='white')
    if a2c_data and len(a2c_data["rewards"]) >= eval_window:
        plt.hist(a2c_data["rewards"][-eval_window:], bins=15, alpha=0.7, label="A2C", color="#ff7f0e", edgecolor='white')
        
    plt.title(f"Final Policy Reward Distribution (Last {eval_window} Episodes)", fontsize=16, fontweight='bold')
    plt.xlabel("Reward Score", fontsize=14)
    plt.ylabel("Frequency (Number of Episodes)", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/plot3_eval_distribution.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Plot 4: Final Policy Episode Lengths - Bar Chart
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    bars, means, stds, bar_colors = [], [], [], []
    if ppo_data and len(ppo_data["lengths"]) >= eval_window:
        bars.append("PPO")
        means.append(np.mean(ppo_data["lengths"][-eval_window:]))
        stds.append(np.std(ppo_data["lengths"][-eval_window:]))
        bar_colors.append("#1f77b4")
    if a2c_data and len(a2c_data["lengths"]) >= eval_window:
        bars.append("A2C")
        means.append(np.mean(a2c_data["lengths"][-eval_window:]))
        stds.append(np.std(a2c_data["lengths"][-eval_window:]))
        bar_colors.append("#ff7f0e")
    
    if bars:
        plt.bar(bars, means, yerr=stds, capsize=12, color=bar_colors, alpha=0.8, width=0.5, edgecolor='black', linewidth=1.2)
    plt.title(f"Average Survival Steps (Last {eval_window} Episodes)", fontsize=16, fontweight='bold')
    plt.ylabel("Mean Steps Survived", fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{metrics_dir}/plot4_eval_barchart.png", dpi=300)
    plt.close()

    print("Successfully generated 4 separate high-resolution plots for LaTeX:")
    print(" 1. ./results/plot1_learning_curve.png")
    print(" 2. ./results/plot2_survival_time.png")
    print(" 3. ./results/plot3_eval_boxplot.png")
    print(" 4. ./results/plot4_eval_barchart.png")

if __name__ == "__main__":
    main()