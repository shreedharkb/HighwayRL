"""
explore_env.py - Explore the Highway-v0 Environment
"""

import gymnasium as gym
import highway_env
import numpy as np

print("=" * 60)
print("EXPLORING THE HIGHWAY-V0 ENVIRONMENT")
print("=" * 60)

env = gym.make('highway-v0', render_mode='rgb_array')

# Observation Space
print("\nOBSERVATION SPACE:")
print(f"  Type: {env.observation_space}")
print(f"  Shape: {env.observation_space.shape}")
print(f"  Each row = [presence, x, y, vx, vy]")
print(f"  Row 0 = ego vehicle, Rows 1-4 = nearby vehicles")

# Action Space
print("\nACTION SPACE:")
print(f"  Type: {env.action_space}")
print(f"  Number of actions: {env.action_space.n}")

actions = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER"
}
for action_id, desc in actions.items():
    print(f"  Action {action_id}: {desc}")

# Run random agent baseline
print("\nRUNNING RANDOM AGENT (Baseline)...")
NUM_EPISODES = 10
episode_rewards = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

print(f"\nRANDOM AGENT RESULTS:")
print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
print(f"  Std Reward:  {np.std(episode_rewards):.2f}")
print(f"  Min Reward:  {np.min(episode_rewards):.2f}")
print(f"  Max Reward:  {np.max(episode_rewards):.2f}")

print("\nConclusion: Random agent performs poorly. PPO should do much better!")
env.close()
