"""
explore_env.py - Explore the Highway-v0 Environment
=====================================================
This script helps us understand the Highway environment before training.
We explore:
    1. What the observation space looks like
    2. What actions are available
    3. How the reward function works
    4. How a random agent performs (baseline)
"""

import gymnasium as gym
import numpy as np

# ============================================================
# STEP 1: Create the Highway Environment
# ============================================================
# highway-env is a collection of environments for autonomous driving
# It is NOT part of standard OpenAI Gym — it's a separate package
# The highway-v0 environment simulates highway driving with traffic

print("=" * 60)
print("EXPLORING THE HIGHWAY-V0 ENVIRONMENT")
print("=" * 60)

# Create the environment
# 'highway-v0' simulates a multi-lane highway with NPC vehicles
env = gym.make('highway-v0', render_mode='rgb_array')

# ============================================================
# STEP 2: Understand the Observation Space
# ============================================================
print("\n📊 OBSERVATION SPACE:")
print(f"  Type: {env.observation_space}")
print(f"  Shape: {env.observation_space.shape}")
print(f"  Low: {env.observation_space.low[0]}")
print(f"  High: {env.observation_space.high[0]}")

# The observation is a matrix where each row represents a vehicle:
# Row 0: The ego vehicle (our agent)
# Rows 1-4: Nearby vehicles
# Each row contains: [presence, x, y, vx, vy]
#   - presence: 1 if vehicle exists, 0 otherwise
#   - x: longitudinal position (along the road)
#   - y: lateral position (across lanes)
#   - vx: longitudinal velocity
#   - vy: lateral velocity

print("\n  Each row = [presence, x, y, vx, vy]")
print("  Row 0 = ego vehicle (our agent)")
print("  Rows 1-4 = nearby vehicles")

# ============================================================
# STEP 3: Understand the Action Space
# ============================================================
print("\n🎮 ACTION SPACE:")
print(f"  Type: {env.action_space}")
print(f"  Number of actions: {env.action_space.n}")

# The 5 discrete actions available are:
actions = {
    0: "LANE_LEFT  — Change to the left lane",
    1: "IDLE       — Stay in current lane, maintain speed",
    2: "LANE_RIGHT — Change to the right lane",
    3: "FASTER     — Accelerate",
    4: "SLOWER     — Decelerate"
}

for action_id, description in actions.items():
    print(f"  Action {action_id}: {description}")

# ============================================================
# STEP 4: Run a Random Agent (Baseline Performance)
# ============================================================
print("\n🎲 RUNNING RANDOM AGENT (Baseline)...")
print("  This shows how a random policy performs")

NUM_EPISODES = 10
episode_rewards = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        # Random action — no intelligence, just random choices
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    episode_rewards.append(total_reward)
    print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")

print(f"\n📈 RANDOM AGENT RESULTS:")
print(f"  Mean Reward: {np.mean(episode_rewards):.2f}")
print(f"  Std Reward:  {np.std(episode_rewards):.2f}")
print(f"  Min Reward:  {np.min(episode_rewards):.2f}")
print(f"  Max Reward:  {np.max(episode_rewards):.2f}")

# ============================================================
# STEP 5: Understanding the Reward Function
# ============================================================
print("\n💰 REWARD FUNCTION:")
print("  The reward in highway-v0 combines:")
print("  1. Speed reward: Higher reward for going faster")
print("  2. Collision penalty: Large negative reward for crashes")
print("  3. Lane reward: Reward for being in the rightmost lanes")
print("  Goal: Learn to drive fast WITHOUT crashing!")

print("\n" + "=" * 60)
print("CONCLUSION: Random agent performs poorly.")
print("We need PPO to learn an intelligent driving policy!")
print("=" * 60)

env.close()
