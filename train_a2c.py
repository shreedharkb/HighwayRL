import os
import time
import json
import numpy as np
import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# ------------------------------------------------------------------
# 1. CUSTOM A2C AGENT
# ------------------------------------------------------------------
class MyA2CAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MyA2CAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

# ------------------------------------------------------------------
# 2. A2C TRAINING (WITH METRICS)
# ------------------------------------------------------------------
def train_a2c(env_name="highway-fast-v0", total_timesteps=50000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training A2C on {device} ---")
    
    # Using a single environment to guarantee 100% bug-free execution
    env = gym.make(env_name, render_mode=None)
    env.unwrapped.configure({
        "observation": {"type": "Kinematics", "vehicles_count": 10},
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4, "vehicles_count": 15, "duration": 60,
        "reward_speed_range": [20, 35]
    })
    
    # Do exactly one reset to give the env a chance to build the 'real' shape
    env.reset()
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # We must compute state_dim from what the env ACTUALLY returns, not the raw space,
    # as highway-env sometimes outputs flattened arrays vs structured ones depending on version
    sample_obs, _ = env.reset()
    sample_obs_flat = np.array(sample_obs).flatten()
    state_dim = sample_obs_flat.shape[0]
    action_dim = env.action_space.n
    agent = MyA2CAgent(state_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=7e-4) # Slightly higher LR for A2C
    
    n_steps, gamma, entropy_coef, critic_coef = 5, 0.99, 0.01, 0.5
    
    current_state, _ = env.reset()
    current_state = torch.tensor(current_state, dtype=torch.float32).view(-1).to(device)
    
    num_updates = total_timesteps // n_steps
    global_step = 0
    metrics = {"timesteps": [], "rewards": [], "lengths": []}
    
    start_time = time.time()
    
    for update in range(1, num_updates + 1):
        log_probs_list, values_list, rewards_list, dones_list, entropies_list = [], [], [], [], []
        
        for step in range(n_steps):
            global_step += 1
            logits, value = agent(current_state.unsqueeze(0))
            values_list.append(value.squeeze())
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            log_probs_list.append(dist.log_prob(action).squeeze())
            entropies_list.append(dist.entropy().squeeze())
            
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = np.logical_or(terminated, truncated)
            
            if "episode" in info:
                metrics["rewards"].append(float(info["episode"]["r"]))
                metrics["lengths"].append(int(info["episode"]["l"]))
                metrics["timesteps"].append(global_step)
            
            rewards_list.append(torch.tensor(reward, dtype=torch.float32).to(device))
            dones_list.append(torch.tensor(float(done), dtype=torch.float32).to(device))
            current_state = torch.tensor(next_state, dtype=torch.float32).view(-1).to(device)
            
            if done:
                next_state, _ = env.reset()
                current_state = torch.tensor(next_state, dtype=torch.float32).view(-1).to(device)
            
        with torch.no_grad():
            _, next_value = agent(current_state.unsqueeze(0))
            next_value = next_value.squeeze()
            returns = []
            R = next_value
            for i in reversed(range(n_steps)):
                R = rewards_list[i] + gamma * R * (1.0 - dones_list[i])
                returns.insert(0, R)
                
        log_probs = torch.stack(log_probs_list).view(-1)
        values = torch.stack(values_list).view(-1)
        returns = torch.stack(returns).view(-1)
        entropies = torch.stack(entropies_list).view(-1)
        
        advantages = returns - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()
        total_loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        optimizer.step()
        
        elapsed_time = time.time() - start_time
        fps = global_step / elapsed_time
        remaining_steps = total_timesteps - global_step
        eta_seconds = remaining_steps / fps if fps > 0 else 0
        
        progress = global_step / total_timesteps
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {progress*100:.1f}% | Steps: {global_step}/{total_timesteps} | FPS: {fps:.0f} | ETA: {eta_seconds:.0f}s | Loss: {total_loss.item():.4f}", end="", flush=True)

    print("\n")
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    torch.save(agent.state_dict(), "./models/my_custom_a2c.pth")
    with open("./results/a2c_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("A2C Model and Metrics Saved.")

if __name__ == "__main__":
    train_a2c()