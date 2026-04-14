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



class MyPPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MyPPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def get_action_value(self, state):
        logits = self.actor(state)
        value = self.critic(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, state, action):
        logits = self.actor(state)
        value = self.critic(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


def train_ppo(env_name="highway-fast-v0", total_timesteps=50000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training PPO on {device} ---")
    
  
    env = gym.make(env_name, render_mode=None)
    env.unwrapped.configure({
        "observation": {"type": "Kinematics", "vehicles_count": 10},
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4, "vehicles_count": 15, "duration": 60,
        "reward_speed_range": [20, 35]
    })
    
   
    env.reset()
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    
    sample_obs, _ = env.reset()
    sample_obs_flat = np.array(sample_obs).flatten()
    state_dim = sample_obs_flat.shape[0]
    action_dim = env.action_space.n
    agent = MyPPOAgent(state_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    n_steps, batch_size, n_epochs = 256, 64, 10
    gamma, gae_lambda, clip_epsilon, entropy_coef, critic_coef = 0.99, 0.95, 0.2, 0.01, 0.5
    
    states = torch.zeros((n_steps, state_dim)).to(device)
    actions = torch.zeros((n_steps,)).to(device)
    log_probs = torch.zeros((n_steps,)).to(device)
    rewards = torch.zeros((n_steps,)).to(device)
    dones = torch.zeros((n_steps,)).to(device)
    values = torch.zeros((n_steps,)).to(device)
    
    current_state, _ = env.reset()
    current_state = torch.tensor(current_state, dtype=torch.float32).view(-1).to(device)
    current_done = torch.tensor(0.0).to(device)
    
    num_iterations = total_timesteps // n_steps
    global_step = 0
    
    metrics = {"timesteps": [], "rewards": [], "lengths": []}
    start_time = time.time()
    
    for iteration in range(1, num_iterations + 1):
        for step in range(n_steps):
            global_step += 1
            states[step], dones[step] = current_state, current_done
            
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_value(current_state.unsqueeze(0))
                values[step] = value.squeeze()
                
            actions[step], log_probs[step] = action.squeeze(), log_prob.squeeze()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = np.logical_or(terminated, truncated)
            
            if "episode" in info:
                metrics["rewards"].append(float(info["episode"]["r"]))
                metrics["lengths"].append(int(info["episode"]["l"]))
                metrics["timesteps"].append(global_step)
            
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            current_state = torch.tensor(next_state, dtype=torch.float32).view(-1).to(device)
            current_done = torch.tensor(float(done)).to(device)
            
            if done:
                next_state, _ = env.reset()
                current_state = torch.tensor(next_state, dtype=torch.float32).view(-1).to(device)
            
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_value(current_state.unsqueeze(0))
            next_value = next_value.squeeze()
            advantages = torch.zeros_like(rewards).to(device)
            last_gae = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_non_terminal = 1.0 - current_done
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - dones[t+1]
                    next_val = values[t+1]
                target = rewards[t] + gamma * next_val * next_non_terminal
                td_error = target - values[t]
                advantages[t] = last_gae = td_error + gamma * gae_lambda * next_non_terminal * last_gae
            returns = advantages + values
            
        b_states = states.view(-1, state_dim)
        b_actions = actions.view(-1)
        b_log_probs = log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        batch_indices = np.arange(n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, len(batch_indices), batch_size):
                end = start + batch_size
                mb_idx = batch_indices[start:end]
                new_log_probs, new_entropy, new_values = agent.evaluate_actions(b_states[mb_idx], b_actions[mb_idx])
                new_values = new_values.squeeze()
                ratio = torch.exp(new_log_probs - b_log_probs[mb_idx])
                unclipped = ratio * b_advantages[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advantages[mb_idx]
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = nn.MSELoss()(new_values, b_returns[mb_idx])
                entropy_loss = new_entropy.mean()
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
    torch.save(agent.state_dict(), "./models/my_custom_ppo.pth")
    with open("./results/ppo_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("PPO Model and Metrics Saved.")

if __name__ == "__main__":
    train_ppo()