import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class ActorCritic(nn.Module):
    """
    Standard Actor-Critic network.
    Shared hidden layers, separate actor and critic heads.
    """
    def __init__(self, obs_shape, action_dim, hidden_sizes=(256, 256)):
        super(ActorCritic, self).__init__()
        self.obs_flat_dim = np.prod(obs_shape)
        
        # Actor head
        actor_layers = []
        in_size = self.obs_flat_dim
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(in_size, h))
            actor_layers.append(nn.Tanh())
            in_size = h
        actor_layers.append(nn.Linear(in_size, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic head
        critic_layers = []
        in_size = self.obs_flat_dim
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(in_size, h))
            critic_layers.append(nn.Tanh())
            in_size = h
        critic_layers.append(nn.Linear(in_size, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, obs):
        x = obs.view(-1, self.obs_flat_dim)
        logits = self.actor(x)
        val = self.critic(x)
        return logits, val
    
    def get_action_and_value(self, obs, action=None):
        logits, val = self.forward(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), val

class CustomA2C:
    """
    Custom implementation of Advantage Actor-Critic (A2C) using PyTorch.
    Follows synchronous implementation style.
    """
    def __init__(self, envs, learning_rate=7e-4, n_steps=5, gamma=0.99,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="cpu", verbose=1):
        
        self.envs = envs
        self.device = torch.device(device)
        self.verbose = verbose
        
        self.n_steps = n_steps
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        # Extract shapes
        if hasattr(envs, "single_observation_space"):
            obs_shape = envs.single_observation_space.shape
        else:
            obs_shape = envs.observation_space.shape
            
        if hasattr(envs, "single_action_space"):
            action_dim = envs.single_action_space.n
        else:
            action_dim = envs.action_space.n

        self.network = ActorCritic(obs_shape, action_dim).to(self.device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=learning_rate, eps=1e-5, alpha=0.99)
        self.num_envs = getattr(envs, "num_envs", 1)
        
    def learn(self, total_timesteps, metrics_callback=None):
        """Train the agent for `total_timesteps`."""
        global_step = 0
        num_updates = total_timesteps // (self.n_steps * self.num_envs)
        
        obs_shape = self.envs.single_observation_space.shape if hasattr(self.envs, "single_observation_space") else self.envs.observation_space.shape
        
        # Storage
        states = torch.zeros((self.n_steps, self.num_envs) + obs_shape).to(self.device)
        actions = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.n_steps, self.num_envs)).to(self.device)

        # Init Env
        next_obs, _ = self.envs.reset(seed=42)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        for update in range(1, num_updates + 1):
            if update % 20 == 0 and self.verbose > 0:
                print(f"Update {update}/{num_updates} | Steps: {global_step:,}")
            
            # 1. Collection Phase
            for step in range(self.n_steps):
                global_step += self.num_envs
                states[step] = next_obs
                dones[step] = next_done
                
                with torch.no_grad():
                    action, _, _, value = self.network.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action

                # Step Environment
                next_obs_np, reward_np, terminated_np, truncated_np, infos = self.envs.step(action.cpu().numpy())
                done_np = np.logical_or(terminated_np, truncated_np)
                
                if metrics_callback is not None:
                    if "episode" in infos and "_episode" in infos:
                        mask = infos["_episode"]
                        if np.any(mask):
                            r_arr = infos["episode"]["r"][mask]
                            l_arr = infos["episode"]["l"][mask]
                            for r, l in zip(r_arr, l_arr):
                                metrics_callback.on_episode_end(float(r), int(l), global_step)
                    elif "final_info" in infos:
                        for final_inf in infos["final_info"]:
                            if final_inf is not None and "episode" in final_inf:
                                r = final_inf["episode"]["r"][0] if isinstance(final_inf["episode"]["r"], np.ndarray) else final_inf["episode"]["r"]
                                l = final_inf["episode"]["l"][0] if isinstance(final_inf["episode"]["l"], np.ndarray) else final_inf["episode"]["l"]
                                metrics_callback.on_episode_end(float(r), int(l), global_step)

                rewards[step] = torch.tensor(reward_np).to(self.device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(self.device)
                next_done = torch.Tensor(done_np).to(self.device)

            # 2. Advantage & Returns Calculation
            with torch.no_grad():
                next_value = self.network.get_action_and_value(next_obs)[3].flatten()
                returns = torch.zeros_like(rewards).to(self.device)
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + self.gamma * next_return * nextnonterminal
                advantages = returns - values

            # Flatten for batch processing
            b_states = states.reshape((-1,) + obs_shape)
            b_actions = actions.reshape(-1)
            b_returns = returns.reshape(-1)
            b_advantages = advantages.reshape(-1)
            
            # Advantage Normalization
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # 3. Optimization Phase
            logits, b_values_new = self.network(b_states)
            probs = Categorical(logits=logits)
            logprobs = probs.log_prob(b_actions)
            entropy = probs.entropy().mean()
            
            # Policy Loss
            policy_loss = -(logprobs * b_advantages).mean()
            
            # Value Loss
            value_loss = 0.5 * ((b_returns - b_values_new.view(-1))**2).mean()
            
            # Total Loss
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return self

    def save(self, path):
        torch.save(self.network.state_dict(), f"{path}.pt")

    @classmethod
    def load(cls, path, envs, device="cpu"):
        model = cls(envs, device=device)
        model.network.load_state_dict(torch.load(f"{path}.pt", map_location=device))
        model.network.eval()
        return model

    def predict(self, obs, deterministic=True):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.network(obs)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = Categorical(logits=logits)
                action = probs.sample()
        return action.cpu().numpy(), None
