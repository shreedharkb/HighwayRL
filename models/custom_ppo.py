import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np

class ActorCritic(nn.Module):
    """
    A simple Actor-Critic network for PPO.
    Uses an MLP architecture.
    """
    def __init__(self, obs_shape, action_dim, hidden_sizes=(256, 256)):
        super(ActorCritic, self).__init__()
        
        # Calculate flattened observation size
        self.obs_flat_dim = np.prod(obs_shape)
        
        # Shared feature extractor (optional, but typical in simple MLPs)
        # Here we follow standard practice: separate actor and critic networks.
        
        # ACTOR
        actor_layers = []
        in_size = self.obs_flat_dim
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(in_size, h))
            actor_layers.append(nn.Tanh())
            in_size = h
        actor_layers.append(nn.Linear(in_size, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # CRITIC
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

class CustomPPO:
    """
    Custom implementation of Proximal Policy Optimization (PPO) using PyTorch.
    Designed to replace Stable Baselines 3 completely.
    """
    def __init__(self, envs, learning_rate=3e-4, n_steps=256, batch_size=64,
                 n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="cpu", verbose=1):
        
        self.envs = envs
        self.device = torch.device(device)
        self.verbose = verbose
        
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
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
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        self.num_envs = getattr(envs, "num_envs", 1)
        
    def learn(self, total_timesteps, metrics_callback=None):
        """Train the agent for `total_timesteps`."""
        global_step = 0
        num_updates = total_timesteps // (self.batch_size * self.n_epochs) # Approximation of iterations
        # Exact updates:
        batch_size_per_env = self.n_steps
        global_batch_size = batch_size_per_env * self.num_envs
        num_updates = total_timesteps // global_batch_size
        
        obs_shape = self.envs.single_observation_space.shape if hasattr(self.envs, "single_observation_space") else self.envs.observation_space.shape
        
        # Rollout Storage
        obs = torch.zeros((self.n_steps, self.num_envs) + obs_shape).to(self.device)
        actions = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        logprobs = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.n_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.n_steps, self.num_envs)).to(self.device)

        # Init Env
        next_obs, _ = self.envs.reset(seed=42)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        for update in range(1, num_updates + 1):
            if self.verbose > 0:
                print(f"Update {update}/{num_updates} | Total Steps: {global_step:,}")
            
            # 1. Rollout Phase
            for step in range(self.n_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                with torch.no_grad():
                    action, logprob, _, value = self.network.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Step Environment
                next_obs_np, reward_np, terminated_np, truncated_np, infos = self.envs.step(action.cpu().numpy())
                done_np = np.logical_or(terminated_np, truncated_np)
                
                # Metric Logging
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

            # 2. Compute Advantages (GAE)
            with torch.no_grad():
                next_value = self.network.get_action_and_value(next_obs)[3].flatten()
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flatten batch for training
            b_obs = obs.reshape((-1,) + obs_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # 3. PPO Optimization Phase
            b_inds = np.arange(global_batch_size)
            clipfracs = []
            
            for epoch in range(self.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, global_batch_size, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.network.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Advantage Normalization
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_range,
                        self.clip_range,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    # Entropy Bonus
                    entropy_loss = entropy.mean()
                    
                    # Total Loss
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

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
            if deterministic:
                logits, _ = self.network(obs)
                action = torch.argmax(logits, dim=-1)
            else:
                action, _, _, _ = self.network.get_action_and_value(obs)
        return action.cpu().numpy(), None
