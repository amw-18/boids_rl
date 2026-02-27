import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

class RunningMeanStd(nn.Module):
    """Tracks the mean and variance of a tensor."""
    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)
        
    def denormalize(self, x: torch.Tensor):
        return x * torch.sqrt(self.var + 1e-8) + self.mean

class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) loop for Parameter-Shared Independent PPO (IPPO).
    Every starling shares the same network weights, meaning we pool their 
    experiences together to train the central brain.
    """
    def __init__(
        self,
        env,
        brain: nn.Module,
        device: torch.device,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.015,
        update_epochs: int = 4,
        batch_size: int = 64,
        stacked_frames: int = 3
    ):
        self.env = env
        self.brain = brain.to(device)
        self.optimizer = Adam([
            {'params': self.brain.actor_feature_extractor.parameters(), 'lr': actor_lr},
            {'params': self.brain.actor_mean.parameters(), 'lr': actor_lr},
            {'params': [self.brain.actor_logstd], 'lr': actor_lr},
            {'params': self.brain.critic.parameters(), 'lr': critic_lr}
        ], eps=1e-5)
        self.device = device
        
        self.value_normalizer = RunningMeanStd(shape=()).to(device)
        
        # PPO Hyperparams
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.stacked_frames = stacked_frames

    def collect_rollouts(self, num_steps=200):
        """
        Run the environment for `num_steps`, collecting (obs, actions,
        log_probs, rewards, values) as on-device tensors — no dicts, no numpy.
        """
        obs = self.env.reset()                      # (N, obs_dim) tensor on device
        global_obs = self.env.get_global_state()

        N = self.env.n_agents
        
        # Initialize rolling frame buffers for POMDP resolution
        rolling_obs = obs.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        rolling_global_obs = global_obs.unsqueeze(1).repeat(1, self.stacked_frames, 1)

        # Pre-allocate buffers
        b_obs      = torch.empty((num_steps, N, self.stacked_frames * self.env.obs_dim), device=self.device)
        b_global_obs = torch.empty((num_steps, N, self.stacked_frames * self.env.global_obs_dim), device=self.device)
        b_actions  = torch.empty((num_steps, N, self.env.action_dim), device=self.device)
        b_logprobs = torch.empty((num_steps, N), device=self.device)
        b_rewards  = torch.empty((num_steps, N), device=self.device)
        b_dones    = torch.empty((num_steps, N), device=self.device)
        b_values   = torch.empty((num_steps, N), device=self.device)

        for step in range(num_steps):
            if step > 0:
                global_obs = self.env.get_global_state()
                # Shift frames left and insert new frame at the end
                rolling_obs = torch.cat([rolling_obs[:, 1:, :], obs.unsqueeze(1)], dim=1)
                rolling_global_obs = torch.cat([rolling_global_obs[:, 1:, :], global_obs.unsqueeze(1)], dim=1)

            flat_obs = rolling_obs.view(N, -1)
            flat_global_obs = rolling_global_obs.view(N, -1)
            
            b_obs[step] = flat_obs
            b_global_obs[step] = flat_global_obs

            with torch.no_grad():
                action, logprob, _, norm_value = self.brain.get_action_and_value(flat_obs, flat_global_obs)
                value = self.value_normalizer.denormalize(norm_value)

            next_obs, rewards, dones = self.env.step(action)

            b_actions[step]  = action
            b_logprobs[step] = logprob
            b_rewards[step]  = rewards
            b_dones[step]    = dones.float()
            b_values[step]   = value.flatten()
            
            # Flush history for agents that died to prevent carrying over dead state momentum
            if dones.any():
                dead_mask = dones.bool()
                rolling_obs[dead_mask] = next_obs[dead_mask].unsqueeze(1).repeat(1, self.stacked_frames, 1)
                # Next iteration's global obs will be fetched at the start of the next step

            obs = next_obs

        # Prepare final rolled frames for GAE bootstrapping
        global_obs = self.env.get_global_state()
        rolling_obs = torch.cat([rolling_obs[:, 1:, :], obs.unsqueeze(1)], dim=1)
        rolling_global_obs = torch.cat([rolling_global_obs[:, 1:, :], global_obs.unsqueeze(1)], dim=1)

        return {
            "obs":      b_obs,
            "global_obs": b_global_obs,
            "actions":  b_actions,
            "logprobs": b_logprobs,
            "rewards":  b_rewards,
            "dones":    b_dones,
            "values":   b_values,
            "final_obs": rolling_obs.view(N, -1),
            "final_global_obs": rolling_global_obs.view(N, -1),
        }

    def compute_advantages(self, rollouts):
        """Generalized Advantage Estimation (GAE) across all agents simultaneously."""
        rewards = rollouts["rewards"]
        dones   = rollouts["dones"]
        values  = rollouts["values"]

        # Bootstrap value for next step
        final_obs = rollouts["final_obs"]   # (N, 16) tensor — always present
        final_global_obs = rollouts["final_global_obs"]
        with torch.no_grad():
            _, _, _, norm_next_value = self.brain.get_action_and_value(final_obs, final_global_obs)
            next_value = self.value_normalizer.denormalize(norm_next_value).flatten()

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        num_steps = len(rewards)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]

            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def train_step(self, rollouts):
        b_obs = rollouts["obs"]
        b_global_obs = rollouts["global_obs"]
        b_actions = rollouts["actions"]
        b_logprobs = rollouts["logprobs"]
        b_values = rollouts["values"]
        
        b_advantages, b_returns = self.compute_advantages(rollouts)
        
        b_dones = rollouts["dones"]
        num_steps, num_agents = b_dones.shape
        
        # Create a valid_mask indicating if an agent is physically alive at step t.
        # If an agent died at t-1 (dones[t-1] == True), the transition at t is a dead dummy frame.
        valid_mask = torch.ones((num_steps, num_agents), dtype=torch.bool, device=self.device)
        valid_mask[1:] = ~b_dones[:-1].bool()
        
        # Flatten all batches using the validity mask to discard padding
        mb_obs = b_obs[valid_mask]
        mb_global_obs = b_global_obs[valid_mask]
        mb_actions = b_actions[valid_mask]
        mb_logprobs = b_logprobs[valid_mask]
        
        mb_advantages = b_advantages[valid_mask]
        mb_returns = b_returns[valid_mask]
        
        # Calculate explained variance using unnormalized values and returns
        y_pred = b_values[valid_mask].cpu().numpy()
        y_true = mb_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # Update the value normalizer target statistics and generate normalized returns for training
        self.value_normalizer.update(mb_returns)
        mb_returns_norm = self.value_normalizer.normalize(mb_returns)
        
        # Normalize advantages based ONLY on valid active transitions
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        
        batch_size_total = mb_obs.shape[0]
        inds = np.arange(batch_size_total)
        
        # Check if we have enough data for batch size, otherwise lower it
        batch_size = min(self.batch_size, batch_size_total)
        
        # Accumulate metrics across minibatches for logging
        epoch_pg_losses = []
        epoch_v_losses = []
        epoch_entropies = []
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)
            approx_kl_divs = []
            
            for start in range(0, batch_size_total, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]
                
                # Fetch mini-batch data
                mini_obs = mb_obs[mb_inds]
                mini_global_obs = mb_global_obs[mb_inds]
                mini_actions = mb_actions[mb_inds]
                mini_advantages = mb_advantages[mb_inds]
                mini_returns_norm = mb_returns_norm[mb_inds]
                mini_logprobs = mb_logprobs[mb_inds]
                
                _, newlogprob, entropy, newvalue = self.brain.get_action_and_value(mini_obs, mini_global_obs, mini_actions)
                logratio = newlogprob - mini_logprobs
                ratio = logratio.exp()
                
                # Policy Loss
                pg_loss1 = -mini_advantages * ratio
                pg_loss2 = -mini_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss: MSE against NORMALIZED returns
                v_loss = 0.5 * ((newvalue.flatten() - mini_returns_norm) ** 2).mean()
                
                # Calculate approximate KL divergence for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                approx_kl_divs.append(approx_kl.item())

                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.brain.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store minibatch metrics
                epoch_pg_losses.append(pg_loss.item())
                epoch_v_losses.append(v_loss.item())
                epoch_entropies.append(entropy_loss.item())
                
            # Early stopping at epoch level
            if np.mean(approx_kl_divs) > self.target_kl:
                print(f"Early stopping at epoch {epoch} due to reaching max KL.")
                break
                
        # Return MEAN losses across all minibatches, not just the very last one
        return np.mean(epoch_pg_losses), np.mean(epoch_v_losses), np.mean(epoch_entropies), mb_returns.mean().item(), explained_var
