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

class AlternatingCoevolutionTrainer:
    """
    Co-Evolutionary Proximal Policy Optimization (PPO) loop.
    Manages two completely decoupled PPO networks (Starlings vs Predators)
    competing in a zero-sum, asymmetric game.
    """
    def __init__(
        self,
        env,
        boid_brain: nn.Module,
        pred_brain: nn.Module,
        device: torch.device,
        boid_actor_lr: float = 3e-4, # Higher LR for prey to learn faster initially
        pred_actor_lr: float = 1e-4, # Lower LR for predators to prevent early collapse
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
        self.device = device
        
        # --- Boid (Starling) Population ---
        self.boid_brain = boid_brain.to(device)
        self.boid_optimizer = Adam([
            {'params': self.boid_brain.actor_feature_extractor.parameters(), 'lr': boid_actor_lr},
            {'params': self.boid_brain.actor_mean.parameters(), 'lr': boid_actor_lr},
            {'params': [self.boid_brain.actor_logstd], 'lr': boid_actor_lr},
            {'params': self.boid_brain.critic.parameters(), 'lr': critic_lr}
        ], eps=1e-5)
        self.boid_value_norm = RunningMeanStd(shape=()).to(device)
        
        # --- Predator (Falcon) Population ---
        self.pred_brain = pred_brain.to(device)
        self.pred_optimizer = Adam([
            {'params': self.pred_brain.actor_feature_extractor.parameters(), 'lr': pred_actor_lr},
            {'params': self.pred_brain.actor_mean.parameters(), 'lr': pred_actor_lr},
            {'params': [self.pred_brain.actor_logstd], 'lr': pred_actor_lr},
            {'params': self.pred_brain.critic.parameters(), 'lr': critic_lr}
        ], eps=1e-5)
        self.pred_value_norm = RunningMeanStd(shape=()).to(device)
        
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
        Run the environment for `num_steps`, collecting joint experience for both populations simultaneously.
        """
        obs_boids, obs_preds = self.env.reset()
        global_obs_boids = self.env.get_global_state(obs_boids)
        global_obs_preds = self.env.get_global_state(obs_preds)

        N = self.env.n_agents
        P = self.env.num_predators
        
        # Buffers
        roll_obs_boids = obs_boids.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        roll_global_boids = global_obs_boids.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        
        roll_obs_preds = obs_preds.unsqueeze(1).repeat(1, self.stacked_frames, 1)
        roll_global_preds = global_obs_preds.unsqueeze(1).repeat(1, self.stacked_frames, 1)

        b_obs       = torch.empty((num_steps, N, self.stacked_frames * obs_boids.shape[-1]), device=self.device)
        b_globs     = torch.empty((num_steps, N, self.stacked_frames * global_obs_boids.shape[-1]), device=self.device)
        b_acts      = torch.empty((num_steps, N, self.env.action_dim), device=self.device)
        b_logps     = torch.empty((num_steps, N), device=self.device)
        b_rews      = torch.empty((num_steps, N), device=self.device)
        b_dones     = torch.empty((num_steps, N), device=self.device)
        b_vals      = torch.empty((num_steps, N), device=self.device)

        p_obs       = torch.empty((num_steps, P, self.stacked_frames * obs_preds.shape[-1]), device=self.device)
        p_globs     = torch.empty((num_steps, P, self.stacked_frames * global_obs_preds.shape[-1]), device=self.device)
        p_acts      = torch.empty((num_steps, P, self.env.action_dim), device=self.device)
        p_logps     = torch.empty((num_steps, P), device=self.device)
        p_rews      = torch.empty((num_steps, P), device=self.device)
        p_dones     = torch.empty((num_steps, P), device=self.device) # Predators don't die, but we track epoch end
        p_vals      = torch.empty((num_steps, P), device=self.device)

        for step in range(num_steps):
            if step > 0:
                global_obs_boids = self.env.get_global_state(obs_boids)
                global_obs_preds = self.env.get_global_state(obs_preds)
                
                roll_obs_boids = torch.cat([roll_obs_boids[:, 1:, :], obs_boids.unsqueeze(1)], dim=1)
                roll_global_boids = torch.cat([roll_global_boids[:, 1:, :], global_obs_boids.unsqueeze(1)], dim=1)
                
                roll_obs_preds = torch.cat([roll_obs_preds[:, 1:, :], obs_preds.unsqueeze(1)], dim=1)
                roll_global_preds = torch.cat([roll_global_preds[:, 1:, :], global_obs_preds.unsqueeze(1)], dim=1)

            flat_obs_boids = roll_obs_boids.view(N, -1)
            flat_globs_boids = roll_global_boids.view(N, -1)
            flat_obs_preds = roll_obs_preds.view(P, -1)
            flat_globs_preds = roll_global_preds.view(P, -1)
            
            b_obs[step], b_globs[step] = flat_obs_boids, flat_globs_boids
            p_obs[step], p_globs[step] = flat_obs_preds, flat_globs_preds

            with torch.no_grad():
                # Boid Actions
                b_action, b_logp, _, b_nv = self.boid_brain.get_action_and_value(flat_obs_boids, flat_globs_boids)
                b_value = self.boid_value_norm.denormalize(b_nv)
                
                # Predator Actions
                p_action, p_logp, _, p_nv = self.pred_brain.get_action_and_value(flat_obs_preds, flat_globs_preds)
                p_value = self.pred_value_norm.denormalize(p_nv)

            next_obs_boids, next_obs_preds, rewards_boids, rewards_preds, dones_boids = self.env.step(
                boid_actions=b_action, predator_actions=p_action
            )

            b_acts[step], b_logps[step], b_rews[step], b_dones[step], b_vals[step] = b_action, b_logp, rewards_boids, dones_boids.float(), b_value.flatten()
            
            # Predators are immortal currently, so dones is False unless truncated
            p_dones_val = torch.zeros(P, device=self.device)
            if torch.all(dones_boids):
                p_dones_val.fill_(1.0) # Episode over for predators too if all boids dead/truncated
                
            p_acts[step], p_logps[step], p_rews[step], p_dones[step], p_vals[step] = p_action, p_logp, rewards_preds, p_dones_val, p_value.flatten()
            
            # Flush dead boid history
            if dones_boids.any():
                dead_mask = dones_boids.bool()
                roll_obs_boids[dead_mask] = next_obs_boids[dead_mask].unsqueeze(1).repeat(1, self.stacked_frames, 1)

            obs_boids, obs_preds = next_obs_boids, next_obs_preds

        # Final Bootstrapping
        global_obs_boids = self.env.get_global_state(obs_boids)
        roll_obs_boids = torch.cat([roll_obs_boids[:, 1:, :], obs_boids.unsqueeze(1)], dim=1)
        roll_global_boids = torch.cat([roll_global_boids[:, 1:, :], global_obs_boids.unsqueeze(1)], dim=1)
        
        global_obs_preds = self.env.get_global_state(obs_preds)
        roll_obs_preds = torch.cat([roll_obs_preds[:, 1:, :], obs_preds.unsqueeze(1)], dim=1)
        roll_global_preds = torch.cat([roll_global_preds[:, 1:, :], global_obs_preds.unsqueeze(1)], dim=1)

        boid_rollouts = {
            "obs": b_obs, "global_obs": b_globs, "actions": b_acts, "logprobs": b_logps,
            "rewards": b_rews, "dones": b_dones, "values": b_vals,
            "final_obs": roll_obs_boids.view(N, -1), "final_global_obs": roll_global_boids.view(N, -1)
        }
        
        pred_rollouts = {
            "obs": p_obs, "global_obs": p_globs, "actions": p_acts, "logprobs": p_logps,
            "rewards": p_rews, "dones": p_dones, "values": p_vals,
            "final_obs": roll_obs_preds.view(P, -1), "final_global_obs": roll_global_preds.view(P, -1)
        }

        return boid_rollouts, pred_rollouts

    def compute_advantages(self, rollouts, brain, value_normalizer):
        rewards = rollouts["rewards"]
        dones   = rollouts["dones"]
        values  = rollouts["values"]

        with torch.no_grad():
            _, _, _, norm_next_value = brain.get_action_and_value(rollouts["final_obs"], rollouts["final_global_obs"])
            next_value = value_normalizer.denormalize(norm_next_value).flatten()

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        num_steps = len(rewards)
        
        for t in reversed(range(num_steps)):
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value if t == num_steps - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def train_population(self, rollouts, brain, optimizer, value_normalizer):
        b_obs = rollouts["obs"]
        b_global_obs = rollouts["global_obs"]
        b_actions = rollouts["actions"]
        b_logprobs = rollouts["logprobs"]
        b_values = rollouts["values"]
        b_dones = rollouts["dones"]
        
        b_advantages, b_returns = self.compute_advantages(rollouts, brain, value_normalizer)
        
        num_steps, num_agents = b_dones.shape
        valid_mask = torch.ones((num_steps, num_agents), dtype=torch.bool, device=self.device)
        valid_mask[1:] = ~b_dones[:-1].bool()
        
        mb_obs = b_obs[valid_mask]
        mb_global_obs = b_global_obs[valid_mask]
        mb_actions = b_actions[valid_mask]
        mb_logprobs = b_logprobs[valid_mask]
        mb_advantages = b_advantages[valid_mask]
        mb_returns = b_returns[valid_mask]
        
        # Explained variance
        y_pred = b_values[valid_mask].cpu().numpy()
        y_true = mb_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        value_normalizer.update(mb_returns)
        mb_returns_norm = value_normalizer.normalize(mb_returns)
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        
        batch_size_total = mb_obs.shape[0]
        inds = np.arange(batch_size_total)
        batch_size = min(self.batch_size, batch_size_total)
        
        epoch_pg_losses, epoch_v_losses, epoch_entropies = [], [], []
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)
            approx_kl_divs = []
            
            for start in range(0, batch_size_total, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]
                
                mini_obs = mb_obs[mb_inds]
                mini_global_obs = mb_global_obs[mb_inds]
                mini_actions = mb_actions[mb_inds]
                mini_advantages = mb_advantages[mb_inds]
                mini_returns_norm = mb_returns_norm[mb_inds]
                mini_logprobs = mb_logprobs[mb_inds]
                
                _, newlogprob, entropy, newvalue = brain.get_action_and_value(mini_obs, mini_global_obs, mini_actions)
                logratio = newlogprob - mini_logprobs
                ratio = logratio.exp()
                
                pg_loss1 = -mini_advantages * ratio
                pg_loss2 = -mini_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((newvalue.flatten() - mini_returns_norm) ** 2).mean()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                approx_kl_divs.append(approx_kl.item())

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(brain.parameters(), self.max_grad_norm)
                optimizer.step()
                
                epoch_pg_losses.append(pg_loss.item())
                epoch_v_losses.append(v_loss.item())
                epoch_entropies.append(entropy_loss.item())
                
            if np.mean(approx_kl_divs) > self.target_kl:
                break
                
        return np.mean(epoch_pg_losses), np.mean(epoch_v_losses), np.mean(epoch_entropies), mb_returns.mean().item(), explained_var

    def train_step(self, boid_rollouts, pred_rollouts):
        """Asynchronous Co-Evolutionary Optimization Phase"""
        
        # 1. Update Prey
        b_metrics = self.train_population(boid_rollouts, self.boid_brain, self.boid_optimizer, self.boid_value_norm)
        
        # 2. Update Predators independently
        p_metrics = self.train_population(pred_rollouts, self.pred_brain, self.pred_optimizer, self.pred_value_norm)
        
        # Return merged metrics tuple (allows backwards-compatibility with logging script)
        return {"boids": b_metrics, "preds": p_metrics}
