import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) loop for Parameter-Shared Multi-Agent RL.
    Every starling shares the same network weights, meaning we pool their 
    experiences together to train the central brain.
    """
    def __init__(
        self,
        env,
        brain: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.015,
        update_epochs: int = 4,
        batch_size: int = 64
    ):
        self.env = env
        self.brain = brain.to(device)
        self.optimizer = Adam(self.brain.parameters(), lr=lr, eps=1e-5)
        self.device = device
        
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

    def collect_rollouts(self, num_steps=200):
        """
        Run the environment for `num_steps`, collecting observations, actions, 
        log_probs, rewards, and values for all active agents.
        """
        obs, _ = self.env.reset()
        
        # Buffers
        b_obs, b_actions, b_logprobs = [], [], []
        b_rewards, b_dones, b_values = [], [], []
        
        for step in range(num_steps):
            # Dict mapping agent to obs
            agent_keys = list(obs.keys())
            if not agent_keys:
                break
                
            # Convert to bulk tensor
            obs_tensor = torch.tensor(np.array(list(obs.values())), dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                action, logprob, _, value = self.brain.get_action_and_value(obs_tensor)
            
            # Step env
            action_np = action.cpu().numpy()
            actions_dict = {k: action_np[i] for i, k in enumerate(agent_keys)}
            
            next_obs, rewards, term, trunc, _ = self.env.step(actions_dict)
            dones = {k: term[k] or trunc[k] for k in term}
            
            # Store data
            b_obs.append(obs_tensor)
            b_actions.append(action)
            b_logprobs.append(logprob)
            
            reward_tensor = torch.tensor(np.array(list(rewards.values())), dtype=torch.float32, device=self.device)
            done_tensor = torch.tensor(np.array(list(dones.values())), dtype=torch.float32, device=self.device)
            
            b_rewards.append(reward_tensor)
            b_dones.append(done_tensor)
            b_values.append(value.flatten())
            
            obs = next_obs
            
        # Return bulk tensors (Num_steps x Num_agents, Dim)
        # Assuming num_agents stays mostly constant or we pad/mask (PettingZoo can drop agents)
        # For simplicity in this env, no agents die, they just truncate at 500 steps.
        return {
            "obs": torch.cat(b_obs),
            "actions": torch.cat(b_actions),
            "logprobs": torch.cat(b_logprobs),
            "rewards": torch.stack(b_rewards), # keeps steps separate for GAE
            "dones": torch.stack(b_dones),
            "values": torch.stack(b_values),
            "final_obs": next_obs
        }

    def compute_advantages(self, rollouts):
        """Generalized Advantage Estimation (GAE) across all agents simultaneously."""
        rewards = rollouts["rewards"]
        dones = rollouts["dones"]
        values = rollouts["values"]
        
        # Bootstrap value for next step
        final_obs = rollouts["final_obs"]
        if final_obs:
            final_tensor = torch.tensor(np.array(list(final_obs.values())), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _, _, _, next_value = self.brain.get_action_and_value(final_tensor)
                next_value = next_value.flatten()
        else:
            next_value = torch.zeros_like(values[-1])
            
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        
        num_steps = len(rewards)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones[t]  # assuming terminal dones have 1
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
        returns = advantages + values
        return advantages.flatten(), returns.flatten()

    def train_step(self, rollouts):
        b_obs = rollouts["obs"]
        b_actions = rollouts["actions"]
        b_logprobs = rollouts["logprobs"]
        
        b_advantages, b_returns = self.compute_advantages(rollouts)
        
        # Flatten all batches: (Steps * Agents, Dim)
        b_values = rollouts["values"].flatten()
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        batch_size_total = b_obs.shape[0]
        inds = np.arange(batch_size_total)
        
        # Check if we have enough data for batch size, otherwise lower it
        batch_size = min(self.batch_size, batch_size_total)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)
            
            for start in range(0, batch_size_total, batch_size):
                end = start + batch_size
                mb_inds = inds[start:end]
                
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_logprobs = b_logprobs[mb_inds]
                
                _, newlogprob, entropy, newvalue = self.brain.get_action_and_value(mb_obs, mb_actions)
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()
                
                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                v_loss = 0.5 * ((newvalue.flatten() - mb_returns) ** 2).mean()
                
                # Entropy Loss
                entropy_loss = entropy.mean()
                
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.brain.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
        return pg_loss.item(), v_loss.item(), b_returns.mean().item()
