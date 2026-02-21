import torch
import torch.nn as nn
from copy import deepcopy
from murmur_rl.agents.starling import StarlingBrain

class ERLPopulation:
    def __init__(self, num_agents, obs_dim, action_dim, hidden_size, device):
        self.num_agents = num_agents
        self.device = device
        
        # Initialize population of StarlingBrains
        self.agents = []
        for _ in range(num_agents):
            brain = StarlingBrain(obs_dim=obs_dim, action_dim=action_dim, hidden_size=hidden_size)
            brain = brain.to(device)
            self.agents.append(brain)
            
        # Agent 0 is the RL agent. Agents 1..N-1 are GA agents.
        
    def mutate(self, agent_idx, mutation_power=0.01):
        """Applies Gaussian noise to the weights of a specific agent on-device."""
        agent = self.agents[agent_idx]
        with torch.no_grad():
            for param in agent.parameters():
                if param.requires_grad:
                    # In-place addition of noise to avoid syncs
                    noise = torch.randn_like(param, device=self.device) * mutation_power
                    param.add_(noise)

    def clone_agent(self, src_idx, dest_idx):
        """Hard copies the weights from src agent to dest agent."""
        src_state = self.agents[src_idx].state_dict()
        self.agents[dest_idx].load_state_dict(src_state)

    def evolve(self, fitness_scores, elite_count=1, mutation_power=0.01):
        """
        fitness_scores: list or tensor of length num_agents
        Selects top `elite_count` agents, replaces the rest by cloning elites and mutating.
        Agent 0 (RL agent) is always preserved as its own entity but it competes.
        """
        assert len(fitness_scores) == self.num_agents
        
        # If fitness_scores is a tensor, convert to list for easy sorting logic
        if isinstance(fitness_scores, torch.Tensor):
            fitness_scores = fitness_scores.tolist()
            
        # Create a list of (fitness, idx)
        ranked = sorted([(fit, idx) for idx, fit in enumerate(fitness_scores)], reverse=True)
        
        # Determine the elite indices
        elite_indices = [idx for (_, idx) in ranked[:elite_count]]
        
        # Agent 0 (RL agent) is implicitly protected by the RL training loop.
        # But for GA replacement, target all non-elite GA agents (idx != 0 and idx not in elite)
        target_indices = [i for i in range(1, self.num_agents) if i not in elite_indices]
        
        if not target_indices:
            return  # No agents to replace

        # Replace non-elites with mutated copies of elites
        for i, target_idx in enumerate(target_indices):
            # Select an elite to copy from (modulo arithmetic to cycle through elites)
            parent_idx = elite_indices[i % len(elite_indices)]
            self.clone_agent(parent_idx, target_idx)
            self.mutate(target_idx, mutation_power=mutation_power)

    def sync_from_ga(self, fitness_scores, rl_idx=0, threshold=1.2):
        """
        If the best GA agent significantly outperforms the RL agent (by `threshold` factor),
        copy its weights into the RL agent.
        """
        rl_fitness = fitness_scores[rl_idx]
        best_ga_idx = rl_idx
        best_ga_fitness = rl_fitness
        
        for i in range(self.num_agents):
            if i != rl_idx and fitness_scores[i] > best_ga_fitness:
                best_ga_idx = i
                best_ga_fitness = fitness_scores[i]
                
        # If best GA is significantly better than RL, sync
        # Note: if fitness is negative, logic changes. Usually fitness > 0 for sync
        if best_ga_fitness > 0 and rl_fitness > 0:
            if best_ga_fitness > rl_fitness * threshold:
                self.clone_agent(best_ga_idx, rl_idx)
                return True, best_ga_idx
        elif best_ga_fitness < 0 and rl_fitness < 0:
            # For negative scores, a smaller absolute value is better
            if abs(best_ga_fitness) < abs(rl_fitness) / threshold:
                 self.clone_agent(best_ga_idx, rl_idx)
                 return True, best_ga_idx
                 
        # Absolute difference sync as fallback
        if best_ga_fitness > rl_fitness + abs(rl_fitness)*(threshold-1.0):
            self.clone_agent(best_ga_idx, rl_idx)
            return True, best_ga_idx
            
        return False, -1

    def evaluate(self, env, max_steps=100):
        """
        Fast batched evaluation of all brains in the population using a single environment batch.
        Slices the `num_agents` dimension equally.
        """
        obs = env.reset()
        N = env.n_agents
        chunk_size = N // self.num_agents
        
        fitness = torch.zeros(self.num_agents, device=self.device)
        actions = torch.empty((N, 3), device=self.device)
        
        for step in range(max_steps):
            with torch.no_grad():
                for i in range(self.num_agents):
                    start = i * chunk_size
                    end = (i+1) * chunk_size if i < self.num_agents - 1 else N
                    
                    sub_obs = obs[start:end]
                    action, _, _, _ = self.agents[i].get_action_and_value(sub_obs)
                    actions[start:end] = action
                    
            obs, rewards, dones = env.step(actions)
            
            # Accumulate fitness
            for i in range(self.num_agents):
                start = i * chunk_size
                end = (i+1) * chunk_size if i < self.num_agents - 1 else N
                fitness[i] += rewards[start:end].sum()
                
        return fitness

