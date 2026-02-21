import torch
import pytest
from murmur_rl.training.evolution import ERLPopulation

class MockEnv:
    def __init__(self, n_agents=10, obs_dim=16, action_dim=3, device="cpu"):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    def reset(self):
        return torch.zeros((self.n_agents, self.obs_dim), device=self.device)

    def step(self, actions):
        obs = torch.zeros((self.n_agents, self.obs_dim), device=self.device)
        rewards = torch.ones(self.n_agents, device=self.device)
        dones = torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        return obs, rewards, dones

def test_erl_population_mutation():
    device = "cpu"
    pop = ERLPopulation(num_agents=3, obs_dim=16, action_dim=3, hidden_size=16, device=device)
    
    # Check original weights
    orig_weight = pop.agents[1].actor_mean[0].weight.clone()
    
    # Mutate agent 1
    pop.mutate(agent_idx=1, mutation_power=0.1)
    
    # Verify mutation happened
    new_weight = pop.agents[1].actor_mean[0].weight.clone()
    assert not torch.allclose(orig_weight, new_weight)
    
    # Verify agent 0 did NOT mutate
    agent0_old = pop.agents[0].actor_mean[0].weight.clone()
    pop.mutate(agent_idx=1, mutation_power=0.1)
    agent0_new = pop.agents[0].actor_mean[0].weight.clone()
    assert torch.allclose(agent0_old, agent0_new)

def test_erl_population_evolve():
    device = "cpu"
    pop = ERLPopulation(num_agents=4, obs_dim=16, action_dim=3, hidden_size=16, device=device)
    
    # Manually set fitness scores
    # Agent 0: RL (score 10)
    # Agent 1: Bad (score -50)
    # Agent 2: Elite GA (score 100)
    # Agent 3: Average GA (score 0)
    scores = [10.0, -50.0, 100.0, 0.0]
    
    # Agent 2 is elite. Agent 1 and 3 should be replaced by mutated versions of 0 and 2 
    # (actually elites are ranked from all). Rank: 2 (100), 0 (10), 3 (0), 1 (-50)
    # If elite_count=2, elites are [2, 0].
    # Agent 0 is protected. Target indices for replacement: 1, 3.
    # They should be replaced by clones of [2, 0].
    
    # Record properties to verify clone
    orig_elite_weight = pop.agents[2].actor_mean[0].weight.clone()
    
    pop.evolve(scores, elite_count=2, mutation_power=0.0) # 0 power to test exact clone
    
    # Agent 1 should be a clone of agent 2 (first replacement)
    new_target1_weight = pop.agents[1].actor_mean[0].weight.clone()
    assert torch.allclose(orig_elite_weight, new_target1_weight)

def test_erl_batched_evaluation():
    device = "cpu"
    pop = ERLPopulation(num_agents=5, obs_dim=16, action_dim=3, hidden_size=16, device=device)
    env = MockEnv(n_agents=10, device=device)
    
    fitness = pop.evaluate(env, max_steps=10)
    
    # 10 agents divided by 5 pop = chunk size 2.
    # Each agent controls 2 boids for 10 steps.
    # MockEnv rewards are 1.0 per boid. So 2 * 10 = 20.0 sum per agent.
    assert len(fitness) == 5
    for fit in fitness:
         assert fit.item() == 20.0
