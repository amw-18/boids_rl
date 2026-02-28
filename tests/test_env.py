import torch
import numpy as np
from murmur_rl.envs.murmuration import MurmurationEnv

def test_env_initialization():
    env = MurmurationEnv(num_agents=10, device='cpu')
    
    # Check agents
    assert len(env.possible_agents) == 10
    
    # Check observation space (Biological Features)
    obs_space = env.observation_space('boid_0')
    assert obs_space.shape[0] == 18  # vel(3) + context(8) + threat(4) + bounds(3)
    
    # Check action space
    act_space = env.action_space('boid_0')
    assert act_space.shape[0] == 3
    print("Env Spaces: PASSED")

def test_env_step():
    env = MurmurationEnv(num_agents=10, device='cpu')
    obs, info = env.reset()
    
    assert len(obs) == 10
    assert obs['boid_0'].shape == env.observation_space('boid_0').shape
    
    # Random actions
    actions = {
        agent: env.action_space(agent).sample() for agent in env.agents
    }
    
    obs_next, rewards, terminations, truncations, infos = env.step(actions)
    
    assert len(obs_next) == 10
    assert len(rewards) == 10
    assert not terminations['boid_0']
    
    print("Env Step: PASSED")

if __name__ == "__main__":
    test_env_initialization()
    test_env_step()
