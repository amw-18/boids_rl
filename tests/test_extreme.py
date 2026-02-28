import torch
import numpy as np
import sys
sys.path.append('src')

from murmur_rl.envs.vector_env import VectorMurmurationEnv
from murmur_rl.envs.murmuration import MurmurationEnv

def _sync_envs(vec_env, pz_env):
    pz_env.physics.positions = vec_env.physics.positions.clone()
    pz_env.physics.velocities = vec_env.physics.velocities.clone()
    pz_env.physics.alive_mask = vec_env.physics.alive_mask.clone()
    pz_env.physics.up_vectors = vec_env.physics.up_vectors.clone()
    pz_env.physics.predator_position = vec_env.physics.predator_position.clone()
    pz_env.physics.predator_velocity = vec_env.physics.predator_velocity.clone()
    pz_env.physics.predator_time_since_cooldown = vec_env.physics.predator_time_since_cooldown.clone()

def test_extreme_cases():
    N = 10
    device = "cpu"
    
    vec_env = VectorMurmurationEnv(num_agents=N, device=device)
    pz_env = MurmurationEnv(num_agents=N, device=device)
    
    vec_env.reset()
    pz_env.reset()
    
    # Intentionally put agents in extreme positions
    # Agent 0: near wall (x < 1.0)
    # Agent 1: near wall (x > space_size - 1.0)
    # Agent 2 & 3: colliding
    # Agent 4: dead
    # Agent 5 & 6: grouped but no one else around
    pos = torch.rand(N, 3) * 50 + 20
    pos[0] = torch.tensor([0.5, 50.0, 50.0])
    pos[1] = torch.tensor([vec_env.space_size - 0.5, 50.0, 50.0])
    pos[2] = torch.tensor([40.0, 40.0, 40.0])
    pos[3] = torch.tensor([40.0, 40.0, 40.5]) # Collision with 2
    pos[5] = torch.tensor([80.0, 80.0, 80.0])
    pos[6] = torch.tensor([80.0, 80.0, 85.0]) # Social with 5
    
    vec_env.physics.positions = pos
    vec_env.physics.alive_mask[4] = False
    
    _sync_envs(vec_env, pz_env)
    
    # Sync death tracking for agent 4 which we manually killed
    vec_env._dead_mask[4] = True
    pz_env.dead_agents.add("boid_4")
    
    # After forcibly moving agents, we must synchronize the PBRS potentials 
    # so the first step doesn't generate massive shaping rewards.
    _, pz_potentials = pz_env._get_rewards()
    for i in range(N):
        pz_env.last_potential[f"boid_{i}"] = pz_potentials[f"boid_{i}"]
        
    _, _, _, new_pot, _ = vec_env._get_rewards()
    vec_env.last_potential = new_pot.clone()
    
    # Run for 15 steps
    for step in range(15):
        # Step vec env
        actions = torch.randn(N, 3).clamp(-1, 1)
        vec_obs, vec_pred_obs, vec_rewards, vec_pred_rewards, vec_dones = vec_env.step(actions)
        
        # Step PettingZoo env
        pz_actions = {f"boid_{i}": actions[i].numpy() for i in range(N)}
        pz_obs_dict, pz_rewards_dict, pz_terms, pz_truncs, infos = pz_env.step(pz_actions)
        
        # Compare
        for i in range(N):
            agent = f"boid_{i}"
            v_obs = vec_obs[i]
            p_obs = torch.tensor(pz_obs_dict[agent])
            if not torch.allclose(v_obs, p_obs, atol=1e-4):
                print(f"Observation mismatch at step {step}, agent {i}!")
                print("Vec:", v_obs)
                print("PZ:", p_obs)
                diff = (v_obs - p_obs).abs()
                print("Max diff:", diff.max().item())
                sys.exit(1)
                
            v_r = vec_rewards[i].item()
            p_r = pz_rewards_dict.get(agent, 0.0) # Pettingzoo might drop agents?
            
            if abs(v_r - p_r) > 1e-4:
                print(f"Reward mismatch at step {step}, agent {i}! Vec: {v_r}, PZ: {p_r}")
                sys.exit(1)
            
    print("All extreme cases passed over multiple steps!")

if __name__ == "__main__":
    test_extreme_cases()
