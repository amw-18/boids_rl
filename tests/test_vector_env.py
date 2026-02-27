"""
Correctness + Performance tests for VectorMurmurationEnv.

Usage:
    python tests/test_vector_env.py            # correctness tests
    python tests/test_vector_env.py benchmark  # benchmark old vs new
"""
import sys
import time
import torch
import numpy as np

from murmur_rl.envs.vector_env import VectorMurmurationEnv
from murmur_rl.envs.murmuration import MurmurationEnv
from murmur_rl.agents.starling import StarlingBrain


# ------------------------------------------------------------------ #
# Correctness — compare vectorised env against original PettingZoo env
# ------------------------------------------------------------------ #

def _sync_envs(vec_env, pz_env):
    """Force both envs to share the exact same physics state."""
    pz_env.physics.positions[:] = vec_env.physics.positions.clone()
    pz_env.physics.velocities[:] = vec_env.physics.velocities.clone()
    pz_env.physics.predator_position[:] = vec_env.physics.predator_position.clone()
    pz_env.physics.predator_velocity[:] = vec_env.physics.predator_velocity.clone()
    pz_env.physics.alive_mask[:] = vec_env.physics.alive_mask.clone()


def test_observations_match():
    """Vectorised obs should match PettingZoo obs (within float tolerance)."""
    N = 20
    device = "cpu"

    vec_env = VectorMurmurationEnv(num_agents=N, device=device)
    pz_env = MurmurationEnv(num_agents=N, device=device)

    # Reset vector env and clone state into pz env
    vec_obs = vec_env.reset()
    _sync_envs(vec_env, pz_env)
    pz_env.num_moves = 0
    pz_env.dead_agents = set()

    # Get PettingZoo observations
    pz_obs_dict = pz_env._get_observations()
    pz_obs = np.stack([pz_obs_dict[f"boid_{i}"] for i in range(N)])
    pz_obs_t = torch.tensor(pz_obs, dtype=torch.float32)

    # Compare
    if torch.allclose(vec_obs, pz_obs_t, atol=1e-4):
        print("test_observations_match: PASSED ✓")
    else:
        diff = (vec_obs - pz_obs_t).abs()
        max_diff = diff.max().item()
        worst_col = diff.max(dim=0).values.argmax().item()
        print(f"test_observations_match: FAILED ✗  (max diff={max_diff:.6f} in col {worst_col})")
        # Print per-column max diffs for debugging
        for c in range(vec_obs.shape[1]):
            col_diff = diff[:, c].max().item()
            if col_diff > 1e-4:
                print(f"  col {c}: max diff = {col_diff:.6f}")


def test_rewards_match():
    """Vectorised rewards should match PettingZoo rewards."""
    N = 20
    device = "cpu"

    vec_env = VectorMurmurationEnv(num_agents=N, device=device)
    pz_env = MurmurationEnv(num_agents=N, device=device)

    vec_env.reset()
    _sync_envs(vec_env, pz_env)
    pz_env.num_moves = 0
    pz_env.dead_agents = set()

    # Take a random step
    actions = torch.randn(N, 3).clamp(-1, 1)
    actions_np = actions.numpy()

    # Step vector env
    _, vec_rewards, _ = vec_env.step(actions)

    # Step PettingZoo env with same actions
    pz_actions = {f"boid_{i}": actions_np[i] for i in range(N)}
    pz_env.physics.step(actions=actions * pz_env.physics.max_force)
    pz_env.num_moves += 1
    pz_rewards_dict = pz_env._get_rewards()
    pz_rewards = torch.tensor([pz_rewards_dict[f"boid_{i}"] for i in range(N)])

    if torch.allclose(vec_rewards, pz_rewards, atol=1.0):
        print("test_rewards_match: PASSED ✓")
    else:
        diff = (vec_rewards - pz_rewards).abs()
        max_diff = diff.max().item()
        worst_idx = diff.argmax().item()
        print(f"test_rewards_match: FAILED ✗  (max diff={max_diff:.6f} at agent {worst_idx})")
        print(f"Vec Reward: {vec_rewards[worst_idx]}")
        print(f"PZ Reward: {pz_rewards[worst_idx]}")
        
        # Calculate intermediate diffs for agent worst_idx
        # Z-penalty
        pos = pz_env.physics.positions
        
        # Original XY boundaries
        dist_lo_xy = pos[:, :2]
        dist_hi_xy = pz_env.space_size - pos[:, :2]
        closest_wall_xy = torch.min(dist_lo_xy, dist_hi_xy).min(dim=1).values
        penetration_xy = torch.clamp((pz_env.space_size*0.1) - closest_wall_xy, min=0.0)
        boundary_penalty_xy = -10.0 * (penetration_xy / (pz_env.space_size*0.1))
        
        # New Z-bound penalty (Floor=0, Ceiling=0.85)
        z_pos = pos[:, 2]
        ceiling = pz_env.space_size * 0.85
        dist_lo_z = z_pos
        dist_hi_z = ceiling - z_pos
        
        # Penalize approaching the floor or the lowered ceiling
        closest_wall_z = torch.min(dist_lo_z, dist_hi_z)
        penetration_z = torch.clamp((pz_env.space_size*0.1) - closest_wall_z, min=0.0)
        
        # Make the ceiling penalty much steeper to discourage hiding near predators
        is_ceiling = dist_hi_z < dist_lo_z
        penalty_z_mult = torch.where(is_ceiling, 20.0, 10.0)
        boundary_penalty_z = -penalty_z_mult * (penetration_z / (pz_env.space_size*0.1))
        
        print(f"Vec Boundary Z-Pen agent {worst_idx}: {boundary_penalty_z[worst_idx].item()}")
        print(f"Vec Boundary XY-Pen agent {worst_idx}: {boundary_penalty_xy[worst_idx].item()}")
        
        # Calculate Vec Env Distance Penalties
        vec_pos = vec_env.physics.positions
        dist_matrix = torch.cdist(vec_pos, vec_pos)
        mask_out = torch.eye(N, dtype=torch.bool, device="cpu") | ~vec_env.physics.alive_mask.unsqueeze(0)
        dist_matrix = torch.where(mask_out, float('inf'), dist_matrix)
        collision_count = (dist_matrix < 1.0).sum(dim=1).float()
        
        # Calculate PZ Distance Penalties
        pz_pos = pz_env.physics.positions
        pz_dist_matrix = torch.cdist(pz_pos, pz_pos)
        pz_dist_matrix.fill_diagonal_(float('inf'))
        pz_collision_count = (pz_dist_matrix < 1.0).sum(dim=1).float()
        
        print(f"Vec Collision Penalty agent {worst_idx}: {-2*collision_count[worst_idx].item()}")
        print(f"PZ Collision Penalty agent {worst_idx}: {-2*pz_collision_count[worst_idx].item()}")
        
        print(f"Vec base reward should be: {0.1 + boundary_penalty_xy[worst_idx].item() + boundary_penalty_z[worst_idx].item() - 2*collision_count[worst_idx].item()}")


def test_multi_step():
    """Run vec env for several steps — smoke test for crashes / shape issues."""
    N = 50
    env = VectorMurmurationEnv(num_agents=N, device="cpu")
    obs = env.reset()
    assert obs.shape == (N, 18), f"Bad obs shape: {obs.shape}"

    for _ in range(20):
        actions = torch.randn(N, 3).clamp(-1, 1)
        obs, rewards, dones = env.step(actions)
        assert obs.shape == (N, 18)
        assert rewards.shape == (N,)
        assert dones.shape == (N,)

    print("test_multi_step: PASSED ✓")


def test_dones_persistence():
    """Vectorised env should return True for dones persistently for already dead agents, matching PettingZoo."""
    N = 10
    device = "cpu"

    vec_env = VectorMurmurationEnv(num_agents=N, device=device)
    pz_env = MurmurationEnv(num_agents=N, device=device)

    vec_env.reset()
    pz_env.reset()
    
    # Force agent 0 to die by giving it a position exactly at the predator
    vec_env.physics.positions[0] = vec_env.physics.predator_position[0].clone()
    _sync_envs(vec_env, pz_env)
    
    actions = torch.zeros(N, 3)
    
    # Step 1: agent 0 dies
    _, vec_rewards_1, vec_dones_1 = vec_env.step(actions)
    pz_obs_1, pz_rewards_1, pz_terms_1, pz_truncs_1, _ = pz_env.step({f"boid_{i}": actions[i].numpy() for i in range(N)})
    
    # Step 2: agent 0 is ALREADY dead
    _, vec_rewards_2, vec_dones_2 = vec_env.step(actions)
    pz_obs_2, pz_rewards_2, pz_terms_2, pz_truncs_2, _ = pz_env.step({f"boid_{i}": actions[i].numpy() for i in range(N)})
    
    # Check if vector env matches PettingZoo behavior
    assert pz_terms_1["boid_0"] == True, "PettingZoo agent 0 should die step 1"
    assert pz_terms_2["boid_0"] == True, "PettingZoo agent 0 should still be dead step 2"
    
    assert vec_dones_1[0].item() == True, "Vector env agent 0 should die step 1"
    if vec_dones_2[0].item() != True:
        print(f"test_dones_persistence: FAILED ✗  (vec_dones[0] became {vec_dones_2[0].item()} on step 2, expected True)")
        sys.exit(1)
        
    print("test_dones_persistence: PASSED ✓")

def test_global_state():
    """Verify the MAPPO global state exact structure and shape."""
    N = 20
    device = "cpu"
    env = VectorMurmurationEnv(num_agents=N, device=device)
    env.reset()
    
    global_state = env.get_global_state()
    
    # Expected dimensions
    # N * 3 (pos) + N * 3 (vel) + N * 3 (up) + P * 3 (pred_pos) + P * 3 (pred_vel) + N (alive)
    expected_dim = (N * 3) + (N * 3) + (N * 3) + (env.num_predators * 3) + (env.num_predators * 3) + N
    
    # It must expand identically across the N agents
    assert global_state.shape == (N, expected_dim), f"Expected shape {(N, expected_dim)}, got {global_state.shape}"
    
    # Verify that agent 0's global state is identical to agent 1's global state
    assert torch.allclose(global_state[0], global_state[1]), "Global state must be identical across all agents"
    assert env.global_obs_dim == expected_dim, "Internal environment tracker must match expected dimensions."
    
    print("test_global_state: PASSED ✓")

def test_centralized_critic():
    """Verify that the CTDE MAPPO network feeds correctly without dimension mismatch."""
    N = 20
    device = "cpu"
    env = VectorMurmurationEnv(num_agents=N, device=device)
    obs = env.reset()
    global_obs = env.get_global_state()
    
    brain = StarlingBrain(obs_dim=env.obs_dim, global_obs_dim=env.global_obs_dim, action_dim=env.action_dim, hidden_size=64)
    
    # Forward pass checking action, logprob, entropy, and global value
    actions, log_probs, entropies, values = brain.get_action_and_value(obs, global_obs)
    
    assert actions.shape == (N, 3), "Actor output mismatch"
    assert log_probs.shape == (N,), "Logprob shape mismatch"
    assert values.shape == (N, 1), "Centralized Critic output mismatch"
    
    # Value function should not fail when training step tries to flatten it
    values.flatten()
    
    print("test_centralized_critic: PASSED ✓")

# ------------------------------------------------------------------ #
# Benchmark — old vs new
# ------------------------------------------------------------------ #

def benchmark():
    N = 250
    STEPS = 100
    device = "cpu"

    # --- New vectorised env ---
    vec_env = VectorMurmurationEnv(num_agents=N, device=device)
    vec_env.reset()
    # warmup
    for _ in range(5):
        vec_env.step(torch.randn(N, 3, device=device).clamp(-1, 1))
    vec_env.reset()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        vec_env.step(torch.randn(N, 3, device=device).clamp(-1, 1))
    vec_time = time.perf_counter() - t0

    # --- Old PettingZoo env ---
    pz_env = MurmurationEnv(num_agents=N, device=device)
    pz_obs, _ = pz_env.reset()
    # warmup
    for _ in range(5):
        pz_actions = {f"boid_{i}": np.random.randn(3).astype(np.float32) for i in range(N)}
        pz_env.step(pz_actions)
    pz_obs, _ = pz_env.reset()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        pz_actions = {f"boid_{i}": np.random.randn(3).astype(np.float32) for i in range(N)}
        pz_env.step(pz_actions)
    pz_time = time.perf_counter() - t0

    print(f"\n{'='*50}")
    print(f"Benchmark: {N} agents × {STEPS} steps on {device}")
    print(f"{'='*50}")
    print(f"  Old (PettingZoo):  {pz_time:.3f}s  ({STEPS/pz_time:.1f} steps/s)")
    print(f"  New (Vectorised):  {vec_time:.3f}s  ({STEPS/vec_time:.1f} steps/s)")
    print(f"  Speedup:           {pz_time/vec_time:.1f}×")
    print(f"{'='*50}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        test_observations_match()
        test_rewards_match()
        test_multi_step()
        test_dones_persistence()
        test_global_state()
        test_centralized_critic()
        print("\nAll correctness tests done.")
