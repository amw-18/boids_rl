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
    pz_obs = np.stack([pz_obs_dict[f"starling_{i}"] for i in range(N)])
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
    pz_actions = {f"starling_{i}": actions_np[i] for i in range(N)}
    pz_env.physics.step(actions=actions * pz_env.physics.max_force)
    pz_env.num_moves += 1
    pz_rewards_dict = pz_env._get_rewards()
    pz_rewards = torch.tensor([pz_rewards_dict[f"starling_{i}"] for i in range(N)])

    if torch.allclose(vec_rewards, pz_rewards, atol=1e-4):
        print("test_rewards_match: PASSED ✓")
    else:
        diff = (vec_rewards - pz_rewards).abs()
        max_diff = diff.max().item()
        worst_idx = diff.argmax().item()
        print(f"test_rewards_match: FAILED ✗  (max diff={max_diff:.6f} at agent {worst_idx})")


def test_multi_step():
    """Run vec env for several steps — smoke test for crashes / shape issues."""
    N = 50
    env = VectorMurmurationEnv(num_agents=N, device="cpu")
    obs = env.reset()
    assert obs.shape == (N, 16), f"Bad obs shape: {obs.shape}"

    for _ in range(20):
        actions = torch.randn(N, 3).clamp(-1, 1)
        obs, rewards, dones = env.step(actions)
        assert obs.shape == (N, 16)
        assert rewards.shape == (N,)
        assert dones.shape == (N,)

    print("test_multi_step: PASSED ✓")


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
        pz_actions = {f"starling_{i}": np.random.randn(3).astype(np.float32) for i in range(N)}
        pz_env.step(pz_actions)
    pz_obs, _ = pz_env.reset()

    t0 = time.perf_counter()
    for _ in range(STEPS):
        pz_actions = {f"starling_{i}": np.random.randn(3).astype(np.float32) for i in range(N)}
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
        print("\nAll correctness tests done.")
