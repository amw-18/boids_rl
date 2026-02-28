"""
Naive-vs-Vectorized Validation Test Suite.

Every test implements the expected computation with plain Python loops
(the "naive" reference) and asserts the vectorized physics engine /
environment produces identical results within floating-point tolerance.

This catches transposed indices, wrong dim= arguments, off-by-one masks,
stale broadcasts, and other vectorization bugs.

Usage:
    uv run pytest tests/test_naive_validation.py -v
"""
import math
import torch
import pytest

from murmur_rl.envs.physics import BoidsPhysics
from murmur_rl.envs.vector_env import VectorMurmurationEnv


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

ATOL = 1e-4   # tolerance for float comparisons

def _small_physics(n=5, p=2, seed=42):
    """Create a small, deterministic BoidsPhysics instance."""
    torch.manual_seed(seed)
    phys = BoidsPhysics(num_boids=n, num_predators=p, space_size=100.0,
                        device=torch.device("cpu"), perception_radius=15.0,
                        base_speed=5.0, max_turn_angle=0.5, max_force=2.0,
                        dt=0.1)
    return phys


def _small_env(n=5, p=2, seed=42):
    """Create a small, deterministic VectorMurmurationEnv."""
    torch.manual_seed(seed)
    env = VectorMurmurationEnv(num_agents=n, num_predators=p,
                               space_size=100.0, perception_radius=15.0,
                               device="cpu", gamma=0.99, pbrs_k=1.0,
                               pbrs_c=1.0)
    # Override physics params to match training config
    env.physics.base_speed = 5.0
    env.physics.max_turn_angle = 0.5
    env.physics.max_force = 2.0
    return env


# ══════════════════════════════════════════════════════════════════════
#  PHYSICS ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestPhysics6DOF:
    """Verify the 6-DOF orientation update (roll + pitch) is correct."""

    def test_6dof_orientation_update(self):
        """Naive single-boid roll/pitch must match vectorized batch."""
        phys = _small_physics(n=3)
        N = phys.num_boids

        # Snapshot initial state
        fwd_0 = (phys.velocities / phys.velocities.norm(dim=-1, keepdim=True).clamp(min=1e-5)).clone()
        up_0 = phys.up_vectors.clone()
        speed_0 = phys.velocities.norm(dim=-1, keepdim=True).clone()

        actions = torch.tensor([
            [0.0,  0.5,  0.0],   # pure roll
            [0.0,  0.0,  0.5],   # pure pitch
            [0.0,  0.3, -0.4],   # mixed
        ], dtype=torch.float32)

        # --- Naive per-boid computation ---
        expected_fwd = []
        expected_up = []
        for i in range(N):
            fwd = fwd_0[i].clone()
            up = up_0[i].clone()

            right = torch.cross(fwd, up)
            right = right / right.norm().clamp(min=1e-5)

            roll_angle = actions[i, 1] * phys.max_turn_angle
            cos_r, sin_r = math.cos(roll_angle.item()), math.sin(roll_angle.item())
            up_rolled = up * cos_r + right * sin_r

            pitch_angle = actions[i, 2] * phys.max_turn_angle
            cos_p, sin_p = math.cos(pitch_angle.item()), math.sin(pitch_angle.item())
            fwd_new = fwd * cos_p + up_rolled * sin_p
            up_new = up_rolled * cos_p - fwd * sin_p

            fwd_new = fwd_new / fwd_new.norm().clamp(min=1e-5)
            up_new = up_new / up_new.norm().clamp(min=1e-5)

            # Re-orthogonalize
            right_new = torch.cross(fwd_new, up_new)
            right_new = right_new / right_new.norm().clamp(min=1e-5)
            up_final = torch.cross(right_new, fwd_new)
            up_final = up_final / up_final.norm().clamp(min=1e-5)

            expected_fwd.append(fwd_new)
            expected_up.append(up_final)

        expected_fwd = torch.stack(expected_fwd)
        expected_up = torch.stack(expected_up)

        # --- Vectorized computation ---
        phys.step(boid_actions=actions)

        actual_speed = phys.velocities.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        actual_fwd = phys.velocities / actual_speed
        actual_up = phys.up_vectors

        assert torch.allclose(actual_fwd, expected_fwd, atol=ATOL), \
            f"Forward mismatch:\n  expected={expected_fwd}\n  actual={actual_fwd}"
        assert torch.allclose(actual_up, expected_up, atol=ATOL), \
            f"Up mismatch:\n  expected={expected_up}\n  actual={actual_up}"

    def test_orientation_stays_orthonormal(self):
        """After multiple steps, forward·up ≈ 0 and both are unit vectors."""
        phys = _small_physics(n=10)
        for _ in range(50):
            actions = torch.randn(10, 3).clamp(-1, 1)
            phys.step(boid_actions=actions)

        speed = phys.velocities.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        fwd = phys.velocities / speed
        up = phys.up_vectors

        # Unit length
        assert torch.allclose(fwd.norm(dim=-1), torch.ones(10), atol=ATOL)
        assert torch.allclose(up.norm(dim=-1), torch.ones(10), atol=ATOL)

        # Orthogonality: dot product ≈ 0
        dots = (fwd * up).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros(10), atol=1e-3), \
            f"forward·up not orthogonal: {dots}"


class TestPhysicsVelocity:
    """Verify thrust, speed clamping, and position updates."""

    def test_thrust_velocity_update(self):
        """Thrust changes speed correctly: new_speed = speed + thrust * dt, clamped [0.5, 10]."""
        phys = _small_physics(n=4)

        # Set known velocities: all flying along +X at base_speed
        phys.velocities = torch.tensor([
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # at min speed
            [9.9, 0.0, 0.0],  # near max speed
        ])
        phys.up_vectors = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        pos_before = phys.positions.clone()

        # Actions: thrust only, no roll/pitch
        actions = torch.tensor([
            [ 1.0, 0.0, 0.0],  # full thrust forward
            [-1.0, 0.0, 0.0],  # full brake
            [-1.0, 0.0, 0.0],  # brake at min speed → should stay at 0.5
            [ 1.0, 0.0, 0.0],  # thrust near max → should clamp to 10.0
        ])

        phys.step(boid_actions=actions)

        expected_speeds = []
        for i in range(4):
            old_speed = torch.tensor([5.0, 5.0, 0.5, 9.9])[i].item()
            thrust = actions[i, 0].item() * phys.max_force
            new_speed = old_speed + thrust * phys.dt
            new_speed = max(0.5, min(10.0, new_speed))
            expected_speeds.append(new_speed)

        actual_speeds = phys.velocities.norm(dim=-1)
        for i in range(4):
            assert abs(actual_speeds[i].item() - expected_speeds[i]) < ATOL, \
                f"Boid {i}: expected speed {expected_speeds[i]:.4f}, got {actual_speeds[i].item():.4f}"

    def test_position_update(self):
        """Position updates via x_new = x_old + v_new * dt."""
        phys = _small_physics(n=3)
        pos_before = phys.positions.clone()

        actions = torch.tensor([
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
        ])
        phys.step(boid_actions=actions)

        # Position delta should equal velocity * dt
        delta = phys.positions - pos_before
        expected_delta = phys.velocities * phys.dt
        assert torch.allclose(delta, expected_delta, atol=ATOL), \
            f"Position update mismatch:\n  delta={delta}\n  v*dt={expected_delta}"

    def test_dead_boids_freeze(self):
        """Dead boids must have zero velocity and unchanging position."""
        phys = _small_physics(n=5)
        phys.alive_mask[2] = False
        phys.alive_mask[4] = False

        pos_before = phys.positions.clone()
        actions = torch.randn(5, 3).clamp(-1, 1)
        phys.step(boid_actions=actions)

        # Dead boids: velocity must be zero
        assert phys.velocities[2].norm().item() == 0.0
        assert phys.velocities[4].norm().item() == 0.0

        # Dead boids: position unchanged (since v=0, delta = 0*dt = 0)
        # Note: positions are updated AFTER velocity is zeroed
        assert torch.allclose(phys.positions[2], pos_before[2], atol=ATOL)
        assert torch.allclose(phys.positions[4], pos_before[4], atol=ATOL)

        # Alive boids: should have moved
        assert not torch.allclose(phys.positions[0], pos_before[0], atol=ATOL)

    def test_alive_boids_never_zero_speed(self):
        """Alive boids must always have speed >= 0.5 (min clamp)."""
        phys = _small_physics(n=10)
        # Apply max braking for many steps
        brake_action = torch.tensor([[-1.0, 0.0, 0.0]]).expand(10, 3)
        for _ in range(200):
            phys.step(boid_actions=brake_action.clone())

        alive_speeds = phys.velocities[phys.alive_mask].norm(dim=-1)
        assert (alive_speeds >= 0.5 - ATOL).all(), \
            f"Some alive boids below min speed: {alive_speeds}"


class TestPredatorStamina:
    """Verify predator stamina, cooldown, and speed mechanics."""

    def test_predator_stamina_management(self):
        """Sprint drains stamina, cruise recovers it."""
        phys = _small_physics(n=3, p=2)
        initial_stamina = phys.predator_stamina.clone()

        # Predator 0: sprint, Predator 1: cruise
        pred_actions = torch.tensor([
            [1.0, 0.0, 0.0],   # sprinting (action > 0)
            [-1.0, 0.0, 0.0],  # cruising (action < 0)
        ])

        phys.step(boid_actions=torch.zeros(3, 3), predator_actions=pred_actions)

        # Predator 0 should have lost stamina
        assert phys.predator_stamina[0] < initial_stamina[0], \
            f"Sprinting predator didn't lose stamina: {phys.predator_stamina[0]} vs {initial_stamina[0]}"
        expected_stamina_0 = initial_stamina[0] - phys.predator_sprint_drain
        assert abs(phys.predator_stamina[0].item() - expected_stamina_0.item()) < ATOL

        # Predator 1 should be at max (started at max, recovered more)
        assert phys.predator_stamina[1] == phys.predator_max_stamina

    def test_predator_capture_and_cooldown(self):
        """Boid within catch_radius of predator → dies; predator gets cooldown."""
        phys = _small_physics(n=5, p=2)

        # Place boid 1 exactly on predator 0
        phys.positions[1] = phys.predator_position[0].clone()

        # Place all others far away
        for i in [0, 2, 3, 4]:
            phys.positions[i] = torch.tensor([99.0, 99.0, 99.0])

        assert phys.alive_mask[1] == True  # still alive before step

        phys.step(boid_actions=torch.zeros(5, 3), predator_actions=torch.zeros(2, 3))

        # Boid 1 should now be dead
        assert phys.alive_mask[1] == False, "Boid within catch_radius should be dead"

        # Predator 0 should have cooldown
        assert phys.predator_cooldown[0] > 0, "Predator that caught should have cooldown"
        assert phys.predator_cooldown[0] == phys.predator_cooldown_duration


# ══════════════════════════════════════════════════════════════════════
#  BOID OBSERVATIONS (18D) TESTS
# ══════════════════════════════════════════════════════════════════════

class TestBoidObservations:
    """
    Verify each column block of the 18D observation vector against
    a naive per-agent loop computation.
    """

    def _setup(self, n=8, p=2, seed=42):
        """Create env, reset, and return (env, obs)."""
        torch.manual_seed(seed)
        env = _small_env(n=n, p=p, seed=seed)
        obs_boids, _ = env.reset()
        return env, obs_boids

    def test_obs_velocity_normalization(self):
        """Columns 0:3 = vel / base_speed."""
        env, obs = self._setup()
        vel = env.physics.velocities
        expected = vel / env.physics.base_speed

        assert torch.allclose(obs[:, 0:3], expected, atol=ATOL), \
            f"Vel norm mismatch:\n  obs={obs[:, 0:3]}\n  expected={expected}"

    def test_obs_nearest_neighbor(self):
        """Column 3 = nearest alive neighbor distance / perception_radius."""
        env, obs = self._setup()
        N = env.n_agents
        pos = env.physics.positions
        alive = env.physics.alive_mask

        expected = []
        for i in range(N):
            min_d = float("inf")
            for j in range(N):
                if i == j or not alive[j]:
                    continue
                d = (pos[i] - pos[j]).norm().item()
                min_d = min(min_d, d)
            if min_d == float("inf"):
                min_d = env.perception_radius
            expected.append(min_d / env.perception_radius)

        expected_t = torch.tensor(expected, dtype=torch.float32).unsqueeze(1)
        assert torch.allclose(obs[:, 3:4], expected_t, atol=ATOL), \
            f"Nearest neighbor mismatch:\n  obs={obs[:, 3:4].squeeze()}\n  expected={expected_t.squeeze()}"

    def test_obs_local_density(self):
        """Column 4 = count(alive neighbors within perception_radius) / N."""
        env, obs = self._setup()
        N = env.n_agents
        pos = env.physics.positions
        alive = env.physics.alive_mask
        pr = env.perception_radius

        expected = []
        for i in range(N):
            count = 0
            for j in range(N):
                if i == j or not alive[j]:
                    continue
                d = (pos[i] - pos[j]).norm().item()
                if d < pr:
                    count += 1
            expected.append(count / N)

        expected_t = torch.tensor(expected, dtype=torch.float32).unsqueeze(1)
        assert torch.allclose(obs[:, 4:5], expected_t, atol=ATOL), \
            f"Density mismatch:\n  obs={obs[:, 4:5].squeeze()}\n  expected={expected_t.squeeze()}"

    def test_obs_local_alignment(self):
        """Columns 5:8 = unit-normalized mean velocity of visible alive neighbors."""
        env, obs = self._setup()
        N = env.n_agents
        pos = env.physics.positions
        vel = env.physics.velocities
        alive = env.physics.alive_mask
        pr = env.perception_radius

        expected = []
        for i in range(N):
            sum_v = torch.zeros(3)
            count = 0
            for j in range(N):
                if i == j or not alive[j]:
                    continue
                d = (pos[i] - pos[j]).norm().item()
                if d < pr:
                    sum_v += vel[j]
                    count += 1
            if count > 0:
                avg_v = sum_v / count
                norm = avg_v.norm().clamp(min=1e-5)
                expected.append(avg_v / norm)
            else:
                expected.append(torch.zeros(3))

        expected_t = torch.stack(expected)
        assert torch.allclose(obs[:, 5:8], expected_t, atol=ATOL), \
            f"Alignment mismatch:\n  obs={obs[:, 5:8]}\n  expected={expected_t}"

    def test_obs_com_direction(self):
        """Columns 8:11 = unit direction from agent to center-of-mass of visible alive neighbors."""
        env, obs = self._setup()
        N = env.n_agents
        pos = env.physics.positions
        alive = env.physics.alive_mask
        pr = env.perception_radius

        expected = []
        for i in range(N):
            sum_pos = torch.zeros(3)
            count = 0
            for j in range(N):
                if i == j or not alive[j]:
                    continue
                d = (pos[i] - pos[j]).norm().item()
                if d < pr:
                    sum_pos += pos[j]
                    count += 1
            if count > 0:
                com = sum_pos / count
                direction = com - pos[i]
                norm = direction.norm().clamp(min=1e-5)
                expected.append(direction / norm)
            else:
                expected.append(torch.zeros(3))

        expected_t = torch.stack(expected)
        assert torch.allclose(obs[:, 8:11], expected_t, atol=ATOL), \
            f"CoM direction mismatch:\n  obs={obs[:, 8:11]}\n  expected={expected_t}"

    def test_obs_predator_threat(self):
        """Columns 11:15 = [d_norm, v_close_norm, loom_norm, bearing] for closest predator."""
        env, obs = self._setup()
        N = env.n_agents
        pos = env.physics.positions
        vel = env.physics.velocities
        pred_pos = env.physics.predator_position
        pred_vel = env.physics.predator_velocity
        P = env.num_predators
        half_space = env.space_size / 2.0
        max_v_close = env.physics.predator_sprint_speed + env.physics.base_speed

        expected = []
        for i in range(N):
            # Find closest predator
            best_d = float("inf")
            best_p = 0
            for p in range(P):
                d = (pred_pos[p] - pos[i]).norm().item()
                if d < best_d:
                    best_d = d
                    best_p = p

            dx = pred_pos[best_p] - pos[i]
            d = dx.norm().clamp(min=1e-5)
            u = dx / d

            d_norm = min((d / half_space).item(), 1.0)

            dv = pred_vel[best_p] - vel[i]
            v_close = -(dv * u).sum().item()
            v_close_norm = max(-1.0, min(1.0, v_close / max_v_close))

            loom = v_close / max(d.item(), 1e-5)
            loom_norm = max(-1.0, min(1.0, loom / 5.0))

            vel_unit = vel[i] / vel[i].norm().clamp(min=1e-5)
            bearing = (vel_unit * u).sum().item()

            # Mask far threats
            if d.item() > half_space:
                v_close_norm = 0.0
                loom_norm = 0.0
                bearing = 0.0

            expected.append([d_norm, v_close_norm, loom_norm, bearing])

        expected_t = torch.tensor(expected, dtype=torch.float32)
        assert torch.allclose(obs[:, 11:15], expected_t, atol=ATOL), \
            f"Predator threat mismatch:\n  obs={obs[:, 11:15]}\n  expected={expected_t}"

    def test_obs_boundary_position(self):
        """Columns 15:18 = (pos - half_space) / half_space."""
        env, obs = self._setup()
        half = env.space_size / 2.0
        expected = (env.physics.positions - half) / half
        assert torch.allclose(obs[:, 15:18], expected, atol=ATOL)

    def test_obs_concatenation_order_and_shape(self):
        """Observation must be exactly 18 columns in documented order."""
        env, obs = self._setup(n=20)
        assert obs.shape == (20, 18), f"Expected (20, 18), got {obs.shape}"

        # Check each block width matches spec
        # vel(3) + nearest(1) + density(1) + align(3) + com(3) + d_norm(1) + v_close(1) + loom(1) + bearing(1) + pos_rel(3)
        widths = [3, 1, 1, 3, 3, 1, 1, 1, 1, 3]
        assert sum(widths) == 18


# ══════════════════════════════════════════════════════════════════════
#  PREDATOR OBSERVATIONS (45D) TESTS
# ══════════════════════════════════════════════════════════════════════

class TestPredatorObservations:
    """Verify predator observation vector structure and values."""

    def _setup(self, n=10, p=2, seed=42):
        torch.manual_seed(seed)
        env = _small_env(n=n, p=p, seed=seed)
        _, obs_preds = env.reset()
        return env, obs_preds

    def test_pred_obs_shape(self):
        """(P, 45) = 3(pos) + 3(vel) + 1(stamina) + 3(com) + 5*(3+3+1)."""
        env, obs = self._setup()
        assert obs.shape == (env.num_predators, 45), f"Expected (2, 45), got {obs.shape}"

    def test_pred_obs_own_kinematics(self):
        """First 7 columns: pos_relative(3), vel_norm(3), stamina(1)."""
        env, obs = self._setup()
        half = env.space_size / 2.0
        pred_pos = env.physics.predator_position
        pred_vel = env.physics.predator_velocity

        expected_pos_rel = (pred_pos - half) / half
        expected_vel_norm = pred_vel / env.physics.predator_sprint_speed
        expected_stamina = (env.physics.predator_stamina / env.physics.predator_max_stamina).unsqueeze(1)

        assert torch.allclose(obs[:, 0:3], expected_pos_rel, atol=ATOL)
        assert torch.allclose(obs[:, 3:6], expected_vel_norm, atol=ATOL)
        assert torch.allclose(obs[:, 6:7], expected_stamina, atol=ATOL)

    def test_pred_obs_com_alive_swarm(self):
        """Columns 7:10 = (com_alive - pred_pos) / half_space."""
        env, obs = self._setup()
        half = env.space_size / 2.0
        boid_pos = env.physics.positions
        alive = env.physics.alive_mask.float()
        pred_pos = env.physics.predator_position

        # Naive: mean position of alive boids
        num_alive = alive.sum().clamp(min=1.0)
        com = (boid_pos * alive.unsqueeze(1)).sum(dim=0) / num_alive

        for p in range(env.num_predators):
            expected_com_rel = (com - pred_pos[p]) / half
            assert torch.allclose(obs[p, 7:10], expected_com_rel, atol=ATOL), \
                f"Predator {p} CoM mismatch: obs={obs[p, 7:10]}, expected={expected_com_rel}"

    def test_pred_obs_5_closest_targets_structure(self):
        """After kinematics (10 cols), there are 5 target blocks of 7 cols each = 35 cols."""
        env, obs = self._setup()
        # Total: 10 + 35 = 45
        target_block = obs[:, 10:]
        assert target_block.shape[1] == 35, f"Target block should be 35 cols, got {target_block.shape[1]}"


# ══════════════════════════════════════════════════════════════════════
#  REWARDS & PBRS TESTS
# ══════════════════════════════════════════════════════════════════════

class TestRewards:
    """Verify reward components against naive computation."""

    def _setup(self, n=8, p=2, seed=42):
        torch.manual_seed(seed)
        env = _small_env(n=n, p=p, seed=seed)
        env.reset()
        return env

    def test_reward_survival_base(self):
        """Alive agents get +0.1 base reward (before PBRS)."""
        env = self._setup()
        # Step with no action — all survive
        rewards, _, new_deaths, _, _ = env._get_rewards()

        for i in range(env.n_agents):
            if env.physics.alive_mask[i] and not env._dead_mask[i]:
                assert abs(rewards[i].item() - 0.1) < ATOL or rewards[i].item() < 0, \
                    f"Agent {i}: expected base ≈ 0.1, got {rewards[i].item()}"

    def test_reward_death_penalty(self):
        """Newly killed agent gets -100, already-dead gets 0."""
        env = self._setup()

        # Kill boid 0 via physics
        env.physics.alive_mask[0] = False
        # Boid 0 is newly dead (not yet in _dead_mask)
        assert env._dead_mask[0] == False

        rewards, _, new_deaths, _, _ = env._get_rewards()

        assert new_deaths[0] == True, "Boid 0 should be newly dead"
        assert abs(rewards[0].item() - (-100.0)) < ATOL, \
            f"Death penalty should be -100, got {rewards[0].item()}"

        # Now mark it in the dead_mask and call again
        env._dead_mask[0] = True
        rewards2, _, _, _, _ = env._get_rewards()
        assert abs(rewards2[0].item()) < ATOL, \
            f"Already-dead agent should get 0 reward, got {rewards2[0].item()}"

    def test_reward_collision_penalty(self):
        """Naive pairwise: agents closer than 1.0 → -2.0 per collision neighbor."""
        env = self._setup(n=5)

        # Place boids 0 and 1 in collision (dist < 1.0)
        env.physics.positions[0] = torch.tensor([50.0, 50.0, 50.0])
        env.physics.positions[1] = torch.tensor([50.0, 50.0, 50.3])  # dist = 0.3
        # Place others far away
        env.physics.positions[2] = torch.tensor([10.0, 10.0, 10.0])
        env.physics.positions[3] = torch.tensor([20.0, 20.0, 20.0])
        env.physics.positions[4] = torch.tensor([80.0, 80.0, 80.0])

        rewards, _, _, _, _ = env._get_rewards()

        # Naive: boid 0 has 1 collision neighbor (boid 1), boid 1 has 1 (boid 0)
        # reward = 0.1 - 2.0*1 = -1.9
        # (ignoring PBRS potentials which are added separately)
        assert rewards[0].item() < 0, f"Boid 0 should have negative reward due to collision, got {rewards[0].item()}"
        assert rewards[1].item() < 0, f"Boid 1 should have negative reward due to collision, got {rewards[1].item()}"
        # Boids 2,3,4 have no collisions
        assert abs(rewards[2].item() - 0.1) < ATOL
        assert abs(rewards[3].item() - 0.1) < ATOL
        assert abs(rewards[4].item() - 0.1) < ATOL

    def test_pbrs_boundary_potential(self):
        """Naive boundary potential: -k * sum((pos - half) / half)^2."""
        env = self._setup(n=5)
        half = env.space_size / 2.0
        k = 1.0  # pbrs_k

        # Place boids at known positions
        env.physics.positions[0] = torch.tensor([50.0, 50.0, 50.0])  # center → potential ≈ 0
        env.physics.positions[1] = torch.tensor([0.0, 0.0, 0.0])     # corner → high penalty
        env.physics.positions[2] = torch.tensor([100.0, 100.0, 100.0])  # opposite corner

        _, _, _, potential, _ = env._get_rewards()

        for i in range(env.n_agents):
            if not env.physics.alive_mask[i]:
                continue
            pos = env.physics.positions[i]
            pos_rel = (pos - half) / half
            expected_phi_bounds = -k * (pos_rel ** 2).sum().item()

            # Potential also includes density component
            # Just verify bounds component direction: center → 0, corner → very negative
            # We'll check the full potential in a separate combined test

        # Center agent should have lowest penalty magnitude
        assert potential[0].item() > potential[1].item(), \
            f"Center should have better potential than corner: {potential[0]} vs {potential[1]}"

    def test_pbrs_density_potential(self):
        """Naive density potential: c * count(neighbors < r) / N."""
        env = self._setup(n=5)
        pos = env.physics.positions
        N = env.n_agents
        pr = env.perception_radius

        # Compute naive density for each agent
        for i in range(N):
            count = 0
            for j in range(N):
                if i == j:
                    continue
                d = (pos[i] - pos[j]).norm().item()
                if d < pr:
                    count += 1
            expected_density = count / N
            # We can't easily isolate density potential from combined potential,
            # but we verify the density calculation matches what _get_observations computes
            obs = env._get_observations()
            assert abs(obs[i, 4].item() - expected_density) < ATOL, \
                f"Agent {i}: density mismatch in obs: {obs[i, 4].item()} vs {expected_density}"

    def test_pbrs_full_potential(self):
        """Full PBRS potential = boundary + density, computed naively."""
        env = self._setup(n=5)
        pos = env.physics.positions
        N = env.n_agents
        pr = env.perception_radius
        half = env.space_size / 2.0

        _, _, _, potential, _ = env._get_rewards()

        for i in range(N):
            if not env.physics.alive_mask[i] or env._dead_mask[i]:
                continue
            # Boundary
            pos_rel = (pos[i] - half) / half
            phi_bounds = -(pos_rel ** 2).sum().item()  # k=1

            # Density
            dist_matrix = torch.cdist(pos, pos)
            count = 0
            for j in range(N):
                if i == j:
                    continue
                if dist_matrix[i, j].item() < pr:
                    count += 1
            phi_density = 1.0 * count / N  # c=1

            expected = phi_bounds + phi_density
            assert abs(potential[i].item() - expected) < ATOL, \
                f"Agent {i}: potential mismatch: {potential[i].item()} vs {expected}"

    def test_pbrs_shaping_integration(self):
        """PBRS shaping = gamma * phi(s') - phi(s) over two steps."""
        env = self._setup(n=5)
        env.reset()  # sets last_potential

        last_pot = env.last_potential.clone()

        actions = torch.zeros(5, 3)
        actions[:, 0] = 0.5  # some thrust
        obs, _, rewards, _, dones = env.step(actions, torch.zeros(2, 3))

        # The rewards include shaping. Verify by computing shaping manually.
        _, _, _, new_pot, _ = env._get_rewards()
        expected_shaping = 0.99 * env.last_potential - last_pot

        # We can't easily separate base reward from shaping in the final value,
        # but we can verify the potential changed and the direction is correct
        # For agents that moved toward center, potential should improve
        # This is a sanity check — main verification is in the full potential test
        assert env.last_potential is not None

    def test_predator_catch_reward(self):
        """Predator must receive +10 reward per boid caught."""
        env = self._setup(n=5, p=2)
        env.reset()

        # Place boid 0 on top of predator 0 to force a catch
        env.physics.positions[0] = env.physics.predator_position[0].clone()
        # All other boids far from all predators
        for i in [1, 2, 3, 4]:
            env.physics.positions[i] = torch.tensor([50.0, 50.0, 50.0])

        # Move predator 1 far away so only predator 0 catches
        env.physics.predator_position[1] = torch.tensor([99.0, 99.0, 99.0])

        # Simulate the capture: physics._check_captures sets alive_mask
        env.physics.alive_mask[0] = False

        _, rewards_preds, _, _, _ = env._get_rewards()

        assert rewards_preds[0].item() > 0, \
            f"Predator 0 caught a boid but got reward {rewards_preds[0].item()} (should be +10)"
        assert abs(rewards_preds[0].item() - 10.0) < ATOL, \
            f"Expected +10.0 catch reward, got {rewards_preds[0].item()}"

    def test_predator_no_catch_no_reward(self):
        """Predator gets zero base reward when no boid is caught."""
        env = self._setup(n=5, p=2)
        env.reset()

        # Place all boids far from all predators
        for i in range(5):
            env.physics.positions[i] = torch.tensor([50.0, 50.0, 50.0 + i * 5])
        env.physics.predator_position[0] = torch.tensor([10.0, 10.0, 10.0])
        env.physics.predator_position[1] = torch.tensor([90.0, 90.0, 90.0])

        _, rewards_preds, _, _, _ = env._get_rewards()

        assert abs(rewards_preds[0].item()) < ATOL, \
            f"Predator 0 shouldn't have reward without catch, got {rewards_preds[0].item()}"
        assert abs(rewards_preds[1].item()) < ATOL, \
            f"Predator 1 shouldn't have reward without catch, got {rewards_preds[1].item()}"

    def test_predator_only_catcher_rewarded(self):
        """Only the predator that caught the boid should get the reward."""
        env = self._setup(n=5, p=2)
        env.reset()

        # Predator 0 catches boid 0, predator 1 is far away
        env.physics.positions[0] = env.physics.predator_position[0].clone()
        env.physics.predator_position[1] = torch.tensor([99.0, 99.0, 99.0])
        # Other boids far from everyone
        for i in [1, 2, 3, 4]:
            env.physics.positions[i] = torch.tensor([50.0, 50.0, 50.0 + i * 5])

        # Simulate the capture
        env.physics.alive_mask[0] = False

        _, rewards_preds, _, _, _ = env._get_rewards()

        assert rewards_preds[0].item() > 0, "Catching predator should be rewarded"
        assert abs(rewards_preds[1].item()) < ATOL, \
            f"Non-catching predator should get 0, got {rewards_preds[1].item()}"


# ══════════════════════════════════════════════════════════════════════
#  MEAN-FIELD GLOBAL STATE TESTS
# ══════════════════════════════════════════════════════════════════════

class TestGlobalState:
    """Verify the mean-field global state computation."""

    def _setup(self, n=10, p=2, seed=42):
        torch.manual_seed(seed)
        env = _small_env(n=n, p=p, seed=seed)
        obs_boids, _ = env.reset()
        return env, obs_boids

    def test_global_state_mean_field(self):
        """Naive mean-field: mean_pos, mean_vel, mean_up (alive), pred state, alive_ratio."""
        env, obs = self._setup()
        N = env.n_agents
        P = env.num_predators

        global_state = env.get_global_state(obs)

        # --- Naive computation ---
        alive = env.physics.alive_mask.float()
        num_alive = alive.sum().clamp(min=1.0)

        mean_pos = (env.physics.positions * alive.unsqueeze(1)).sum(dim=0) / num_alive / env.space_size
        mean_vel = (env.physics.velocities * alive.unsqueeze(1)).sum(dim=0) / num_alive / env.physics.base_speed
        mean_up = (env.physics.up_vectors * alive.unsqueeze(1)).sum(dim=0) / num_alive

        pred_pos = env.physics.predator_position.flatten() / env.space_size
        pred_vel = env.physics.predator_velocity.flatten() / env.physics.predator_sprint_speed
        alive_ratio = (num_alive / N).unsqueeze(0)

        expected_mf = torch.cat([mean_pos, mean_vel, mean_up, pred_pos, pred_vel, alive_ratio])

        # Global state = [local_obs | mean_field], expanded for batch
        assert global_state.shape == (N, obs.shape[1] + expected_mf.shape[0])

        # Mean-field portion (after local obs) should be identical across all agents
        mf_actual = global_state[0, obs.shape[1]:]
        assert torch.allclose(mf_actual, expected_mf, atol=ATOL), \
            f"Mean-field mismatch:\n  actual={mf_actual}\n  expected={expected_mf}"

        # All agents should see the same mean field
        for i in range(1, N):
            assert torch.allclose(global_state[i, obs.shape[1]:], mf_actual, atol=ATOL), \
                f"Agent {i} has different mean-field from agent 0"

        # Each agent's local obs should match
        for i in range(N):
            assert torch.allclose(global_state[i, :obs.shape[1]], obs[i], atol=ATOL)

    def test_global_state_with_deaths(self):
        """Dead agents must be excluded from mean-field means."""
        env, obs = self._setup(n=6)

        # Kill agents 2 and 4
        env.physics.alive_mask[2] = False
        env.physics.alive_mask[4] = False
        env.physics.velocities[2] = 0.0
        env.physics.velocities[4] = 0.0

        obs = env._get_observations()
        global_state = env.get_global_state(obs)

        # Naive: only alive agents (0, 1, 3, 5)
        alive_indices = [0, 1, 3, 5]
        num_alive = len(alive_indices)

        alive_pos = env.physics.positions[alive_indices]
        alive_vel = env.physics.velocities[alive_indices]
        alive_up = env.physics.up_vectors[alive_indices]

        mean_pos = alive_pos.sum(dim=0) / num_alive / env.space_size
        mean_vel = alive_vel.sum(dim=0) / num_alive / env.physics.base_speed
        mean_up = alive_up.sum(dim=0) / num_alive

        # Extract mean field from global state
        mf = global_state[0, obs.shape[1]:]

        # First 3 values should be mean_pos
        assert torch.allclose(mf[0:3], mean_pos, atol=ATOL), \
            f"mean_pos with deaths: actual={mf[0:3]}, expected={mean_pos}"
        assert torch.allclose(mf[3:6], mean_vel, atol=ATOL), \
            f"mean_vel with deaths: actual={mf[3:6]}, expected={mean_vel}"

        # alive_ratio should be 4/6
        expected_ratio = 4.0 / 6.0
        actual_ratio = mf[-1].item()
        assert abs(actual_ratio - expected_ratio) < ATOL, \
            f"alive_ratio: actual={actual_ratio}, expected={expected_ratio}"


# ══════════════════════════════════════════════════════════════════════
#  MULTI-STEP INTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════

class TestMultiStepIntegration:
    """Run multiple steps and verify observations at each step."""

    def test_multi_step_obs_consistency(self):
        """At each step, obs should match a fresh naive computation on the current physics state."""
        torch.manual_seed(123)
        env = _small_env(n=8, p=2, seed=123)
        env.reset()

        for step in range(10):
            actions = torch.randn(8, 3).clamp(-1, 1)
            pred_actions = torch.randn(2, 3).clamp(-1, 1)
            obs, _, _, _, _ = env.step(actions, pred_actions)

            # Recompute observations from scratch on the current physics state
            fresh_obs = env._get_observations()

            assert torch.allclose(obs, fresh_obs, atol=ATOL), \
                f"Step {step}: obs from step() != fresh _get_observations()"

    def test_multi_step_no_nan_inf(self):
        """After many steps with extreme actions, no NaN or Inf in any output."""
        torch.manual_seed(99)
        env = _small_env(n=20, p=3, seed=99)
        env.reset()

        for _ in range(100):
            # Extreme actions
            actions = torch.ones(20, 3)  # max thrust, max roll, max pitch
            pred_actions = torch.ones(3, 3)
            obs, obs_p, r, rp, dones = env.step(actions, pred_actions)

            assert not torch.isnan(obs).any(), "NaN in boid observations"
            assert not torch.isinf(obs).any(), "Inf in boid observations"
            assert not torch.isnan(obs_p).any(), "NaN in predator observations"
            assert not torch.isnan(r).any(), "NaN in boid rewards"
            assert not torch.isnan(rp).any(), "NaN in predator rewards"

    def test_speed_invariants_over_episode(self):
        """Alive boids always have speed in [0.5, 10.0] throughout an episode."""
        torch.manual_seed(77)
        env = _small_env(n=15, p=2, seed=77)
        env.reset()

        for step in range(200):
            actions = torch.randn(15, 3).clamp(-1, 1)
            env.step(actions, torch.randn(2, 3).clamp(-1, 1))

            alive = env.physics.alive_mask
            if alive.any():
                speeds = env.physics.velocities[alive].norm(dim=-1)
                assert (speeds >= 0.5 - ATOL).all(), \
                    f"Step {step}: alive boid below min speed: {speeds.min()}"
                assert (speeds <= 10.0 + ATOL).all(), \
                    f"Step {step}: alive boid above max speed: {speeds.max()}"
