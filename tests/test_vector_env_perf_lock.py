import torch

from murmur_rl.agents.starling import FalconBrain, StarlingBrain
from murmur_rl.training.ppo import AlternatingCoevolutionTrainer
from murmur_rl.envs.vector_env import VectorMurmurationEnv


ATOL = 1e-6


def _make_env(*, n=8, p=2, seed=0):
    torch.manual_seed(seed)
    return VectorMurmurationEnv(
        num_agents=n,
        num_predators=p,
        space_size=100.0,
        perception_radius=15.0,
        base_speed=5.0,
        max_turn_angle=0.5,
        max_force=2.0,
        min_speed=2.5,
        device="cpu",
    )


def _sync_env_state(src: VectorMurmurationEnv, dst: VectorMurmurationEnv):
    dst.num_moves = src.num_moves
    dst.env_step_counter = src.env_step_counter
    dst.physics.predator_catch_radius = src.physics.predator_catch_radius
    dst._dead_mask.copy_(src._dead_mask)
    dst.last_potential = src.last_potential.clone()
    dst.last_pred_potential = src.last_pred_potential.clone()

    for name in [
        "positions",
        "velocities",
        "up_vectors",
        "predator_position",
        "predator_velocity",
        "predator_up_vectors",
        "predator_stamina",
        "predator_cooldown",
        "predator_time_since_cooldown",
        "last_capture_mask",
        "last_capture_predators",
        "last_capture_counts",
        "alive_mask",
    ]:
        value = getattr(src.physics, name)
        target = getattr(dst.physics, name)
        if torch.is_tensor(value):
            if target.shape != value.shape or target.dtype != value.dtype:
                setattr(dst.physics, name, value.clone())
            else:
                target.copy_(value)
        else:
            setattr(dst.physics, name, value)

    dst._invalidate_step_cache()


def _reference_boid_observations(env: VectorMurmurationEnv):
    pos = env.physics.positions
    vel = env.physics.velocities
    alive = env.physics.alive_mask

    dist_matrix = torch.cdist(pos, pos)
    mask_out = env._diag_mask | ~alive.unsqueeze(0)
    dist_matrix = torch.where(mask_out, env._inf, dist_matrix)

    nearest_dist = dist_matrix.min(dim=1, keepdim=True).values
    nearest_dist = torch.where(
        nearest_dist == float("inf"),
        env._perception_r,
        nearest_dist,
    )
    nearest_dist = nearest_dist / env.perception_radius

    in_radius = dist_matrix < env.perception_radius
    in_radius_f = in_radius.float()
    local_density = in_radius_f.sum(dim=1, keepdim=True) / env.n_agents

    neighbor_counts = in_radius_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    avg_vel = (in_radius_f @ vel) / neighbor_counts
    local_alignment = avg_vel / avg_vel.norm(dim=-1, keepdim=True).clamp(min=1e-5)

    avg_pos = (in_radius_f @ pos) / neighbor_counts
    dir_to_com = avg_pos - pos
    com_direction = dir_to_com / dir_to_com.norm(dim=-1, keepdim=True).clamp(min=1e-5)

    has_neighbors = in_radius.any(dim=1, keepdim=True)
    is_active = has_neighbors & alive.unsqueeze(1)
    local_alignment = local_alignment * is_active.float()
    com_direction = com_direction * is_active.float()

    if env.num_predators > 0:
        pred_pos = env.physics.predator_position
        pred_vel = env.physics.predator_velocity
        dist_to_preds = torch.cdist(pos, pred_pos)
        closest_pred_idx = torch.argmin(dist_to_preds, dim=1)

        closest_pred_pos = pred_pos[closest_pred_idx]
        closest_pred_vel = pred_vel[closest_pred_idx]
        dx = closest_pred_pos - pos
        dv = closest_pred_vel - vel

        d = dx.norm(dim=-1, keepdim=True)
        d_norm = (d / (env.space_size / 2.0)).clamp(max=1.0)

        u = dx / d.clamp(min=1e-5)
        max_v_close = env.physics.predator_sprint_speed + env.physics.base_speed
        v_close = -(dv * u).sum(dim=-1, keepdim=True)
        v_close_norm = (v_close / max_v_close).clamp(-1.0, 1.0)

        loom = v_close / d.clamp(min=1e-5)
        loom_norm = (loom / 5.0).clamp(-1.0, 1.0)

        vel_unit = vel / vel.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        in_front = (vel_unit * u).sum(dim=-1, keepdim=True)

        far = d > env._half_space
        v_close_norm = torch.where(far, env._zero, v_close_norm)
        loom_norm = torch.where(far, env._zero, loom_norm)
        in_front = torch.where(far, env._zero, in_front)
    else:
        d_norm = torch.zeros((env.n_agents, 1), device=env.device)
        v_close_norm = torch.zeros((env.n_agents, 1), device=env.device)
        loom_norm = torch.zeros((env.n_agents, 1), device=env.device)
        in_front = torch.zeros((env.n_agents, 1), device=env.device)

    pos_relative = (pos - env._half_space) / env._half_space
    vel_norm = vel / env.physics.base_speed
    return torch.cat(
        [
            vel_norm,
            nearest_dist,
            local_density,
            local_alignment,
            com_direction,
            d_norm,
            v_close_norm,
            loom_norm,
            in_front,
            pos_relative,
        ],
        dim=1,
    )


def _reference_predator_observations(env: VectorMurmurationEnv):
    pred_pos = env.physics.predator_position
    pred_vel = env.physics.predator_velocity
    boid_pos = env.physics.positions
    alive = env.physics.alive_mask

    pos_relative = (pred_pos - env._half_space) / env._half_space
    vel_norm = pred_vel / env.physics.predator_sprint_speed
    stamina_norm = (env.physics.predator_stamina / env.physics.predator_max_stamina).unsqueeze(1)

    alive_f = alive.float().unsqueeze(1)
    num_alive = alive_f.sum().clamp(min=1.0)
    com = (boid_pos * alive_f).sum(dim=0) / num_alive
    com_relative = (com - pred_pos) / env._half_space

    dist_matrix = torch.cdist(pred_pos, boid_pos)
    dist_matrix = torch.where(alive.unsqueeze(0), dist_matrix, env._inf)

    k = min(5, env.n_agents)
    closest_dists, closest_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False)

    target_obs = []
    for i in range(k):
        target_ids = closest_idx[:, i]
        dists = closest_dists[:, i:i+1]
        is_valid = (dists < env._inf).float()

        target_positions = boid_pos[target_ids]
        target_velocities = env.physics.velocities[target_ids]

        b_b_dist = torch.cdist(target_positions, boid_pos)
        b_b_dist = torch.where(alive.unsqueeze(0), b_b_dist, env._inf)
        target_density = (b_b_dist < env.perception_radius).float().sum(dim=1, keepdim=True) / env.n_agents

        sigma = target_density * env.predator_visual_noise_variance
        noise = torch.randn_like(target_positions) * sigma

        obfuscated_target_pos = target_positions + noise
        rel_pos = (obfuscated_target_pos - pred_pos) / env._half_space
        rel_vel = (target_velocities - pred_vel) / (env.physics.predator_sprint_speed + env.physics.base_speed)

        target_obs.append(rel_pos * is_valid)
        target_obs.append(rel_vel * is_valid)
        target_obs.append((dists / (env.space_size * 1.5)) * is_valid)

    target_obs_tensor = (
        torch.cat(target_obs, dim=1)
        if target_obs
        else torch.zeros((env.num_predators, 0), device=env.device)
    )
    return torch.cat([pos_relative, vel_norm, stamina_norm, com_relative, target_obs_tensor], dim=1)


def _reference_global_state(env: VectorMurmurationEnv, local_obs, focal_pos, *, exclude_self):
    batch_size = local_obs.shape[0]
    pos = env.physics.positions
    vel = env.physics.velocities
    alive = env.physics.alive_mask

    K = min(10, env.n_agents)
    dist_matrix = torch.cdist(focal_pos, pos)
    dist_matrix = torch.where(alive.unsqueeze(0), dist_matrix, env._inf)
    if exclude_self:
        dist_matrix = torch.where(env._diag_mask, env._inf, dist_matrix)

    _, closest_idx = torch.topk(dist_matrix, k=K, dim=1, largest=False)

    k_pos = pos[closest_idx]
    k_vel = vel[closest_idx]
    k_alive = alive[closest_idx].float()
    if exclude_self:
        focal_idx = torch.arange(batch_size, device=env.device).unsqueeze(1)
        k_alive = torch.where(closest_idx == focal_idx, 0.0, k_alive)
    k_alive = k_alive.unsqueeze(-1)

    rel_pos = ((k_pos - focal_pos.unsqueeze(1)) / env._half_space) * k_alive
    rel_vel = (k_vel / env.physics.base_speed) * k_alive
    k_features = torch.cat([rel_pos, rel_vel, k_alive], dim=-1).view(batch_size, -1)

    pred_pos = env.physics.predator_position
    pred_vel = env.physics.predator_velocity
    rel_pred_pos = (pred_pos.unsqueeze(0) - focal_pos.unsqueeze(1)) / env._half_space
    rel_pred_vel = pred_vel.unsqueeze(0).expand(batch_size, -1, -1) / env.physics.predator_sprint_speed
    pred_features = torch.cat([rel_pred_pos, rel_pred_vel], dim=-1).view(batch_size, -1)
    return torch.cat([local_obs, k_features, pred_features], dim=1)


def _reference_rewards(env: VectorMurmurationEnv):
    pos = env.physics.positions
    alive = env.physics.alive_mask

    dist_matrix = torch.cdist(pos, pos)
    dist_matrix = torch.where(env._diag_mask, env._inf, dist_matrix)
    live_pair_mask = alive.unsqueeze(0) & alive.unsqueeze(1)
    live_dist_matrix = torch.where(live_pair_mask, dist_matrix, env._inf)

    collision_count = (live_dist_matrix < 2.0).sum(dim=1).float()
    new_deaths = env.physics.last_capture_mask

    pos_relative = (pos - env._half_space) / env._half_space
    d_center_sq = (pos_relative**2).sum(dim=-1)
    phi_bounds = -env._pbrs_k * d_center_sq

    local_density = (live_dist_matrix < env.perception_radius).float().sum(dim=1) / env.n_agents
    phi_density = env._pbrs_c * local_density
    new_potential = phi_bounds + phi_density

    rewards = torch.full((env.n_agents,), env._survival_reward.item(), device=env.device)
    rewards -= env._collision_penalty * collision_count
    rewards = torch.where(new_deaths, env._death_penalty, rewards)
    rewards = torch.where(env._dead_mask, env._zero, rewards)
    new_potential = torch.where(env._dead_mask | new_deaths, env._zero, new_potential)

    pred_pos = env.physics.predator_position
    pred_pos_relative = (pred_pos - env._half_space) / env._half_space
    pred_d_center_sq = (pred_pos_relative**2).sum(dim=-1)
    pred_phi_bounds = -env._pbrs_k * pred_d_center_sq

    rewards_preds = torch.zeros(env.num_predators, device=env.device)
    catches_per_pred = env.physics.last_capture_counts
    rewards_preds += env._predator_catch_reward * catches_per_pred

    is_cooldown = env.physics.predator_cooldown > 0
    made_catch = catches_per_pred > 0
    rewards_preds += torch.where(
        ~is_cooldown & ~made_catch,
        env._predator_hunger_penalty,
        env._zero,
    )

    return rewards, rewards_preds, new_deaths, new_potential, pred_phi_bounds


def _reference_step(env: VectorMurmurationEnv, boid_actions: torch.Tensor, predator_actions: torch.Tensor):
    env.physics.step(boid_actions=boid_actions, predator_actions=predator_actions)
    env.num_moves += 1
    env.env_step_counter += 1
    env._apply_curriculum()
    env._invalidate_step_cache()

    obs_boids = _reference_boid_observations(env)
    obs_preds = _reference_predator_observations(env)
    rewards_boids, rewards_preds, new_deaths, new_potential, new_pred_potential = _reference_rewards(env)

    rewards_boids = rewards_boids + ((env._gamma * new_potential) - env.last_potential)
    env.last_potential = new_potential.clone()

    rewards_preds = rewards_preds + ((env._gamma * new_pred_potential) - env.last_pred_potential)
    env.last_pred_potential = new_pred_potential.clone()

    env._dead_mask |= new_deaths

    if env.num_moves >= env.max_steps:
        dones = torch.ones(env.n_agents, dtype=torch.bool, device=env.device)
    else:
        dones = env._dead_mask.clone()

    return obs_boids, obs_preds, rewards_boids, rewards_preds, dones


def _make_trainer_for_rollout(seed: int):
    torch.manual_seed(seed)
    env = VectorMurmurationEnv(
        num_agents=5,
        num_predators=2,
        device="cpu",
        max_steps=50,
    )
    boid_brain = StarlingBrain(
        obs_dim=env.obs_dim,
        global_obs_dim=env.global_obs_dim,
        action_dim=env.action_dim,
        hidden_size=32,
        critic_hidden_size=64,
        stacked_frames=3,
    )
    pred_brain = FalconBrain(
        obs_dim=env.pred_obs_dim,
        global_obs_dim=env.pred_global_obs_dim,
        action_dim=env.action_dim,
        hidden_size=32,
        critic_hidden_size=64,
        stacked_frames=3,
    )
    return AlternatingCoevolutionTrainer(
        env=env,
        boid_brain=boid_brain,
        pred_brain=pred_brain,
        device=torch.device("cpu"),
        update_epochs=1,
        batch_size=16,
        stacked_frames=3,
    )


def _reference_collect_rollouts(trainer: AlternatingCoevolutionTrainer, num_steps=5):
    obs_boids, obs_preds = trainer.env.reset()
    global_obs_boids = trainer.env.get_boid_global_state(obs_boids)
    global_obs_preds = trainer.env.get_predator_global_state(obs_preds)

    N = trainer.env.n_agents
    P = trainer.env.num_predators

    roll_obs_boids = obs_boids.unsqueeze(1).repeat(1, trainer.stacked_frames, 1)
    roll_global_boids = global_obs_boids.unsqueeze(1).repeat(1, trainer.stacked_frames, 1)

    roll_obs_preds = obs_preds.unsqueeze(1).repeat(1, trainer.stacked_frames, 1)
    roll_global_preds = global_obs_preds.unsqueeze(1).repeat(1, trainer.stacked_frames, 1)

    b_obs = torch.empty((num_steps, N, trainer.stacked_frames * obs_boids.shape[-1]), device=trainer.device)
    b_globs = torch.empty((num_steps, N, trainer.stacked_frames * global_obs_boids.shape[-1]), device=trainer.device)
    b_acts = torch.empty((num_steps, N, trainer.env.action_dim), device=trainer.device)
    b_logps = torch.empty((num_steps, N), device=trainer.device)
    b_rews = torch.empty((num_steps, N), device=trainer.device)
    b_dones = torch.empty((num_steps, N), device=trainer.device)
    b_vals = torch.empty((num_steps, N), device=trainer.device)

    p_obs = torch.empty((num_steps, P, trainer.stacked_frames * obs_preds.shape[-1]), device=trainer.device)
    p_globs = torch.empty((num_steps, P, trainer.stacked_frames * global_obs_preds.shape[-1]), device=trainer.device)
    p_acts = torch.empty((num_steps, P, trainer.env.action_dim), device=trainer.device)
    p_logps = torch.empty((num_steps, P), device=trainer.device)
    p_rews = torch.empty((num_steps, P), device=trainer.device)
    p_dones = torch.empty((num_steps, P), device=trainer.device)
    p_vals = torch.empty((num_steps, P), device=trainer.device)

    for step in range(num_steps):
        if step > 0:
            global_obs_boids = trainer.env.get_boid_global_state(obs_boids)
            global_obs_preds = trainer.env.get_predator_global_state(obs_preds)

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
            b_action, b_logp, _, b_nv = trainer.boid_brain.get_action_and_value(flat_obs_boids, flat_globs_boids)
            b_value = trainer.boid_value_norm.denormalize(b_nv)

            p_action, p_logp, _, p_nv = trainer.pred_brain.get_action_and_value(flat_obs_preds, flat_globs_preds)
            p_value = trainer.pred_value_norm.denormalize(p_nv)

        next_obs_boids, next_obs_preds, rewards_boids, rewards_preds, dones_boids = trainer.env.step(
            boid_actions=b_action, predator_actions=p_action
        )

        b_acts[step], b_logps[step], b_rews[step], b_dones[step], b_vals[step] = (
            b_action,
            b_logp,
            rewards_boids,
            dones_boids.float(),
            b_value.flatten(),
        )

        p_dones_val = torch.zeros(P, device=trainer.device)
        if torch.all(dones_boids):
            p_dones_val.fill_(1.0)

        p_acts[step], p_logps[step], p_rews[step], p_dones[step], p_vals[step] = (
            p_action,
            p_logp,
            rewards_preds,
            p_dones_val,
            p_value.flatten(),
        )

        if dones_boids.any():
            dead_mask = dones_boids.bool()
            roll_obs_boids[dead_mask] = next_obs_boids[dead_mask].unsqueeze(1).repeat(1, trainer.stacked_frames, 1)

        obs_boids, obs_preds = next_obs_boids, next_obs_preds

    global_obs_boids = trainer.env.get_boid_global_state(obs_boids)
    roll_obs_boids = torch.cat([roll_obs_boids[:, 1:, :], obs_boids.unsqueeze(1)], dim=1)
    roll_global_boids = torch.cat([roll_global_boids[:, 1:, :], global_obs_boids.unsqueeze(1)], dim=1)

    global_obs_preds = trainer.env.get_predator_global_state(obs_preds)
    roll_obs_preds = torch.cat([roll_obs_preds[:, 1:, :], obs_preds.unsqueeze(1)], dim=1)
    roll_global_preds = torch.cat([roll_global_preds[:, 1:, :], global_obs_preds.unsqueeze(1)], dim=1)

    boid_rollouts = {
        "obs": b_obs,
        "global_obs": b_globs,
        "actions": b_acts,
        "logprobs": b_logps,
        "rewards": b_rews,
        "dones": b_dones,
        "values": b_vals,
        "final_obs": roll_obs_boids.view(N, -1),
        "final_global_obs": roll_global_boids.view(N, -1),
    }

    pred_rollouts = {
        "obs": p_obs,
        "global_obs": p_globs,
        "actions": p_acts,
        "logprobs": p_logps,
        "rewards": p_rews,
        "dones": p_dones,
        "values": p_vals,
        "final_obs": roll_obs_preds.view(P, -1),
        "final_global_obs": roll_global_preds.view(P, -1),
    }

    return boid_rollouts, pred_rollouts


def test_fixed_seed_rollout_outputs_match_reference():
    env = _make_env(n=8, p=3, seed=123)
    env.reset()

    ref_env = _make_env(n=8, p=3, seed=999)
    _sync_env_state(env, ref_env)
    _, _, _, ref_potential, ref_pred_potential = _reference_rewards(ref_env)
    ref_env.last_potential = ref_potential.clone()
    ref_env.last_pred_potential = ref_pred_potential.clone()

    action_gen = torch.Generator().manual_seed(2024)
    for step in range(3):
        boid_actions = torch.randn((env.n_agents, env.action_dim), generator=action_gen).clamp(-1, 1)
        pred_actions = torch.randn((env.num_predators, env.action_dim), generator=action_gen).clamp(-1, 1)
        noise_seed = 5000 + step

        torch.manual_seed(noise_seed)
        obs_boids, obs_preds, rewards_boids, rewards_preds, dones = env.step(boid_actions, pred_actions)
        boid_global = env.get_boid_global_state(obs_boids)
        pred_global = env.get_predator_global_state(obs_preds)

        torch.manual_seed(noise_seed)
        ref_obs_boids, ref_obs_preds, ref_rewards_boids, ref_rewards_preds, ref_dones = _reference_step(
            ref_env,
            boid_actions,
            pred_actions,
        )
        ref_boid_global = _reference_global_state(ref_env, ref_obs_boids, ref_env.physics.positions, exclude_self=True)
        ref_pred_global = _reference_global_state(ref_env, ref_obs_preds, ref_env.physics.predator_position, exclude_self=False)

        assert torch.allclose(obs_boids, ref_obs_boids, atol=ATOL, rtol=ATOL)
        assert torch.allclose(obs_preds, ref_obs_preds, atol=ATOL, rtol=ATOL)
        assert torch.allclose(rewards_boids, ref_rewards_boids, atol=ATOL, rtol=ATOL)
        assert torch.allclose(rewards_preds, ref_rewards_preds, atol=ATOL, rtol=ATOL)
        assert torch.equal(dones, ref_dones)
        assert torch.allclose(boid_global, ref_boid_global, atol=ATOL, rtol=ATOL)
        assert torch.allclose(pred_global, ref_pred_global, atol=ATOL, rtol=ATOL)


def test_predator_target_ordering_matches_reference():
    env = _make_env(n=6, p=2, seed=0)
    env.reset()
    env.physics.positions[:] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
            [19.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    env.physics.predator_position[:] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    env.physics.alive_mask[:] = torch.tensor([True, True, False, True, True, False])

    env._invalidate_step_cache()
    cache = env._ensure_step_cache()

    dist_matrix = torch.cdist(env.physics.predator_position, env.physics.positions)
    dist_matrix = torch.where(env.physics.alive_mask.unsqueeze(0), dist_matrix, env._inf)
    _, expected_idx = torch.topk(dist_matrix, k=min(5, env.n_agents), dim=1, largest=False)

    assert torch.equal(cache["pred_target_idx"], expected_idx)


def test_predator_invalid_slots_stay_zeroed():
    env = _make_env(n=4, p=2, seed=1)
    env.reset()
    env.physics.alive_mask[:] = torch.tensor([True, False, False, False])

    torch.manual_seed(77)
    pred_obs = env._get_predator_observations()
    target_blocks = pred_obs[:, 10:].view(env.num_predators, min(5, env.n_agents), 7)

    assert torch.allclose(target_blocks[:, 1:], torch.zeros_like(target_blocks[:, 1:]), atol=ATOL, rtol=ATOL)


def test_predator_density_scaled_noise_matches_reference():
    env = _make_env(n=7, p=2, seed=5)
    env.reset()
    env.physics.positions[:] = torch.tensor(
        [
            [50.0, 50.0, 50.0],
            [51.0, 50.0, 50.0],
            [52.0, 50.0, 50.0],
            [80.0, 80.0, 80.0],
            [81.0, 80.0, 80.0],
            [82.0, 80.0, 80.0],
            [83.0, 80.0, 80.0],
        ],
        dtype=torch.float32,
    )
    env.physics.alive_mask[:] = torch.tensor([True, True, True, True, False, True, True])

    torch.manual_seed(1234)
    actual = env._get_predator_observations()
    torch.manual_seed(1234)
    expected = _reference_predator_observations(env)

    assert torch.allclose(actual, expected, atol=ATOL, rtol=ATOL)


def test_cache_rebuilds_after_manual_state_change():
    env = _make_env(n=5, p=2, seed=21)
    env.reset()
    first = env._get_observations()
    env.physics.positions[0] = env.physics.positions[0] + torch.tensor([10.0, 0.0, 0.0])

    second = env._get_observations()
    expected = _reference_boid_observations(env)

    assert not torch.allclose(first, second, atol=ATOL, rtol=ATOL)
    assert torch.allclose(second, expected, atol=ATOL, rtol=ATOL)


def test_collect_rollouts_matches_reference_history_updates():
    trainer = _make_trainer_for_rollout(seed=11)
    ref_trainer = _make_trainer_for_rollout(seed=11)

    torch.manual_seed(222)
    actual_boids, actual_preds = trainer.collect_rollouts(num_steps=5)
    torch.manual_seed(222)
    ref_boids, ref_preds = _reference_collect_rollouts(ref_trainer, num_steps=5)

    for key in actual_boids:
        assert torch.allclose(actual_boids[key], ref_boids[key], atol=ATOL, rtol=ATOL)
    for key in actual_preds:
        assert torch.allclose(actual_preds[key], ref_preds[key], atol=ATOL, rtol=ATOL)


def test_cached_global_state_matches_reference_across_edge_cases():
    scenarios = [
        {"n": 3, "p": 3, "seed": 0, "dead": None},
        {"n": 6, "p": 2, "seed": 1, "dead": torch.tensor([False, True, False, False, True, False])},
        {"n": 4, "p": 1, "seed": 2, "dead": torch.tensor([False, False, True, True])},
    ]

    for scenario in scenarios:
        env = _make_env(n=scenario["n"], p=scenario["p"], seed=scenario["seed"])
        obs_boids, _ = env.reset()
        if scenario["dead"] is not None:
            env.physics.alive_mask[:] = ~scenario["dead"]
            obs_boids = env._get_observations()

        torch.manual_seed(404)
        obs_preds = env._get_predator_observations()

        boid_state = env.get_boid_global_state(obs_boids)
        pred_state = env.get_predator_global_state(obs_preds)

        ref_boid_state = _reference_global_state(env, obs_boids, env.physics.positions, exclude_self=True)
        ref_pred_state = _reference_global_state(env, obs_preds, env.physics.predator_position, exclude_self=False)

        assert torch.allclose(boid_state, ref_boid_state, atol=ATOL, rtol=ATOL)
        assert torch.allclose(pred_state, ref_pred_state, atol=ATOL, rtol=ATOL)
