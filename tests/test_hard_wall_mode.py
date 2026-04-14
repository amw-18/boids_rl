import torch

from murmur_rl.envs.vector_env import VectorMurmurationEnv


def test_hard_walls_keep_agents_inside_cube_and_reflect_velocity():
    env = VectorMurmurationEnv(
        num_agents=2,
        num_predators=1,
        space_size=10.0,
        dt=1.0,
        boundary_mode="hard_walls",
        wall_soft_margin=2.0,
        wall_penalty=0.5,
        predator_wall_penalty=0.25,
        device="cpu",
    )
    env.reset()

    env.physics.positions[:] = torch.tensor(
        [
            [9.8, 5.0, 5.0],
            [0.2, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    env.physics.velocities[:] = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    env.physics.predator_position[:] = torch.tensor([[9.7, 3.0, 3.0]], dtype=torch.float32)
    env.physics.predator_velocity[:] = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32)

    boid_actions = torch.zeros((env.n_agents, env.action_dim), dtype=torch.float32)
    pred_actions = torch.zeros((env.num_predators, env.action_dim), dtype=torch.float32)
    env.step(boid_actions, pred_actions)

    assert torch.all(env.physics.positions >= 0.0)
    assert torch.all(env.physics.positions <= env.space_size)
    assert torch.all(env.physics.predator_position >= 0.0)
    assert torch.all(env.physics.predator_position <= env.space_size)
    assert env.physics.velocities[0, 0] < 0.0
    assert env.physics.velocities[1, 0] > 0.0
    assert env.physics.predator_velocity[0, 0] < 0.0


def test_hard_wall_rewards_include_wall_penalty_and_shared_team_catch_reward():
    env = VectorMurmurationEnv(
        num_agents=1,
        num_predators=2,
        space_size=10.0,
        boundary_mode="hard_walls",
        wall_soft_margin=2.0,
        wall_penalty=0.5,
        predator_wall_penalty=0.25,
        predator_catch_reward=10.0,
        predator_team_catch_reward=1.0,
        curriculum_enabled=False,
        predator_catch_radius_start=2.0,
        device="cpu",
    )
    env.reset()

    env.physics.positions[:] = torch.tensor([[0.0, 5.0, 5.0]], dtype=torch.float32)
    env.physics.velocities[:] = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    env.physics.predator_position[:] = torch.tensor(
        [
            [0.0, 5.0, 5.0],
            [9.5, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    env.physics.predator_velocity[:] = torch.zeros_like(env.physics.predator_velocity)
    env.physics.alive_mask[:] = torch.tensor([True])
    env._dead_mask.zero_()
    env._invalidate_step_cache()

    env.physics._check_captures()
    rewards_boids, rewards_preds, new_deaths, _, _ = env._get_rewards()

    assert bool(new_deaths[0].item()) is True
    assert rewards_boids[0].item() <= -100.0
    assert rewards_preds.max().item() >= 10.5
    assert rewards_preds.min().item() >= 0.75
