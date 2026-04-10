import pytest
import torch

from murmur_rl.agents.starling import FalconBrain, StarlingBrain
from murmur_rl.envs.physics import BoidsPhysics
from murmur_rl.envs.vector_env import VectorMurmurationEnv


@pytest.mark.parametrize(
    ("brain_cls", "obs_dim"),
    [
        (StarlingBrain, 18),
        (FalconBrain, 31),
    ],
)
def test_policy_actions_are_bounded_and_logprob_is_reproducible(brain_cls, obs_dim):
    torch.manual_seed(0)
    brain = brain_cls(
        obs_dim=obs_dim,
        global_obs_dim=64,
        action_dim=3,
        hidden_size=64,
        critic_hidden_size=128,
        stacked_frames=1,
    )

    obs = torch.randn(512, obs_dim)
    global_obs = torch.randn(512, 64)

    actions, log_prob, _, values = brain.get_action_and_value(obs, global_obs)
    _, recomputed_log_prob, _, _ = brain.get_action_and_value(obs, global_obs, actions)
    deterministic_actions = brain.get_deterministic_action(obs)

    assert actions.shape == (512, 3)
    assert values.shape == (512, 1)
    assert torch.isfinite(log_prob).all()
    assert torch.all(actions.abs() <= 1.0 + 1e-6)
    assert torch.all(deterministic_actions.abs() <= 1.0 + 1e-6)
    assert torch.allclose(log_prob, recomputed_log_prob, atol=1e-5, rtol=1e-5)


def test_corpses_do_not_retrigger_predator_cooldown():
    phys = BoidsPhysics(num_boids=1, num_predators=1, device=torch.device("cpu"))
    phys.positions[0] = phys.predator_position[0].clone()

    phys._check_captures()
    assert phys.predator_cooldown[0].item() == phys.predator_cooldown_duration
    assert phys.last_capture_counts[0].item() == 1.0

    phys.predator_cooldown[0] = phys.predator_cooldown_duration - 1
    phys._check_captures()

    assert phys.predator_cooldown[0].item() == phys.predator_cooldown_duration - 1
    assert phys.last_capture_counts[0].item() == 0.0


def test_single_kill_is_rewarded_to_only_one_predator():
    env = VectorMurmurationEnv(num_agents=1, num_predators=2, device="cpu")
    env.reset()

    env.physics.positions[0] = torch.tensor([25.0, 25.0, 25.0])
    env.physics.predator_position[0] = torch.tensor([25.0, 25.0, 25.0])
    env.physics.predator_position[1] = torch.tensor([25.0, 25.0, 26.5])
    env.physics._check_captures()

    _, rewards_preds, new_deaths, _, _ = env._get_rewards()

    assert new_deaths[0].item() is True
    assert env.physics.last_capture_counts.tolist() == [1.0, 0.0]
    assert rewards_preds[0].item() == pytest.approx(10.0)
    assert rewards_preds[1].item() == pytest.approx(-0.05)
    assert rewards_preds.sum().item() == pytest.approx(9.95)


def test_dead_prey_are_excluded_from_collision_penalties():
    env = VectorMurmurationEnv(num_agents=2, num_predators=1, device="cpu")
    env.reset()

    env.physics.predator_position[0] = torch.tensor([99.0, 99.0, 99.0])
    env.physics.positions[0] = torch.tensor([50.0, 50.0, 50.0])
    env.physics.positions[1] = torch.tensor([50.0, 50.0, 50.3])
    env.physics.alive_mask[0] = False
    env._dead_mask[0] = True

    rewards, _, new_deaths, _, _ = env._get_rewards()

    assert not new_deaths.any()
    assert rewards[0].item() == pytest.approx(0.0)
    assert rewards[1].item() == pytest.approx(0.1)


def test_global_state_requires_explicit_agent_type_when_counts_match():
    env = VectorMurmurationEnv(num_agents=3, num_predators=3, device="cpu")
    obs_boids, obs_preds = env.reset()

    with pytest.raises(ValueError):
        env.get_global_state(obs_preds)

    boid_state = env.get_boid_global_state(obs_boids)
    pred_state = env.get_predator_global_state(obs_preds)

    assert boid_state.shape == (env.n_agents, env.global_obs_dim)
    assert pred_state.shape == (env.num_predators, env.pred_global_obs_dim)


def test_invalid_k_nearest_slots_are_fully_zeroed():
    env = VectorMurmurationEnv(num_agents=4, num_predators=1, device="cpu")
    obs_boids, _ = env.reset()

    env.physics.alive_mask[:] = torch.tensor([True, False, False, False])
    obs_boids = env._get_observations()
    global_state = env.get_boid_global_state(obs_boids)

    k = min(10, env.n_agents)
    k_features = global_state[0, env.obs_dim:env.obs_dim + k * 7].view(k, 7)

    assert torch.allclose(k_features, torch.zeros_like(k_features))


def test_small_swarm_predator_metadata_and_motion_limits_are_consistent():
    env = VectorMurmurationEnv(
        num_agents=3,
        num_predators=2,
        base_speed=7.0,
        max_turn_angle=0.3,
        max_force=1.5,
        device="cpu",
    )
    _, pred_obs = env.reset()
    pred_global_state = env.get_predator_global_state(pred_obs)

    assert env.pred_obs_dim == 31
    assert pred_obs.shape == (env.num_predators, env.pred_obs_dim)
    assert pred_global_state.shape == (env.num_predators, env.pred_global_obs_dim)
    assert env.physics.base_speed == pytest.approx(7.0)
    assert env.physics.max_force == pytest.approx(1.5)
    assert env.physics.predator_base_speed == pytest.approx(7.0)
    assert env.physics.predator_sprint_speed == pytest.approx(10.5)
    assert env.physics.predator_turn_angle == pytest.approx(0.45)
