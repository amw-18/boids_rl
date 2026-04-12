from pathlib import Path

import pytest

from murmur_rl.experiment import (
    DEFAULT_STUDY_CONFIG,
    apply_environment_overrides,
    build_vector_env,
    load_analysis_experiment_config,
    load_study_config,
)


def test_default_study_config_matches_current_shaped_defaults():
    study_config = load_study_config(DEFAULT_STUDY_CONFIG)

    assert study_config.study.id == "v1-shaped-current"
    assert study_config.environment.num_agents == 100
    assert study_config.environment.num_predators == 10
    assert study_config.environment.space_size == pytest.approx(50.0)
    assert study_config.training.rollout_steps == 500
    assert study_config.training.predator_actor_lr == pytest.approx(1e-4)

    env = build_vector_env(study_config, "cpu")
    assert env.n_agents == 100
    assert env.num_predators == 10
    assert env.physics.predator_sprint_speed == pytest.approx(7.5)
    assert env.max_steps == 500


def test_analysis_config_resolves_relative_study_path():
    experiment_config = load_analysis_experiment_config("configs/experiments/baselines/random_vs_random.toml")

    assert experiment_config.study_config_path == Path(DEFAULT_STUDY_CONFIG).resolve()
    assert experiment_config.prey_policy.type == "random"
    assert experiment_config.predator_policy.type == "random"
    assert experiment_config.seeds == (0, 1, 2)


def test_environment_override_can_disable_predators():
    study_config = load_study_config(DEFAULT_STUDY_CONFIG)
    disabled_predator_config = apply_environment_overrides(study_config, {"num_predators": 0})
    env = build_vector_env(disabled_predator_config, "cpu")

    obs_boids, obs_preds = env.reset()

    assert obs_boids.shape == (env.n_agents, env.obs_dim)
    assert obs_preds.shape[0] == 0
    assert env.num_predators == 0
    assert env.physics.predator_position.shape[0] == 0
    assert env.physics.last_capture_counts.shape[0] == 0
    assert obs_boids[:, 11:15].abs().sum().item() == pytest.approx(0.0)
