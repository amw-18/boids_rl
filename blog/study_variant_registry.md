# Study Variant Registry

## Active Frozen Study Variants

### `v1-shaped-current`

- Status: active
- Config: `configs/studies/v1_shaped_current.toml`
- Purpose: benchmark the current shaped predator-prey setup before ablations
- Notes:
  - this is the only frozen research variant approved in week 1
  - training and analysis should reference this config directly instead of hardcoded runner defaults

## Baseline Experiment Configs

These are not separate environment variants. They are named evaluation definitions that all point back to the same frozen study version.

- `configs/experiments/baselines/random_vs_random.toml`
- `configs/experiments/baselines/learned_prey_vs_random_predator.toml`
- `configs/experiments/baselines/random_prey_vs_learned_predator.toml`
- `configs/experiments/baselines/learned_prey_vs_learned_predator.toml`
- `configs/experiments/baselines/heuristic_boids_vs_random_predator.toml`
- `configs/experiments/baselines/learned_prey_no_predator_pressure.toml`

## Planned Future Variants

These belong to later roadmap steps and are intentionally not frozen yet:

- `study_b_no_density_reward`
- `study_c_no_visual_confusion`
- `study_d_no_social_features`
- `study_e_no_curriculum`

They should only be added once the team is ready to start the ablation campaign and can define each change as a deliberate config-level variant.
