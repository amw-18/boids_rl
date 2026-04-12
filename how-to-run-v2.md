The recommended study for new training is [v2_hard_walls.toml](/D:/workspace/boids_rl/configs/studies/v2_hard_walls.toml).

Why this exists:

- `v1_shaped_current.toml` is kept for reproducing the older escape-equilibrium runs.
- `v2_hard_walls.toml` is the corrected setup with a real arena, wall penalties, and stronger predator-side learning defaults.

Recommended first run:

1. Use [v2_hard_walls_scratch.toml](/D:/workspace/boids_rl/configs/studies/v2_hard_walls_scratch.toml) for a `1000` epoch smoke run.
2. Keep `checkpoint_interval = 500`.
3. If you are on CPU, consider setting `compile_models = false` and `compile_env = false` in the scratch config to reduce startup overhead.

Scratch training command:

```powershell
uv run python -m murmur_rl.training.runner --config configs/studies/v2_hard_walls_scratch.toml --run-id scratch-s0 --seed 0 --device auto --no-wandb
```

Full training command:

```powershell
uv run python -m murmur_rl.training.runner --config configs/studies/v2_hard_walls.toml --run-id full-s0 --seed 0 --device auto --no-wandb
```

Checkpoint playback:

```powershell
uv run python simulate.py --checkpoint checkpoints/scratch-s0/starling_brain_ep500.pth --frames 600 --num-boids 100 --num-predators 10
```

Metric summary:

```powershell
uv run python analyze_run.py --study-config configs/studies/v2_hard_walls.toml --prey-policy checkpoint --prey-checkpoint checkpoints/scratch-s0/starling_brain_ep500.pth --predator-policy checkpoint --predator-checkpoint checkpoints/scratch-s0/falcon_brain_ep500.pth --num-episodes 1 --seeds 0 --run-id scratch-s0-eval
```
