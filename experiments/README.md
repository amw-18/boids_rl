# Experiment Registry

`registry.csv` is the week-1 append-only experiment index.

Each row should point to one concrete training or analysis run and include:

- `run_id`
- `run_type`
- `study_id`
- `config_path`
- `git_commit`
- `seed`
- `status`
- `artifact_path`

Detailed metadata lives alongside each run in `experiments/runs/<run-id>/`.

For training runs:

- `run_manifest.json` captures the frozen study config, device, seed, checkpoint directory, and latest checkpoint paths.

For analysis runs:

- `analysis_manifest.json` captures the study config, evaluation config, policy sources, and artifact directory.
- `summary.json` is the fixed machine-readable metric report.
- `summary.md` is the reviewer-facing summary table.
