# Metric Report Schema

`analyze_run.py` writes three standard artifacts:

- `summary.json`
- `summary.md`
- plot PNGs for key curves

## `summary.json`

Top-level fields:

- `analysis_version`
- `generated_at_utc`
- `run_id`
- `git_commit`
- `study`
- `experiment`
- `aggregate_summary`
- `episodes`
- `plots`

Each episode entry contains:

- `seed`
- `summary`
- `curves`
- `histograms`

Required curve keys:

- `survival_fraction`
- `capture_rate_per_step`
- `polarization`
- `heading_alignment`
- `connected_components`
- `radial_spread`
- `fringe_fraction`

Required histogram keys:

- `nearest_neighbor_distance`
- `local_density`

## Interpretation Notes

- `capture_rate_per_step` is normalized by predator count
- `connected_components` is computed on the alive-prey distance graph
- `fringe_fraction` is the fraction of alive prey within the outer `90%` radial shell of the swarm

This schema is meant to be stable across week-2 baseline comparisons so tables and review notes can reuse the same fields without ad hoc notebook logic.
