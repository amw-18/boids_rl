from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from murmur_rl.analysis import EpisodeMetricAccumulator, aggregate_episode_metrics, build_policy_controller
from murmur_rl.experiment import (
    AnalysisExperimentConfig,
    DEFAULT_STUDY_CONFIG,
    PolicyConfig,
    append_registry_row,
    build_run_id,
    build_vector_env,
    ensure_registry,
    git_commit_hash,
    load_analysis_experiment_config,
    load_study_config,
    seed_everything,
    select_device,
    write_json,
)
from murmur_rl.experiment.runtime import analysis_manifest_payload


def _build_cli_experiment_config(args: argparse.Namespace) -> AnalysisExperimentConfig:
    study_path = Path(args.study_config or DEFAULT_STUDY_CONFIG).resolve()
    prey_policy = PolicyConfig(
        type=args.prey_policy,
        checkpoint=args.prey_checkpoint or "",
        deterministic=not args.stochastic_prey,
        label=args.prey_policy,
    )
    predator_policy = PolicyConfig(
        type=args.predator_policy,
        checkpoint=args.predator_checkpoint or "",
        deterministic=not args.stochastic_predator,
        label=args.predator_policy,
    )
    seeds = tuple(args.seeds or [0])
    return AnalysisExperimentConfig(
        id=args.run_id or "adhoc-analysis",
        name=args.run_id or "Adhoc analysis",
        description="CLI-specified analysis run.",
        study_config_path=study_path,
        num_episodes=args.num_episodes or len(seeds),
        seeds=seeds,
        prey_policy=prey_policy,
        predator_policy=predator_policy,
        output_subdir="analysis",
    )


def _coerce_episode_seeds(experiment_config: AnalysisExperimentConfig) -> tuple[int, ...]:
    seeds = list(experiment_config.seeds)
    target_count = experiment_config.num_episodes
    if not seeds:
        seeds = list(range(target_count))
    while len(seeds) < target_count:
        seeds.append(seeds[-1] + 1 if seeds else 0)
    return tuple(seeds[:target_count])


def _stack_curves(episodes: list[dict[str, Any]], key: str) -> np.ndarray:
    max_len = max(len(episode["curves"][key]) for episode in episodes)
    matrix = np.zeros((len(episodes), max_len), dtype=np.float64)
    for index, episode in enumerate(episodes):
        curve = np.asarray(episode["curves"][key], dtype=np.float64)
        matrix[index, : curve.size] = curve
        if curve.size < max_len and curve.size > 0:
            matrix[index, curve.size :] = curve[-1]
    return matrix


def _plot_curve(output_dir: Path, episodes: list[dict[str, Any]], key: str, title: str, ylabel: str) -> str:
    curve_matrix = _stack_curves(episodes, key)
    x_axis = np.arange(curve_matrix.shape[1])
    mean_curve = curve_matrix.mean(axis=0)
    std_curve = curve_matrix.std(axis=0)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x_axis, mean_curve, color="#173F5F", linewidth=2.0)
    axis.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, color="#20639B", alpha=0.2)
    axis.set_title(title)
    axis.set_xlabel("Frame")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    figure.tight_layout()

    file_name = f"{key}.png"
    output_path = output_dir / file_name
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return file_name


def _write_summary_markdown(
    output_dir: Path,
    experiment_config: AnalysisExperimentConfig,
    aggregate_summary: dict[str, Any],
    plot_files: dict[str, str],
) -> None:
    lines = [
        f"# {experiment_config.name}",
        "",
        experiment_config.description,
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Std |",
        "| --- | ---: | ---: |",
    ]
    for key, values in aggregate_summary.items():
        lines.append(f"| `{key}` | {values['mean']:.4f} | {values['std']:.4f} |")

    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for label, file_name in plot_files.items():
        lines.append(f"- `{label}`: [{file_name}](./{file_name})")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_episode(experiment_config: AnalysisExperimentConfig, study_config, device_name: str, seed: int) -> dict[str, Any]:
    seed_everything(seed)
    env = build_vector_env(study_config, device_name)
    prey_controller = build_policy_controller(
        role="prey",
        policy=experiment_config.prey_policy,
        study_config=study_config,
        experiment_config=experiment_config,
        env=env,
        device_name=device_name,
    )
    predator_controller = build_policy_controller(
        role="predator",
        policy=experiment_config.predator_policy,
        study_config=study_config,
        experiment_config=experiment_config,
        env=env,
        device_name=device_name,
    )

    obs_boids, obs_preds = env.reset()
    prey_controller.reset(obs_boids)
    predator_controller.reset(obs_preds)

    accumulator = EpisodeMetricAccumulator(
        population_size=env.n_agents,
        num_predators=env.num_predators,
        space_size=env.space_size,
        perception_radius=env.perception_radius,
        graph_radius=study_config.metrics.distance_graph_radius,
        fringe_radius_fraction=study_config.metrics.fringe_radius_fraction,
        histogram_bins=study_config.metrics.histogram_bins,
    )
    accumulator.record_frame(
        positions=env.physics.positions,
        velocities=env.physics.velocities,
        alive_mask=env.physics.alive_mask,
        captures_this_step=0,
    )

    step_limit = min(study_config.evaluation.episode_steps, env.max_steps)
    for _ in range(step_limit):
        boid_actions = prey_controller.act(env, obs_boids)
        predator_actions = predator_controller.act(env, obs_preds)
        obs_boids, obs_preds, _, _, dones = env.step(boid_actions, predator_actions)
        accumulator.record_frame(
            positions=env.physics.positions,
            velocities=env.physics.velocities,
            alive_mask=env.physics.alive_mask,
            captures_this_step=int(env.physics.last_capture_mask.sum().item()),
        )
        prey_controller.observe(obs_boids, dones)
        predator_controller.observe(obs_preds, None)
        if dones.all():
            break

    episode_result = accumulator.finalize()
    episode_result["seed"] = seed
    return episode_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=str, default="", help="Path to a baseline/evaluation TOML config.")
    parser.add_argument("--study-config", type=str, default="", help="Study config to use for an ad hoc analysis run.")
    parser.add_argument("--prey-policy", type=str, default="random", help="Ad hoc prey policy: random, checkpoint, heuristic_boids, idle.")
    parser.add_argument("--predator-policy", type=str, default="random", help="Ad hoc predator policy: random, checkpoint, idle.")
    parser.add_argument("--prey-checkpoint", type=str, default="", help="Checkpoint path when --prey-policy=checkpoint.")
    parser.add_argument("--predator-checkpoint", type=str, default="", help="Checkpoint path when --predator-policy=checkpoint.")
    parser.add_argument("--stochastic-prey", action="store_true", help="Sample the prey checkpoint policy instead of using deterministic actions.")
    parser.add_argument("--stochastic-predator", action="store_true", help="Sample the predator checkpoint policy instead of using deterministic actions.")
    parser.add_argument("--num-episodes", type=int, default=0, help="Override the number of episodes for ad hoc runs.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Explicit seeds for ad hoc runs.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--output-dir", type=str, default="", help="Optional directory for analysis artifacts.")
    parser.add_argument("--run-id", type=str, default="", help="Optional run identifier for artifacts and registry entries.")
    parser.add_argument("--registry", type=str, default="experiments/registry.csv", help="CSV registry path.")
    args = parser.parse_args()

    if args.experiment_config:
        experiment_config = load_analysis_experiment_config(args.experiment_config)
    else:
        experiment_config = _build_cli_experiment_config(args)

    study_config = experiment_config.load_study_config()
    device_name = select_device(args.device)
    run_id = args.run_id or build_run_id("analysis", experiment_config.id, _coerce_episode_seeds(experiment_config)[0])
    output_dir = Path(args.output_dir).resolve() if args.output_dir else REPO_ROOT / "experiments" / "runs" / run_id / experiment_config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = Path(args.registry).resolve() if Path(args.registry).is_absolute() else (REPO_ROOT / args.registry).resolve()
    git_commit = git_commit_hash(REPO_ROOT)
    manifest_path = output_dir / "analysis_manifest.json"

    manifest = analysis_manifest_payload(
        run_id=run_id,
        experiment_config=experiment_config,
        study_config=study_config,
        config_path=experiment_config.source_path,
        output_dir=output_dir,
        device_name=device_name,
        git_commit=git_commit,
        status="running",
    )
    write_json(manifest_path, manifest)

    try:
        episode_seeds = _coerce_episode_seeds(experiment_config)
        episode_results = [
            _evaluate_episode(experiment_config, study_config, device_name, seed)
            for seed in episode_seeds
        ]
        aggregate_summary = aggregate_episode_metrics(episode_results)

        plot_files = {
            "survival_fraction": _plot_curve(output_dir, episode_results, "survival_fraction", "Survival Fraction", "Fraction alive"),
            "capture_rate_per_step": _plot_curve(output_dir, episode_results, "capture_rate_per_step", "Capture Rate Per Step", "Captures / predator"),
            "polarization": _plot_curve(output_dir, episode_results, "polarization", "Polarization", "Alignment score"),
            "connected_components": _plot_curve(output_dir, episode_results, "connected_components", "Connected Components", "Components"),
        }

        summary_payload = {
            "analysis_version": "week1-v1",
            "generated_at_utc": manifest["created_at_utc"],
            "run_id": run_id,
            "git_commit": git_commit,
            "study": study_config.to_dict(),
            "experiment": experiment_config.to_dict(),
            "aggregate_summary": aggregate_summary,
            "episodes": episode_results,
            "plots": plot_files,
        }
        write_json(output_dir / "summary.json", summary_payload)
        _write_summary_markdown(output_dir, experiment_config, aggregate_summary, plot_files)

        manifest["status"] = "completed"
        manifest["summary_path"] = str((output_dir / "summary.json").resolve())
        write_json(manifest_path, manifest)

        ensure_registry(registry_path)
        append_registry_row(
            registry_path,
            {
                "created_at_utc": manifest["created_at_utc"],
                "run_id": run_id,
                "run_type": "analysis",
                "study_id": study_config.study.id,
                "config_path": str(experiment_config.source_path or ""),
                "git_commit": git_commit,
                "seed": ",".join(str(seed) for seed in episode_seeds),
                "status": "completed",
                "artifact_path": str((output_dir / "summary.json").resolve()),
                "notes": experiment_config.description,
            },
        )
        print(f"Analysis complete. Summary written to {(output_dir / 'summary.json').resolve()}")
    except Exception:
        manifest["status"] = "failed"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
