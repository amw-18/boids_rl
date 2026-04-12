from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import torch

from murmur_rl.envs.vector_env import _HAS_TRITON
from murmur_rl.experiment import (
    DEFAULT_STUDY_CONFIG,
    build_boid_brain,
    build_predator_brain,
    build_vector_env,
    load_study_config,
    seed_everything,
    select_device,
)
from murmur_rl.training.ppo import AlternatingCoevolutionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeatable profiler for the single-process single-GPU training path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_STUDY_CONFIG),
        help="Path to a frozen study config TOML.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed seed used for repeatable warmup and measured runs.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=50,
        help="Number of rollout steps in the measured pass.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of rollout steps in the warmup pass.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured repeats to average.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text summary.",
    )
    return parser.parse_args()


def _sync(device_name: str) -> None:
    if device_name == "cuda":
        torch.cuda.synchronize()
    elif device_name == "mps":
        torch.mps.synchronize()


def _maybe_compile_for_profile(study_config, device_name: str, env, boid_brain, pred_brain):
    if device_name == "cuda":
        torch.set_float32_matmul_precision("high")

    if study_config.training.compile_models:
        if device_name == "cuda" and not _HAS_TRITON:
            pass
        else:
            mode = "reduce-overhead" if _HAS_TRITON else "default"
            try:
                boid_brain = torch.compile(boid_brain, mode=mode)
                pred_brain = torch.compile(pred_brain, mode=mode)
            except Exception:
                pass

    if study_config.training.compile_env:
        try:
            env.compile()
        except Exception:
            pass

    return boid_brain, pred_brain


def _build_trainer(study_config, device_name: str, seed: int) -> AlternatingCoevolutionTrainer:
    seed_everything(seed)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(seed)

    env = build_vector_env(study_config, device_name)
    boid_brain = build_boid_brain(study_config, env)
    pred_brain = build_predator_brain(study_config, env)
    boid_brain, pred_brain = _maybe_compile_for_profile(
        study_config,
        device_name,
        env,
        boid_brain,
        pred_brain,
    )

    return AlternatingCoevolutionTrainer(
        env=env,
        boid_brain=boid_brain,
        pred_brain=pred_brain,
        device=torch.device(device_name),
        boid_actor_lr=study_config.training.actor_lr,
        pred_actor_lr=study_config.training.predator_actor_lr,
        critic_lr=study_config.training.critic_lr,
        gamma=study_config.environment.gamma,
        gae_lambda=study_config.training.gae_lambda,
        clip_coef=study_config.training.clip_coef,
        ent_coef=study_config.training.ent_coef,
        vf_coef=study_config.training.vf_coef,
        max_grad_norm=study_config.training.max_grad_norm,
        target_kl=study_config.training.target_kl,
        update_epochs=study_config.training.update_epochs,
        batch_size=study_config.training.batch_size,
        stacked_frames=study_config.training.stacked_frames,
    )


def _wrap_timed_method(
    bound_method: Callable,
    *,
    key: str,
    timings: dict[str, float],
    counts: dict[str, int],
    device_name: str,
):
    def wrapper(*args, **kwargs):
        _sync(device_name)
        start = time.perf_counter()
        result = bound_method(*args, **kwargs)
        _sync(device_name)
        timings[key] += time.perf_counter() - start
        counts[key] += 1
        return result

    return wrapper


def _profile_once(
    *,
    study_config,
    device_name: str,
    seed: int,
    rollout_steps: int,
    warmup_steps: int,
) -> dict[str, object]:
    warmup_trainer = _build_trainer(study_config, device_name, seed)
    if warmup_steps > 0:
        warmup_trainer.collect_rollouts(num_steps=warmup_steps)

    trainer = _build_trainer(study_config, device_name, seed)
    env = trainer.env
    timings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    env.step = _wrap_timed_method(
        env.step,
        key="env_step",
        timings=timings,
        counts=counts,
        device_name=device_name,
    )
    env.physics.step = _wrap_timed_method(
        env.physics.step,
        key="physics_step",
        timings=timings,
        counts=counts,
        device_name=device_name,
    )
    env._get_observations = _wrap_timed_method(
        env._get_observations,
        key="boid_obs",
        timings=timings,
        counts=counts,
        device_name=device_name,
    )
    env._get_predator_observations = _wrap_timed_method(
        env._get_predator_observations,
        key="predator_obs",
        timings=timings,
        counts=counts,
        device_name=device_name,
    )
    env._get_rewards = _wrap_timed_method(
        env._get_rewards,
        key="rewards",
        timings=timings,
        counts=counts,
        device_name=device_name,
    )
    env.get_boid_global_state = _wrap_timed_method(
        env.get_boid_global_state,
        key="boid_global_state",
        timings=timings,
        counts=counts,
        device_name=device_name,
    )
    env.get_predator_global_state = _wrap_timed_method(
        env.get_predator_global_state,
        key="predator_global_state",
        timings=timings,
        counts=counts,
        device_name=device_name,
    )

    _sync(device_name)
    start = time.perf_counter()
    boid_rollouts, pred_rollouts = trainer.collect_rollouts(num_steps=rollout_steps)
    _sync(device_name)
    collect_time = time.perf_counter() - start

    _sync(device_name)
    start = time.perf_counter()
    trainer.train_step(boid_rollouts, pred_rollouts)
    _sync(device_name)
    train_time = time.perf_counter() - start

    return {
        "collect_rollouts_s": collect_time,
        "train_step_s": train_time,
        "timings_s": dict(timings),
        "counts": dict(counts),
    }


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    scalar_keys = ["collect_rollouts_s", "train_step_s"]
    timing_keys = sorted(
        {
            key
            for run in runs
            for key in run["timings_s"]
        }
    )

    summary = {
        "averages": {
            key: mean(float(run[key]) for run in runs)
            for key in scalar_keys
        },
        "stddev": {
            key: pstdev(float(run[key]) for run in runs)
            for key in scalar_keys
        },
        "timings_s": {
            key: mean(float(run["timings_s"].get(key, 0.0)) for run in runs)
            for key in timing_keys
        },
        "timing_stddev_s": {
            key: pstdev(float(run["timings_s"].get(key, 0.0)) for run in runs)
            for key in timing_keys
        },
        "counts": {
            key: int(mean(int(run["counts"].get(key, 0)) for run in runs))
            for key in timing_keys
        },
        "runs": runs,
    }
    return summary


def _format_text_report(
    *,
    config_path: Path,
    device_name: str,
    seed: int,
    rollout_steps: int,
    warmup_steps: int,
    repeats: int,
    summary: dict[str, object],
) -> str:
    averages = summary["averages"]
    timings = summary["timings_s"]
    counts = summary["counts"]
    collect_time = float(averages["collect_rollouts_s"])
    train_time = float(averages["train_step_s"])
    env_step_time = float(timings.get("env_step", 0.0))

    lines = [
        "Single-GPU Training Profiler",
        f"config: {config_path}",
        f"device: {device_name}",
        f"seed: {seed}",
        f"rollout_steps: {rollout_steps}",
        f"warmup_steps: {warmup_steps}",
        f"repeats: {repeats}",
        "",
        f"collect_rollouts: {collect_time:.6f}s",
        f"train_step: {train_time:.6f}s",
        "",
        "timed subcomponents:",
    ]

    ordered_keys = [
        "env_step",
        "physics_step",
        "boid_obs",
        "predator_obs",
        "rewards",
        "boid_global_state",
        "predator_global_state",
    ]
    for key in ordered_keys:
        total = float(timings.get(key, 0.0))
        if total == 0.0 and counts.get(key, 0) == 0:
            continue
        share = (total / collect_time * 100.0) if collect_time > 0 else 0.0
        per_call = (total / counts[key]) if counts.get(key, 0) else 0.0
        lines.append(
            f"  {key}: {total:.6f}s total | {share:5.1f}% of collect_rollouts | {per_call:.6f}s/call over {counts[key]} calls"
        )

    if env_step_time > 0:
        lines.extend(
            [
                "",
                "env.step breakdown:",
            ]
        )
        for key in ["predator_obs", "physics_step", "boid_obs", "rewards"]:
            total = float(timings.get(key, 0.0))
            share = (total / env_step_time * 100.0) if env_step_time > 0 else 0.0
            lines.append(f"  {key}: {total:.6f}s total | {share:5.1f}% of env_step")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    study_config = load_study_config(config_path)
    device_name = select_device(args.device)

    runs = [
        _profile_once(
            study_config=study_config,
            device_name=device_name,
            seed=args.seed,
            rollout_steps=args.rollout_steps,
            warmup_steps=args.warmup_steps,
        )
        for _ in range(args.repeats)
    ]
    summary = _aggregate_runs(runs)

    payload = {
        "config": str(config_path),
        "device": device_name,
        "gpu_name": torch.cuda.get_device_name(0) if device_name == "cuda" else None,
        "seed": args.seed,
        "rollout_steps": args.rollout_steps,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "summary": summary,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(
        _format_text_report(
            config_path=config_path,
            device_name=device_name,
            seed=args.seed,
            rollout_steps=args.rollout_steps,
            warmup_steps=args.warmup_steps,
            repeats=args.repeats,
            summary=summary,
        )
    )


if __name__ == "__main__":
    main()
