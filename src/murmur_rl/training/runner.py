from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch
import wandb

from murmur_rl.envs.vector_env import _HAS_TRITON
from murmur_rl.experiment import (
    DEFAULT_STUDY_CONFIG,
    append_registry_row,
    build_boid_brain,
    build_predator_brain,
    build_run_id,
    build_vector_env,
    ensure_registry,
    git_commit_hash,
    load_study_config,
    seed_everything,
    select_device,
    write_json,
)
from murmur_rl.experiment.runtime import training_manifest_payload
from murmur_rl.training.ppo import AlternatingCoevolutionTrainer


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_STUDY_CONFIG), help="Path to a frozen study config TOML.")
    parser.add_argument("--resume", type=str, default="", help="Path to a starling checkpoint to resume from.")
    parser.add_argument("--start-epoch", type=int, default=None, help="Epoch to start training from when resuming.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--checkpoints-dir", type=str, default="", help="Directory to save checkpoints. Defaults to checkpoints/<run-id>.")
    parser.add_argument("--run-id", type=str, default="", help="Optional run identifier used for manifests and artifact directories.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed for the run.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--registry", type=str, default="experiments/registry.csv", help="CSV registry path.")
    return parser.parse_args()


def _resolve_resume_start_epoch(resume_path: str, explicit_start_epoch: int | None) -> int:
    if explicit_start_epoch is not None:
        return explicit_start_epoch
    match = re.search(r"ep(\d+)", resume_path)
    if not match:
        return 1
    return int(match.group(1)) + 1


def _load_resume_weights(boid_brain, pred_brain, resume_path: str, device: torch.device) -> None:
    boid_checkpoint = Path(resume_path).resolve()
    pred_checkpoint = Path(str(boid_checkpoint).replace("starling_brain", "falcon_brain"))
    boid_brain.load_state_dict(torch.load(boid_checkpoint, map_location=device, weights_only=True))
    if pred_checkpoint.exists():
        pred_brain.load_state_dict(torch.load(pred_checkpoint, map_location=device, weights_only=True))
        print(f"Loaded FalconBrain checkpoint: {pred_checkpoint}")
    else:
        print(f"Warning: FalconBrain checkpoint not found at {pred_checkpoint}. Starting predator weights from scratch.")


def _maybe_init_wandb(study_config, args: argparse.Namespace, run_id: str) -> bool:
    if args.no_wandb:
        return False
    try:
        wandb.init(
            project=study_config.training.wandb_project,
            name=run_id if run_id else study_config.training.wandb_run_name,
            config={
                **study_config.to_dict(),
                "seed": args.seed,
                "run_id": run_id,
            },
            mode="online",
        )
        return True
    except Exception as exc:
        print(f"Warning: Failed to initialize W&B ({exc}). Running without W&B.")
        return False


def _maybe_compile_models(study_config, device_name: str, env, boid_brain, pred_brain):
    if study_config.training.compile_models:
        if device_name == "cuda" and not _HAS_TRITON:
            print("  brain compile skipped (CUDA requires Triton, not installed)")
        else:
            mode = "reduce-overhead" if _HAS_TRITON else "default"
            try:
                boid_brain = torch.compile(boid_brain, mode=mode)
                pred_brain = torch.compile(pred_brain, mode=mode)
                print(f"  brains compiled (mode={mode})")
            except Exception as exc:
                print(f"  brain compile skipped ({exc})")

    if study_config.training.compile_env:
        env.compile()
    return boid_brain, pred_brain


def main() -> None:
    args = parse_args()
    study_config = load_study_config(args.config)
    config_path = study_config.source_path or Path(args.config).resolve()
    device_name = select_device(args.device)
    device = torch.device(device_name)
    seed_everything(args.seed)

    run_id = args.run_id or build_run_id("train", study_config.study.id, args.seed)
    checkpoints_dir = Path(args.checkpoints_dir).resolve() if args.checkpoints_dir else (REPO_ROOT / "checkpoints" / run_id)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = REPO_ROOT / "experiments" / "runs" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "run_manifest.json"
    registry_path = Path(args.registry).resolve() if Path(args.registry).is_absolute() else (REPO_ROOT / args.registry).resolve()
    git_commit = git_commit_hash(REPO_ROOT)

    start_epoch = _resolve_resume_start_epoch(args.resume, args.start_epoch) if args.resume else 1
    manifest = training_manifest_payload(
        run_id=run_id,
        study_config=study_config,
        config_path=config_path,
        checkpoint_dir=checkpoints_dir,
        device_name=device_name,
        seed=args.seed,
        git_commit=git_commit,
        resume_path=args.resume,
        start_epoch=start_epoch,
        status="running",
    )
    write_json(manifest_path, manifest)

    use_wandb = _maybe_init_wandb(study_config, args, run_id)

    print(f"Starting training on {device}...")
    env = build_vector_env(study_config, device_name)
    dummy_obs_boids, dummy_obs_preds = env.reset()
    dummy_global_obs_boids = env.get_boid_global_state(dummy_obs_boids)
    dummy_global_obs_preds = env.get_predator_global_state(dummy_obs_preds)

    boid_brain = build_boid_brain(study_config, env)
    pred_brain = build_predator_brain(study_config, env)

    if args.resume:
        print(f"Resuming from checkpoints: {args.resume}")
        try:
            _load_resume_weights(boid_brain, pred_brain, args.resume, device)
            print(f"Resuming at epoch {start_epoch}")
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failure_reason"] = f"resume-load-error: {exc}"
            write_json(manifest_path, manifest)
            raise SystemExit(f"Failed to load checkpoint {args.resume}: {exc}") from exc

    if device_name == "cuda":
        torch.set_float32_matmul_precision("high")

    boid_brain, pred_brain = _maybe_compile_models(study_config, device_name, env, boid_brain, pred_brain)
    print("torch.compile setup complete")

    trainer = AlternatingCoevolutionTrainer(
        env=env,
        boid_brain=boid_brain,
        pred_brain=pred_brain,
        device=device,
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

    offset = (study_config.training.stacked_frames - 1) * env.obs_dim
    col_predator_dist = 11 + offset
    col_local_density = 4 + offset

    ensure_registry(registry_path)

    try:
        for epoch in range(start_epoch, study_config.training.num_epochs + 1):
            boid_rollouts, pred_rollouts = trainer.collect_rollouts(num_steps=study_config.training.rollout_steps)

            mean_predator_dist_norm = boid_rollouts["obs"][:, :, col_predator_dist].mean().item()
            actual_predator_dist = mean_predator_dist_norm * (study_config.environment.space_size / 2.0)

            mean_local_density_norm = boid_rollouts["obs"][:, :, col_local_density].mean().item()
            actual_social_neighbors = mean_local_density_norm * study_config.environment.num_agents

            progress = min(1.0, (epoch - 1) / 1000.0)
            trainer.ent_coef = study_config.training.ent_coef * (1.0 - progress)
            metrics = trainer.train_step(boid_rollouts, pred_rollouts)

            if use_wandb:
                try:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "run_id": run_id,
                            "seed": args.seed,
                            "boid_loss/policy_loss": metrics["boids"][0],
                            "boid_loss/value_loss": metrics["boids"][1],
                            "boid_loss/entropy": metrics["boids"][2],
                            "boid_loss/explained_variance": metrics["boids"][4],
                            "boid_reward/mean_gae_return": metrics["boids"][3],
                            "pred_loss/policy_loss": metrics["preds"][0],
                            "pred_loss/value_loss": metrics["preds"][1],
                            "pred_loss/entropy": metrics["preds"][2],
                            "pred_loss/explained_variance": metrics["preds"][4],
                            "pred_reward/mean_gae_return": metrics["preds"][3],
                            "biology/mean_predator_distance": actual_predator_dist,
                            "biology/mean_social_neighbors": actual_social_neighbors,
                        }
                    )
                except Exception as exc:
                    print(f"Warning: W&B log failed at epoch {epoch} ({exc})")

            if epoch % 50 == 0 or epoch == start_epoch:
                b_ploss, b_vloss, b_ent, b_ret, _ = metrics["boids"]
                p_ploss, p_vloss, p_ent, p_ret, _ = metrics["preds"]
                print(f"Epoch {epoch:04d} | Cohort: {actual_social_neighbors:>4.1f} | EvasionDist: {actual_predator_dist:>5.1f}m")
                print(f"  [BOIDS] Ret: {b_ret:>7.4f} | Ent: {b_ent:>6.4f} | VLoss: {b_vloss:>7.4f} | Ploss: {b_ploss:>7.4f}")
                print(f"  [PREDS] Ret: {p_ret:>7.4f} | Ent: {p_ent:>6.4f} | VLoss: {p_vloss:>7.4f} | Ploss: {p_ploss:>7.4f}")

            if epoch % study_config.training.checkpoint_interval == 0:
                boid_checkpoint_path = checkpoints_dir / f"starling_brain_ep{epoch}.pth"
                pred_checkpoint_path = checkpoints_dir / f"falcon_brain_ep{epoch}.pth"
                torch.save(boid_brain.state_dict(), boid_checkpoint_path)
                torch.save(pred_brain.state_dict(), pred_checkpoint_path)
                manifest["latest_epoch"] = epoch
                manifest["latest_boid_checkpoint"] = str(boid_checkpoint_path)
                manifest["latest_predator_checkpoint"] = str(pred_checkpoint_path)
                write_json(manifest_path, manifest)
                print(f"Saved Checkpoints: {boid_checkpoint_path} & {pred_checkpoint_path}")

            del boid_rollouts, pred_rollouts
            if device_name == "mps":
                gc.collect()
                torch.mps.empty_cache()

        manifest["status"] = "completed"
        write_json(manifest_path, manifest)
        append_registry_row(
            registry_path,
            {
                "created_at_utc": manifest["created_at_utc"],
                "run_id": run_id,
                "run_type": "training",
                "study_id": study_config.study.id,
                "config_path": str(config_path),
                "git_commit": git_commit,
                "seed": args.seed,
                "status": "completed",
                "artifact_path": str(manifest_path),
                "notes": study_config.study.description,
            },
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = str(exc)
        write_json(manifest_path, manifest)
        append_registry_row(
            registry_path,
            {
                "created_at_utc": manifest["created_at_utc"],
                "run_id": run_id,
                "run_type": "training",
                "study_id": study_config.study.id,
                "config_path": str(config_path),
                "git_commit": git_commit,
                "seed": args.seed,
                "status": "failed",
                "artifact_path": str(manifest_path),
                "notes": str(exc),
            },
        )
        raise
    finally:
        if use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

    print("Training Complete. Final model saved.")


if __name__ == "__main__":
    main()
