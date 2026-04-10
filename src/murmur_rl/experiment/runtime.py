from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import json
import random
import subprocess

import numpy as np
import torch

from murmur_rl.agents.starling import FalconBrain, StarlingBrain
from murmur_rl.envs.vector_env import VectorMurmurationEnv
from murmur_rl.experiment.config import AnalysisExperimentConfig, StudyConfig


REGISTRY_HEADERS = [
    "created_at_utc",
    "run_id",
    "run_type",
    "study_id",
    "config_path",
    "git_commit",
    "seed",
    "status",
    "artifact_path",
    "notes",
]


def select_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vector_env(study_config: StudyConfig, device_name: str) -> VectorMurmurationEnv:
    env_kwargs = study_config.to_env_kwargs()
    env_kwargs["device"] = device_name
    return VectorMurmurationEnv(**env_kwargs)


def build_boid_brain(study_config: StudyConfig, env: VectorMurmurationEnv) -> StarlingBrain:
    training = study_config.training
    return StarlingBrain(
        obs_dim=env.obs_dim,
        global_obs_dim=env.global_obs_dim,
        action_dim=env.action_dim,
        hidden_size=training.boid_hidden_size,
        critic_hidden_size=training.critic_hidden_size,
        stacked_frames=training.stacked_frames,
    )


def build_predator_brain(study_config: StudyConfig, env: VectorMurmurationEnv) -> FalconBrain:
    training = study_config.training
    return FalconBrain(
        obs_dim=env.pred_obs_dim,
        global_obs_dim=env.pred_global_obs_dim,
        action_dim=env.action_dim,
        hidden_size=training.predator_hidden_size,
        critic_hidden_size=training.critic_hidden_size,
        stacked_frames=training.stacked_frames,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def build_run_id(prefix: str, study_id: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    normalized_study_id = study_id.replace("_", "-")
    return f"{prefix}-{normalized_study_id}-s{seed}-{timestamp}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def ensure_registry(registry_path: Path) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        return
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_HEADERS)
        writer.writeheader()


def append_registry_row(registry_path: Path, row: dict[str, Any]) -> None:
    ensure_registry(registry_path)
    serialized_row = {key: row.get(key, "") for key in REGISTRY_HEADERS}
    with registry_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_HEADERS)
        writer.writerow(serialized_row)


def training_manifest_payload(
    *,
    run_id: str,
    study_config: StudyConfig,
    config_path: Path,
    checkpoint_dir: Path,
    device_name: str,
    seed: int,
    git_commit: str,
    resume_path: str,
    start_epoch: int,
    status: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": "training",
        "status": status,
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit,
        "device": device_name,
        "seed": seed,
        "resume_path": resume_path,
        "start_epoch": start_epoch,
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "study": study_config.to_dict(),
    }


def analysis_manifest_payload(
    *,
    run_id: str,
    experiment_config: AnalysisExperimentConfig,
    study_config: StudyConfig,
    config_path: Path | None,
    output_dir: Path,
    device_name: str,
    git_commit: str,
    status: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_type": "analysis",
        "status": status,
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit,
        "device": device_name,
        "config_path": str(config_path) if config_path else "",
        "output_dir": str(output_dir),
        "experiment": experiment_config.to_dict(),
        "study": study_config.to_dict(),
    }
