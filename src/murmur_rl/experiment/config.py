from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDY_CONFIG = REPO_ROOT / "configs" / "studies" / "v1_shaped_current.toml"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _select_dataclass_fields(raw: dict[str, Any], data_cls: type) -> dict[str, Any]:
    valid_fields = {item.name for item in fields(data_cls)}
    return {key: value for key, value in raw.items() if key in valid_fields}


def _tupled(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class StudyMetadata:
    id: str
    name: str
    claim: str
    description: str
    notes: str = ""
    default_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentConfig:
    num_agents: int = 100
    num_predators: int = 10
    space_size: float = 50.0
    perception_radius: float = 15.0
    base_speed: float = 5.0
    max_turn_angle: float = 0.5
    max_force: float = 2.0
    min_speed: float = 2.5
    dt: float = 0.1
    max_steps: int = 500
    gamma: float = 0.99
    pbrs_k: float = 1.0
    pbrs_c: float = 1.0
    curriculum_enabled: bool = True
    predator_catch_radius_start: float = 2.0
    predator_catch_radius_end: float = 0.5
    predator_catch_radius_decay_steps: int = 5_000_000
    predator_visual_noise_variance: float = 5.0
    predator_sprint_multiplier: float = 1.5
    predator_turn_multiplier: float = 1.5
    predator_cooldown_duration: int = 50
    predator_max_stamina: float = 100.0
    predator_sprint_drain: float = 1.0
    predator_recovery_rate: float = 0.5
    survival_reward: float = 0.1
    collision_penalty: float = 2.0
    death_penalty: float = -100.0
    predator_catch_reward: float = 10.0
    predator_hunger_penalty: float = -0.05


@dataclass(frozen=True)
class TrainingConfig:
    rollout_steps: int = 500
    num_epochs: int = 30_000
    actor_lr: float = 3e-4
    predator_actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015
    update_epochs: int = 4
    batch_size: int = 1024
    stacked_frames: int = 4
    boid_hidden_size: int = 128
    predator_hidden_size: int = 256
    critic_hidden_size: int = 512
    checkpoint_interval: int = 500
    compile_models: bool = True
    compile_env: bool = True
    wandb_project: str = "murmur_rl"
    wandb_run_name: str = "co-evolution run 2"


@dataclass(frozen=True)
class EvaluationConfig:
    num_episodes: int = 3
    episode_steps: int = 500
    render_frames: int = 1800
    seed_offset: int = 1000


@dataclass(frozen=True)
class MetricSuiteConfig:
    distance_graph_radius: float = 15.0
    fringe_radius_fraction: float = 0.9
    histogram_bins: int = 20


@dataclass(frozen=True)
class StudyConfig:
    study: StudyMetadata
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    metrics: MetricSuiteConfig
    source_path: Path | None = None

    def to_env_kwargs(self) -> dict[str, Any]:
        return asdict(self.environment)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "study": asdict(self.study),
            "environment": asdict(self.environment),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "metrics": asdict(self.metrics),
        }
        if self.source_path is not None:
            payload["source_path"] = str(self.source_path)
        return payload


@dataclass(frozen=True)
class PolicyConfig:
    type: str
    checkpoint: str = ""
    deterministic: bool = True
    label: str = ""


@dataclass(frozen=True)
class AnalysisExperimentConfig:
    id: str
    name: str
    description: str
    study_config_path: Path
    num_episodes: int = 3
    seeds: tuple[int, ...] = (0,)
    prey_policy: PolicyConfig = field(default_factory=lambda: PolicyConfig(type="random", label="random-prey"))
    predator_policy: PolicyConfig = field(default_factory=lambda: PolicyConfig(type="random", label="random-predator"))
    environment_overrides: dict[str, Any] = field(default_factory=dict)
    output_subdir: str = "analysis"
    source_path: Path | None = None

    def load_study_config(self) -> StudyConfig:
        base_study = load_study_config(self.study_config_path)
        if not self.environment_overrides:
            return base_study
        return apply_environment_overrides(base_study, self.environment_overrides)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "study_config_path": str(self.study_config_path),
            "num_episodes": self.num_episodes,
            "seeds": list(self.seeds),
            "prey_policy": asdict(self.prey_policy),
            "predator_policy": asdict(self.predator_policy),
            "environment_overrides": self.environment_overrides,
            "output_subdir": self.output_subdir,
        }
        if self.source_path is not None:
            payload["source_path"] = str(self.source_path)
        return payload


def load_study_config(path: str | Path | None = None) -> StudyConfig:
    config_path = Path(path or DEFAULT_STUDY_CONFIG).resolve()
    raw = _load_toml(config_path)

    study_raw = dict(raw.get("study", {}))
    if "default_tags" in study_raw:
        study_raw["default_tags"] = _tupled(study_raw["default_tags"])

    config = StudyConfig(
        study=StudyMetadata(**_select_dataclass_fields(study_raw, StudyMetadata)),
        environment=EnvironmentConfig(**_select_dataclass_fields(raw.get("environment", {}), EnvironmentConfig)),
        training=TrainingConfig(**_select_dataclass_fields(raw.get("training", {}), TrainingConfig)),
        evaluation=EvaluationConfig(**_select_dataclass_fields(raw.get("evaluation", {}), EvaluationConfig)),
        metrics=MetricSuiteConfig(**_select_dataclass_fields(raw.get("metrics", {}), MetricSuiteConfig)),
        source_path=config_path,
    )
    return config


def load_analysis_experiment_config(path: str | Path) -> AnalysisExperimentConfig:
    config_path = Path(path).resolve()
    raw = _load_toml(config_path)
    study_config_value = raw.get("study_config") or raw.get("experiment", {}).get("study_config")
    if not study_config_value:
        raise ValueError("Analysis experiment config must define study_config.")
    study_path = Path(study_config_value)
    if not study_path.is_absolute():
        study_path = (config_path.parent / study_path).resolve()

    prey_policy = PolicyConfig(**_select_dataclass_fields(raw.get("prey_policy", {}), PolicyConfig))
    predator_policy = PolicyConfig(**_select_dataclass_fields(raw.get("predator_policy", {}), PolicyConfig))
    overrides = dict(raw.get("environment_overrides", {}))

    return AnalysisExperimentConfig(
        id=raw["experiment"]["id"],
        name=raw["experiment"]["name"],
        description=raw["experiment"]["description"],
        study_config_path=study_path,
        num_episodes=raw.get("evaluation", {}).get("num_episodes", 3),
        seeds=tuple(raw.get("evaluation", {}).get("seeds", [0])),
        prey_policy=prey_policy,
        predator_policy=predator_policy,
        environment_overrides=overrides,
        output_subdir=raw.get("output", {}).get("subdir", "analysis"),
        source_path=config_path,
    )


def apply_environment_overrides(study_config: StudyConfig, overrides: dict[str, Any]) -> StudyConfig:
    allowed = {item.name for item in fields(EnvironmentConfig)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown environment override fields: {names}")

    updated_env = replace(study_config.environment, **overrides)
    return replace(study_config, environment=updated_env)
