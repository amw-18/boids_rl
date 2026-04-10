from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

import torch

from murmur_rl.experiment.config import AnalysisExperimentConfig, PolicyConfig, StudyConfig
from murmur_rl.experiment.runtime import build_boid_brain, build_predator_brain


class BasePolicyController:
    def reset(self, obs: torch.Tensor) -> None:
        self.observe(obs)

    def act(self, env, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def observe(self, obs: torch.Tensor, dones: torch.Tensor | None = None) -> None:
        return None


class RandomPolicyController(BasePolicyController):
    def act(self, env, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[0] == 0:
            return torch.zeros((0, env.action_dim), device=env.device)
        return (torch.rand((obs.shape[0], env.action_dim), device=env.device) * 2.0) - 1.0


class ZeroPolicyController(BasePolicyController):
    def act(self, env, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((obs.shape[0], env.action_dim), device=env.device)


class NeuralPolicyController(BasePolicyController):
    def __init__(self, brain, stacked_frames: int, deterministic: bool):
        self.brain = brain
        self.stacked_frames = stacked_frames
        self.deterministic = deterministic
        self.rolling_obs: torch.Tensor | None = None

    def reset(self, obs: torch.Tensor) -> None:
        if obs.shape[0] == 0:
            self.rolling_obs = torch.zeros((0, self.stacked_frames, 0), device=obs.device)
            return
        self.rolling_obs = obs.unsqueeze(1).repeat(1, self.stacked_frames, 1)

    def act(self, env, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[0] == 0:
            return torch.zeros((0, env.action_dim), device=env.device)
        if self.rolling_obs is None:
            self.reset(obs)
        flat_obs = self.rolling_obs.view(obs.shape[0], -1)
        with torch.no_grad():
            if self.deterministic:
                return self.brain.get_deterministic_action(flat_obs)
            actions, _, _, _ = self.brain.get_action_and_value(flat_obs)
            return actions

    def observe(self, obs: torch.Tensor, dones: torch.Tensor | None = None) -> None:
        if obs.shape[0] == 0:
            self.rolling_obs = torch.zeros((0, self.stacked_frames, 0), device=obs.device)
            return
        if self.rolling_obs is None:
            self.reset(obs)
            return
        self.rolling_obs = torch.cat([self.rolling_obs[:, 1:, :], obs.unsqueeze(1)], dim=1)
        if dones is not None and dones.any():
            dead_mask = dones.bool()
            self.rolling_obs[dead_mask] = obs[dead_mask].unsqueeze(1).repeat(1, self.stacked_frames, 1)


class HeuristicBoidsPolicyController(BasePolicyController):
    def __init__(
        self,
        *,
        separation_weight: float = 2.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 0.8,
        predator_weight: float = 4.0,
        center_weight: float = 0.35,
        separation_radius_fraction: float = 0.35,
    ):
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.predator_weight = predator_weight
        self.center_weight = center_weight
        self.separation_radius_fraction = separation_radius_fraction

    def act(self, env, obs: torch.Tensor) -> torch.Tensor:
        if env.n_agents == 0:
            return torch.zeros((0, env.action_dim), device=env.device)

        pos = env.physics.positions
        vel = env.physics.velocities
        alive = env.physics.alive_mask
        device = pos.device

        pairwise = torch.cdist(pos, pos)
        diagonal = torch.eye(env.n_agents, dtype=torch.bool, device=device)
        live_pair = alive.unsqueeze(0) & alive.unsqueeze(1) & ~diagonal
        pairwise = torch.where(live_pair, pairwise, torch.full_like(pairwise, float("inf")))
        neighbor_mask = pairwise < env.perception_radius
        separation_mask = pairwise < (env.perception_radius * self.separation_radius_fraction)

        neighbor_mask_f = neighbor_mask.float()
        neighbor_counts = neighbor_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        relative_pos = pos.unsqueeze(1) - pos.unsqueeze(0)

        separation = (
            relative_pos / pairwise.unsqueeze(-1).clamp(min=1e-5).pow(2)
        ) * separation_mask.unsqueeze(-1).float()
        separation = separation.sum(dim=1)

        alignment = ((neighbor_mask_f @ vel) / neighbor_counts) - vel
        cohesion = ((neighbor_mask_f @ pos) / neighbor_counts) - pos
        has_neighbors = neighbor_mask.any(dim=1, keepdim=True) & alive.unsqueeze(1)
        alignment = torch.where(has_neighbors, alignment, torch.zeros_like(alignment))
        cohesion = torch.where(has_neighbors, cohesion, torch.zeros_like(cohesion))

        predator_avoidance = torch.zeros_like(pos)
        threat_strength = torch.zeros((env.n_agents, 1), device=device)
        if env.num_predators > 0:
            predator_pos = env.physics.predator_position
            predator_dist = torch.cdist(pos, predator_pos)
            predator_mask = predator_dist < (env.perception_radius * 2.0)
            if predator_mask.any():
                predator_rel = pos.unsqueeze(1) - predator_pos.unsqueeze(0)
                predator_avoidance = (
                    predator_rel / predator_dist.unsqueeze(-1).clamp(min=1e-5).pow(2)
                ) * predator_mask.unsqueeze(-1).float()
                predator_avoidance = predator_avoidance.sum(dim=1)
                nearest_predator = predator_dist.min(dim=1, keepdim=True).values
                threat_strength = (1.0 - (nearest_predator / (env.perception_radius * 2.0))).clamp(min=0.0, max=1.0)

        center = torch.full_like(pos, env.space_size / 2.0)
        center_pull = center - pos

        desired = (
            self.separation_weight * separation
            + self.alignment_weight * alignment
            + self.cohesion_weight * cohesion
            + self.predator_weight * predator_avoidance
            + self.center_weight * center_pull
        )

        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        forward = vel / speed
        up = env.physics.up_vectors
        right = torch.linalg.cross(forward, up, dim=-1)
        right = right / right.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        desired_norm = desired.norm(dim=-1, keepdim=True)
        desired_direction = torch.where(
            desired_norm > 1e-5,
            desired / desired_norm,
            forward,
        )

        desired_forward = (desired_direction * forward).sum(dim=-1, keepdim=True)
        desired_right = (desired_direction * right).sum(dim=-1, keepdim=True)
        desired_up = (desired_direction * up).sum(dim=-1, keepdim=True)

        thrust = torch.where(threat_strength > 0.0, torch.ones_like(desired_forward), desired_forward).clamp(-1.0, 1.0)
        roll = desired_right.clamp(-1.0, 1.0)
        pitch = desired_up.clamp(-1.0, 1.0)
        actions = torch.cat([thrust, roll, pitch], dim=1)
        return torch.where(alive.unsqueeze(1), actions, torch.zeros_like(actions))


def _resolve_checkpoint_path(checkpoint: str, base_path: Path | None) -> Path:
    resolved = Path(checkpoint)
    if resolved.is_absolute() or base_path is None:
        return resolved
    return (base_path / resolved).resolve()


def build_policy_controller(
    *,
    role: str,
    policy: PolicyConfig,
    study_config: StudyConfig,
    experiment_config: AnalysisExperimentConfig,
    env,
    device_name: str,
):
    policy_type = policy.type.lower()
    if policy_type == "random":
        return RandomPolicyController()
    if policy_type in {"idle", "zero"}:
        return ZeroPolicyController()
    if policy_type == "heuristic_boids":
        if role != "prey":
            raise ValueError("heuristic_boids is only implemented for prey controllers.")
        return HeuristicBoidsPolicyController()
    if policy_type != "checkpoint":
        raise ValueError(f"Unsupported policy type: {policy.type}")

    if not policy.checkpoint:
        raise ValueError(f"Policy '{policy.label or policy.type}' requires a checkpoint path.")

    checkpoint_path = _resolve_checkpoint_path(
        policy.checkpoint,
        experiment_config.source_path.parent if experiment_config.source_path else None,
    )
    map_location = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    if role == "prey":
        brain = build_boid_brain(study_config, env).to(map_location)
    else:
        brain = build_predator_brain(study_config, env).to(map_location)
    brain.load_state_dict(checkpoint, strict=False)
    brain.eval()
    return NeuralPolicyController(
        brain=brain,
        stacked_frames=study_config.training.stacked_frames,
        deterministic=policy.deterministic,
    )
