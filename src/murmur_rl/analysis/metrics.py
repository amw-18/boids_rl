from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np
import torch


def _count_connected_components(adjacency: np.ndarray) -> int:
    node_count = adjacency.shape[0]
    if node_count == 0:
        return 0

    visited = np.zeros(node_count, dtype=bool)
    components = 0
    for start in range(node_count):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            neighbors = np.where(adjacency[node] & ~visited)[0]
            if neighbors.size == 0:
                continue
            visited[neighbors] = True
            stack.extend(neighbors.tolist())
    return components


def compute_population_metrics(
    *,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    alive_mask: torch.Tensor,
    perception_radius: float,
    graph_radius: float,
    space_size: float,
    population_size: int,
    num_predators: int,
    captures_this_step: int,
    fringe_radius_fraction: float,
) -> dict[str, Any]:
    alive_positions = positions[alive_mask]
    alive_velocities = velocities[alive_mask]
    alive_count = int(alive_mask.sum().item())

    if alive_count == 0:
        return {
            "alive_count": 0,
            "survival_fraction": 0.0,
            "capture_rate_per_step": 0.0,
            "polarization": 0.0,
            "heading_alignment": 0.0,
            "connected_components": 0.0,
            "radial_spread": 0.0,
            "distance_to_center": 0.0,
            "fringe_fraction": 0.0,
            "outside_fraction": 0.0,
            "nearest_neighbor_distance_values": [],
            "local_density_values": [],
        }

    unit_velocities = alive_velocities / alive_velocities.norm(dim=-1, keepdim=True).clamp(min=1e-5)
    polarization = float(unit_velocities.mean(dim=0).norm().item())

    if alive_count == 1:
        nearest_neighbor = torch.zeros(1, device=positions.device)
        local_density = torch.zeros(1, device=positions.device)
        connected_components = 1
    else:
        pairwise = torch.cdist(alive_positions, alive_positions)
        diagonal = torch.eye(alive_count, dtype=torch.bool, device=positions.device)
        pairwise = torch.where(diagonal, torch.full_like(pairwise, float("inf")), pairwise)
        nearest_neighbor = pairwise.min(dim=1).values
        adjacency = pairwise < graph_radius
        connected_components = _count_connected_components(adjacency.cpu().numpy())
        local_density = (pairwise < perception_radius).sum(dim=1).float() / max(population_size, 1)

    center_of_mass = alive_positions.mean(dim=0)
    radial_distances = torch.norm(alive_positions - center_of_mass, dim=-1)
    radial_spread = float(radial_distances.mean().item()) if alive_count > 0 else 0.0
    world_center = torch.full((3,), space_size / 2.0, device=positions.device, dtype=alive_positions.dtype)
    distance_to_center = float(torch.norm(alive_positions - world_center, dim=-1).mean().item())
    outside_mask = ((alive_positions < 0.0) | (alive_positions > space_size)).any(dim=1)
    outside_fraction = float(outside_mask.float().mean().item()) if alive_count > 0 else 0.0

    max_radius = float(radial_distances.max().item()) if alive_count > 0 else 0.0
    if max_radius <= 0.0:
        fringe_fraction = 0.0
    else:
        fringe_fraction = float((radial_distances >= max_radius * fringe_radius_fraction).float().mean().item())

    return {
        "alive_count": alive_count,
        "survival_fraction": alive_count / max(population_size, 1),
        "capture_rate_per_step": captures_this_step / max(num_predators, 1) if num_predators > 0 else 0.0,
        "polarization": polarization,
        "heading_alignment": polarization,
        "connected_components": float(connected_components),
        "radial_spread": radial_spread,
        "distance_to_center": distance_to_center,
        "fringe_fraction": fringe_fraction,
        "outside_fraction": outside_fraction,
        "nearest_neighbor_distance_values": nearest_neighbor.cpu().tolist(),
        "local_density_values": local_density.cpu().tolist(),
    }


@dataclass
class EpisodeMetricAccumulator:
    population_size: int
    num_predators: int
    space_size: float
    perception_radius: float
    graph_radius: float
    fringe_radius_fraction: float
    histogram_bins: int
    curves: dict[str, list[float]] = field(
        default_factory=lambda: {
            "survival_fraction": [],
            "capture_rate_per_step": [],
            "polarization": [],
            "heading_alignment": [],
            "connected_components": [],
            "radial_spread": [],
            "distance_to_center": [],
            "fringe_fraction": [],
            "outside_fraction": [],
        }
    )
    nearest_neighbor_values: list[float] = field(default_factory=list)
    local_density_values: list[float] = field(default_factory=list)
    total_captures: int = 0

    def record_frame(self, positions: torch.Tensor, velocities: torch.Tensor, alive_mask: torch.Tensor, captures_this_step: int) -> None:
        frame_metrics = compute_population_metrics(
            positions=positions,
            velocities=velocities,
            alive_mask=alive_mask,
            perception_radius=self.perception_radius,
            graph_radius=self.graph_radius,
            space_size=self.space_size,
            population_size=self.population_size,
            num_predators=self.num_predators,
            captures_this_step=captures_this_step,
            fringe_radius_fraction=self.fringe_radius_fraction,
        )
        self.total_captures += captures_this_step
        for key in self.curves:
            self.curves[key].append(float(frame_metrics[key]))
        self.nearest_neighbor_values.extend(frame_metrics["nearest_neighbor_distance_values"])
        self.local_density_values.extend(frame_metrics["local_density_values"])

    def finalize(self) -> dict[str, Any]:
        nearest_range_max = math.sqrt(3.0) * self.space_size
        nearest_hist = np.histogram(
            np.asarray(self.nearest_neighbor_values, dtype=np.float64),
            bins=self.histogram_bins,
            range=(0.0, nearest_range_max),
        )
        density_hist = np.histogram(
            np.asarray(self.local_density_values, dtype=np.float64),
            bins=self.histogram_bins,
            range=(0.0, 1.0),
        )

        summary = {
            "episode_frames": len(self.curves["survival_fraction"]),
            "final_survival_fraction": self.curves["survival_fraction"][-1] if self.curves["survival_fraction"] else 0.0,
            "total_captures": int(self.total_captures),
        }
        for key, values in self.curves.items():
            array = np.asarray(values, dtype=np.float64)
            summary[f"mean_{key}"] = float(array.mean()) if array.size else 0.0
            summary[f"std_{key}"] = float(array.std()) if array.size else 0.0

        return {
            "summary": summary,
            "curves": self.curves,
            "histograms": {
                "nearest_neighbor_distance": {
                    "bin_edges": nearest_hist[1].tolist(),
                    "counts": nearest_hist[0].astype(int).tolist(),
                },
                "local_density": {
                    "bin_edges": density_hist[1].tolist(),
                    "counts": density_hist[0].astype(int).tolist(),
                },
            },
        }


def aggregate_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {}

    summary_keys = sorted(episodes[0]["summary"].keys())
    aggregated: dict[str, Any] = {}
    for key in summary_keys:
        values = np.asarray([episode["summary"][key] for episode in episodes], dtype=np.float64)
        aggregated[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return aggregated
