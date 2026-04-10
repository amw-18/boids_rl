import pytest
import torch

from murmur_rl.analysis.metrics import EpisodeMetricAccumulator, compute_population_metrics


def test_population_metrics_on_simple_line_swarm():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    velocities = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    alive_mask = torch.tensor([True, True, True, True])

    metrics = compute_population_metrics(
        positions=positions,
        velocities=velocities,
        alive_mask=alive_mask,
        perception_radius=1.5,
        graph_radius=1.5,
        space_size=10.0,
        population_size=4,
        num_predators=2,
        captures_this_step=2,
        fringe_radius_fraction=0.9,
    )

    assert metrics["survival_fraction"] == pytest.approx(1.0)
    assert metrics["polarization"] == pytest.approx(1.0)
    assert metrics["connected_components"] == pytest.approx(1.0)
    assert metrics["capture_rate_per_step"] == pytest.approx(1.0)
    assert metrics["nearest_neighbor_distance_values"] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert metrics["local_density_values"] == pytest.approx([0.25, 0.5, 0.5, 0.25])
    assert metrics["radial_spread"] == pytest.approx(1.0)
    assert metrics["fringe_fraction"] == pytest.approx(0.5)


def test_metric_accumulator_emits_curves_and_histograms():
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    velocities = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    alive_mask = torch.tensor([True, True, True, True])

    accumulator = EpisodeMetricAccumulator(
        population_size=4,
        num_predators=2,
        space_size=10.0,
        perception_radius=1.5,
        graph_radius=1.5,
        fringe_radius_fraction=0.9,
        histogram_bins=5,
    )
    accumulator.record_frame(positions, velocities, alive_mask, captures_this_step=0)
    accumulator.record_frame(positions, velocities, alive_mask, captures_this_step=2)
    result = accumulator.finalize()

    assert result["summary"]["episode_frames"] == 2
    assert result["summary"]["final_survival_fraction"] == pytest.approx(1.0)
    assert result["summary"]["total_captures"] == 2
    assert result["summary"]["mean_capture_rate_per_step"] == pytest.approx(0.5)
    assert len(result["curves"]["survival_fraction"]) == 2
    assert sum(result["histograms"]["nearest_neighbor_distance"]["counts"]) == 8
    assert sum(result["histograms"]["local_density"]["counts"]) == 8
