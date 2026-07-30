from pathlib import Path

import numpy as np

from src.core.types import (
    AlignmentResult,
    AlignmentStatus,
    PointCloud,
    ReconstructionResult,
)
from src.experiments.runner import ExperimentConfig, ExperimentRunner


def runner_without_initialization(config: ExperimentConfig) -> ExperimentRunner:
    runner = object.__new__(ExperimentRunner)
    runner.config = config
    return runner


def test_find_videos_can_select_a_small_reprocessing_subset(tmp_path: Path):
    sizes = {
        "large.MP4": 30,
        "small.MP4": 10,
        "medium.MP4": 20,
    }
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(b"x" * size)

    config = ExperimentConfig(
        name="selection",
        model="must3r",
        input_folder=tmp_path,
        output_folder=tmp_path / "out",
        video_names=["medium", "small.MP4"],
        max_videos=1,
    )

    selected = runner_without_initialization(config)._find_videos()

    assert [path.name for path in selected] == ["small.MP4"]


def test_global_alignment_is_applied_to_each_source_without_merging():
    source = ReconstructionResult(
        pointcloud=PointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            confidence=np.array([0.75], dtype=np.float32),
        ),
        window_metadata={"window_id": 4},
    )
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 20.0, 30.0]
    alignment = AlignmentResult(
        transform=transform,
        enu_to_ecef_transform=np.eye(4),
        scale=1.0,
        method="fixture",
        status=AlignmentStatus.ALIGNED,
        inlier_count=10,
        correspondence_count=10,
        rmse_m=0.1,
        horizontal_rmse_m=0.1,
        vertical_rmse_m=0.0,
        anchor_wgs84=(37.0, -121.0, 100.0),
        anchor_ecef=np.ones(3),
    )

    transformed = ExperimentRunner._apply_global_alignment(source, alignment)

    np.testing.assert_allclose(transformed.pointcloud.points, [[11.0, 22.0, 33.0]])
    np.testing.assert_array_equal(transformed.pointcloud.confidence, [0.75])
    assert transformed.pointcloud.is_metric
    assert transformed.window_metadata == {"window_id": 4}


def test_single_source_uses_future_proof_capture_granularity():
    source = ReconstructionResult(
        pointcloud=PointCloud(points=np.zeros((1, 3))),
        window_metadata={"window_id": 0},
    )

    package_sources = ExperimentRunner._build_package_sources([source], frame_count=2)

    assert package_sources[0].kind == "capture"
    assert package_sources[0].frame_start == 0
    assert package_sources[0].frame_end == 1
