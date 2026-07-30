from __future__ import annotations

import numpy as np
import pytest

from src.alignment.window_aligner import WindowAligner
from src.core.types import CameraPoses, PointCloud, ReconstructionResult
from src.experiments.metrics import MetricsCalculator


def _chunk(window_id: int, start: int, points: np.ndarray) -> ReconstructionResult:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 3, axis=0)
    poses[:, 0, 3] = np.arange(start, start + 3)
    return ReconstructionResult(
        pointcloud=PointCloud(
            points=points,
            colors=np.full(points.shape, 100, dtype=np.uint8),
            confidence=np.full(len(points), window_id + 1, dtype=np.float32),
            is_metric=True,
        ),
        poses=CameraPoses(
            poses=poses,
            frame_indices=np.arange(start, start + 3),
            timestamps=np.arange(start, start + 3, dtype=float),
        ),
        window_metadata={"window_id": window_id},
    )


def test_align_chunks_preserves_source_units_without_concatenating():
    chunks = [
        _chunk(0, 0, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
        _chunk(1, 3, np.array([[2.0, 0.0, 0.0]])),
    ]
    original_points = chunks[0].pointcloud.points
    aligner = WindowAligner({"window_alignment_method": "none"})

    transformed, poses, metadata = aligner.align_chunks(chunks, is_metric=True)

    assert len(transformed) == 2
    assert [len(chunk.pointcloud) for chunk in transformed] == [2, 1]
    assert transformed[0].pointcloud.points is not original_points
    assert chunks[0] is transformed[0]
    assert transformed[1].window_metadata["window_id"] == 1
    assert poses is not None
    assert len(poses) == 6
    assert metadata["chunks"][1]["method"] == "none"
    assert metadata["chunks"][0]["rotation"] == np.eye(3).tolist()
    assert metadata["chunks"][1]["translation"] == [0.0, 0.0, 0.0]


def test_align_and_merge_remains_compatible():
    chunks = [
        _chunk(0, 0, np.array([[0.0, 0.0, 0.0]])),
        _chunk(1, 3, np.array([[1.0, 0.0, 0.0]])),
    ]

    merged, poses, _ = WindowAligner(
        {"window_alignment_method": "none"}
    ).align_and_merge(chunks, is_metric=True)

    assert len(merged) == 2
    assert merged.confidence.tolist() == [1.0, 2.0]
    assert poses is not None


def test_chunk_metrics_match_merged_point_metrics():
    chunks = [
        _chunk(0, 0, np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])),
        _chunk(1, 3, np.array([[2.0, 4.0, 6.0]])),
    ]
    clouds = [chunk.pointcloud for chunk in chunks]
    merged = PointCloud(
        points=np.concatenate([cloud.points for cloud in clouds]),
        colors=np.concatenate([cloud.colors for cloud in clouds]),
        confidence=np.concatenate([cloud.confidence for cloud in clouds]),
        is_metric=True,
    )
    calculator = MetricsCalculator()

    chunk_metrics = calculator.compute_chunks(clouds, None, None)
    merged_metrics = calculator.compute_all(merged, None, None)

    for key in (
        "point_count",
        "point_density",
        "bounding_box_volume_m3",
        "confidence_mean",
        "confidence_std",
        "confidence_min",
        "confidence_max",
    ):
        assert chunk_metrics[key] == pytest.approx(merged_metrics[key])
