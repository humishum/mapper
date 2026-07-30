from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np

from src.alignment.gps_aligner import _enu_to_ecef_transform
from src.core.types import (
    AlignmentResult,
    AlignmentStatus,
    ARTIFACT_LOCAL_ENU_FRAME,
    CameraPoses,
    GPSTrack,
    IMUData,
    PointCloud,
)
from src.publisher.package import (
    PackageIdentity,
    PackageSource,
    ReconstructionPackagePublisher,
)
from src.publisher import CopcPublisher, CopcPublisherError
from viewer.backend.domain.package import CaptureMetadata, Producer


class FakeCopcPublisher:
    """Exercise package orchestration without invoking the external converter."""

    def publish(self, inputs, output):
        inputs = [Path(path) for path in inputs]
        point_count = 0
        minimums = []
        maximums = []
        for path in inputs:
            with laspy.open(path) as reader:
                point_count += reader.header.point_count
                minimums.append(reader.header.mins)
                maximums.append(reader.header.maxs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake independently tested COPC boundary")
        return SimpleNamespace(
            output=output,
            point_count=point_count,
            file_bytes=output.stat().st_size,
            sha256="0" * 64,
            wall_seconds=0.125,
            converter_version="0.11.0",
            structure=SimpleNamespace(
                bounds_min=tuple(np.minimum.reduce(minimums)),
                bounds_max=tuple(np.maximum.reduce(maximums)),
            ),
        )


def producer() -> Producer:
    return Producer(
        model_name="fixture",
        model_config={"window_size": 2},
        capabilities={"outputs_confidence": True},
        git_status="clean",
        adapter_name="fixture_adapter",
        adapter_version="1.0.0",
        publisher_name="copc-converter",
        publisher_version="0.11.0",
    )


def aligned_result() -> AlignmentResult:
    anchor = (37.0, -121.0, 100.0)
    anchor_ecef = GPSTrack(
        latitudes=np.array([anchor[0]]),
        longitudes=np.array([anchor[1]]),
        altitudes=np.array([anchor[2]]),
    ).to_ecef()[0]
    return AlignmentResult(
        transform=np.eye(4),
        enu_to_ecef_transform=_enu_to_ecef_transform(anchor, anchor_ecef),
        scale=1.0,
        method="gps_robust_weighted_umeyama",
        status=AlignmentStatus.ALIGNED,
        inlier_count=3,
        correspondence_count=3,
        rmse_m=0.1,
        horizontal_rmse_m=0.08,
        vertical_rmse_m=0.06,
        anchor_wgs84=anchor,
        anchor_ecef=anchor_ecef,
    )


def test_publishes_complete_validated_and_cataloged_package(tmp_path: Path):
    cloud = PointCloud(
        points=np.array(
            [[0.001, 0.001, 0.001], [0.009, 0.009, 0.009], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        ),
        colors=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
        confidence=np.array([0.1, 0.9, 0.5], dtype=np.float32),
        is_metric=True,
    )
    poses = CameraPoses(
        poses=np.repeat(np.eye(4)[None], 2, axis=0),
        timestamps=np.array([0.0, 1.0]),
        intrinsics=np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]]),
        frame_indices=np.array([0, 1]),
        coordinate_frame=ARTIFACT_LOCAL_ENU_FRAME,
    )
    gps = GPSTrack(
        latitudes=np.array([37.0, 37.00001]),
        longitudes=np.array([-121.0, -121.00001]),
        altitudes=np.array([100.0, 100.1]),
        timestamps=np.array([0.0, 1.0]),
        fixes=np.array([3, 3]),
        position_dops=np.array([0.8, 0.9]),
    )
    imu = IMUData(
        accelerometer=np.zeros((2, 3)),
        gyroscope=np.zeros((2, 3)),
        timestamps=np.array([0.0, 1.0]),
    )
    package_root = tmp_path / "package"
    catalog = tmp_path / "catalog.sqlite3"

    result = ReconstructionPackagePublisher(FakeCopcPublisher()).publish(
        package_root,
        identity=PackageIdentity("cap_test", "run_test", "art_test"),
        sources=[
            PackageSource(
                cloud,
                kind="window",
                name="window_000",
                frame_start=0,
                frame_end=1,
                frame_indices=[0, 1],
            )
        ],
        alignment=aligned_result(),
        poses=poses,
        gps_track=gps,
        imu_data=imu,
        capture=CaptureMetadata(video_name="capture.mp4", frame_count=2, fps=10),
        producer=producer(),
        reconstruction_metrics={"point_count": 3},
        catalog_path=catalog,
    )

    assert result.package.manifest_path == package_root / "manifest.json"
    assert result.catalog_record["artifact_id"] == "art_test"
    assert result.copc.point_count == 2
    assert {artifact.path for artifact in result.manifest.artifacts} == {
        "geometry/points.copc.laz",
        "sources.json",
        "metrics.json",
        "cameras/poses.parquet",
        "telemetry/gps.parquet",
        "telemetry/imu.parquet",
    }
    assert result.manifest.coordinate_frame.transform_to_ecef is not None
    assert result.manifest.coordinate_frame.crs is None
    assert result.manifest.coordinate_frame.proj_pipeline is None
    assert result.manifest.lossy_operations[0].points_in == 3
    assert result.manifest.lossy_operations[0].points_out == 2
    assert not (package_root / "aligned_pointcloud.ply").exists()
    assert not (package_root / "metadata.json").exists()


def test_unaligned_metric_package_keeps_metric_units_and_can_consolidate(
    tmp_path: Path,
):
    cloud = PointCloud(
        points=np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]]),
        colors=np.zeros((2, 3), dtype=np.uint8),
        confidence=np.ones(2, dtype=np.float32),
        is_metric=True,
    )

    result = ReconstructionPackagePublisher(FakeCopcPublisher()).publish(
        tmp_path / "unaligned",
        identity=PackageIdentity("cap_u", "run_u", "art_u"),
        sources=[PackageSource(cloud, kind="capture")],
        alignment=AlignmentResult.unaligned("gps_unavailable"),
        poses=None,
        gps_track=None,
        imu_data=None,
        capture=CaptureMetadata(video_name="capture.mp4"),
        producer=producer(),
        reconstruction_metrics={},
    )

    assert result.manifest.coordinate_frame.units == "metre"
    assert result.manifest.coordinate_frame.transform_to_ecef is None
    # Metric local data is safe to consolidate even when global placement is
    # unavailable.
    assert len(result.manifest.lossy_operations) == 1


def test_metric_package_without_confidence_uses_deterministic_voxel_policy(
    tmp_path: Path,
):
    cloud = PointCloud(
        points=np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]]),
        is_metric=True,
    )

    result = ReconstructionPackagePublisher(FakeCopcPublisher()).publish(
        tmp_path / "without-confidence",
        identity=PackageIdentity("cap_c", "run_c", "art_c"),
        sources=[PackageSource(cloud, kind="capture")],
        alignment=AlignmentResult.unaligned("gps_unavailable"),
        poses=None,
        gps_track=None,
        imu_data=None,
        capture=CaptureMetadata(video_name="capture.mp4"),
        producer=producer(),
        reconstruction_metrics={},
    )

    assert result.copc.point_count == 1
    assert result.manifest.lossy_operations[0].operation == (
        "deterministic_voxel_consolidation"
    )
    assert "Confidence" not in result.manifest.artifacts[0].required_dimensions


def test_unaligned_nonmetric_package_does_not_apply_metre_voxel(tmp_path: Path):
    cloud = PointCloud(
        points=np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]]),
        confidence=np.ones(2, dtype=np.float32),
        is_metric=False,
    )

    result = ReconstructionPackagePublisher(FakeCopcPublisher()).publish(
        tmp_path / "nonmetric",
        identity=PackageIdentity("cap_n", "run_n", "art_n"),
        sources=[PackageSource(cloud, kind="capture")],
        alignment=AlignmentResult.unaligned("gps_unavailable"),
        poses=None,
        gps_track=None,
        imu_data=None,
        capture=CaptureMetadata(video_name="capture.mp4"),
        producer=producer(),
        reconstruction_metrics={},
    )

    assert result.copc.point_count == 2
    assert result.manifest.coordinate_frame.units == "unknown"
    assert result.manifest.lossy_operations == []
    assert result.package.metrics.validation["voxel_consolidation"] == (
        "skipped_nonmetric_coordinates"
    )


def test_real_converter_publishes_a_valid_package_when_installed(tmp_path: Path):
    try:
        CopcPublisher().resolve_executable()
    except CopcPublisherError:
        import pytest

        pytest.skip("pinned copc_converter is not installed")

    cloud = PointCloud(
        points=np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001], [1.0, 1.0, 1.0]]),
        colors=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
        confidence=np.array([0.1, 0.9, 0.5], dtype=np.float32),
        is_metric=True,
    )

    result = ReconstructionPackagePublisher().publish(
        tmp_path / "real",
        identity=PackageIdentity("cap_real", "run_real", "art_real"),
        sources=[PackageSource(cloud, kind="batch")],
        alignment=AlignmentResult.unaligned("gps_unavailable"),
        poses=None,
        gps_track=None,
        imu_data=None,
        capture=CaptureMetadata(video_name="capture.mp4"),
        producer=producer(),
        reconstruction_metrics={"point_count": 3},
    )

    with laspy.open(result.copc.output) as reader:
        assert reader.header.point_count == 2
        assert tuple(reader.header.scales) == (0.001, 0.001, 0.001)
    assert result.copc.structure.hierarchy_pages == 1
    assert result.package.manifest.artifact_id == "art_real"
