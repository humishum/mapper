from __future__ import annotations

import json
import os
from pathlib import Path

import laspy
import numpy as np
import pyarrow.parquet as pq
import pytest

from src.alignment.gps_aligner import _enu_to_ecef_transform
from src.core.types import AlignmentResult, AlignmentStatus, GPSTrack
from src.publisher import (
    CopcPublisher,
    CopcPublisherConfig,
    ExistingCopcPackager,
    PackageIdentity,
    write_laz_shard,
)
from src.publisher.onboarding import _stage_canonical_laz


def _legacy(root: Path, counts: tuple[int, ...]) -> None:
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "video_name": "capture.mp4",
                "fps": 10,
                "frames": 4,
                "is_metric": False,
            }
        )
    )
    for window_id, count in enumerate(counts):
        directory = root / "windows" / f"window_{window_id:03d}"
        directory.mkdir(parents=True)
        start = window_id * 2
        frames = np.arange(start, start + 2, dtype=np.int64)
        metadata = {
            "window_metadata": {
                "window_id": window_id,
                "frame_start": int(frames[0]),
                "frame_end": int(frames[-1]),
                "frame_indices": frames.tolist(),
            },
            "model_metadata": {"model": "fixture"},
            "point_count": count,
            "is_metric": False,
            "has_poses": True,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata))
        poses = np.repeat(np.eye(4)[None], len(frames), axis=0)
        poses[:, 0, 3] = frames
        np.savez(
            directory / "poses.npz",
            poses=poses,
            timestamps=frames.astype(np.float64),
            intrinsics=np.eye(3),
            frame_indices=frames,
        )


def _copc(
    root: Path,
    counts: tuple[int, ...],
    *,
    contributor_count: bool = True,
) -> Path:
    shards = []
    for source_id, count in enumerate(counts):
        path = root / f"source-{source_id}.laz"
        points = np.column_stack(
            (
                np.arange(count, dtype=np.float64) + source_id * 20,
                np.full(count, source_id, dtype=np.float64),
                np.zeros(count),
            )
        )
        if contributor_count:
            write_laz_shard(
                path,
                points,
                np.full((count, 3), source_id + 1, dtype=np.uint8),
                source_id,
                contributor_count=1,
            )
        else:
            # write_laz_shard defaults to the canonical dimension; make a
            # deliberately old LAS shard with no extra dimensions instead.
            header = laspy.LasHeader(point_format=7, version="1.4")
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
            las.point_source_id = np.full(count, source_id, dtype=np.uint16)
            las.red = las.green = las.blue = np.full(count, 257, dtype=np.uint16)
            las.write(path)
        shards.append(path)
    output = root / "existing.copc.laz"
    config = CopcPublisherConfig(
        executable=Path(".tools/copc_converter/0.11.0/copc_converter"),
        memory_limit="256M",
        threads=2,
        temp_dir=root,
        required_dimensions=("ContributorCount",) if contributor_count else (),
    )
    CopcPublisher(config).publish(shards, output)
    return output


def test_compatible_copc_is_hardlinked_and_legacy_sidecars_are_canonical(tmp_path):
    counts = (17, 13)
    source = _copc(tmp_path, counts)
    legacy = tmp_path / "legacy"
    _legacy(legacy, counts)
    package = tmp_path / "package"

    result = ExistingCopcPackager().package(
        source,
        legacy,
        package,
        tmp_path / "catalog.sqlite3",
        identity=PackageIdentity("cap_fixture", "run_fixture", "art_fixture"),
    )

    geometry = package / "geometry" / "points.copc.laz"
    assert result.geometry_disposition == "hardlinked"
    assert os.stat(source).st_ino == os.stat(geometry).st_ino
    assert result.input_sha256 == result.geometry_sha256
    assert result.manifest.alignment.rejection_reason == "source_video_missing"
    assert result.manifest.coordinate_frame.transform_to_ecef is None
    assert result.catalog_record["artifact_id"] == "art_fixture"
    sources = json.loads((package / "sources.json").read_text())
    assert [item["point_count"] for item in sources["sources"]] == list(counts)
    poses = pq.read_table(package / "cameras" / "poses.parquet")
    assert poses.num_rows == 4


def test_missing_contributor_count_is_republished_with_explicit_default(tmp_path):
    counts = (19,)
    source = _copc(tmp_path, counts, contributor_count=False)
    legacy = tmp_path / "legacy"
    _legacy(legacy, counts)

    result = ExistingCopcPackager().package(
        source,
        legacy,
        tmp_path / "package",
        tmp_path / "catalog.sqlite3",
    )

    assert result.geometry_disposition == "republished"
    assert result.incompatibilities == ("missing_contributor_count",)
    points_artifact = result.manifest.artifacts[0]
    assert points_artifact.metadata["migration_defaults"] == [
        "ContributorCount=1 (legacy migration default)"
    ]
    assert points_artifact.metadata["confidence_policy"] == "absent"
    with laspy.open(tmp_path / "package/geometry/points.copc.laz") as reader:
        points = reader.read()
    np.testing.assert_array_equal(points["ContributorCount"], 1)
    assert "Confidence" not in reader.header.point_format.dimension_names


def test_window_distribution_mismatch_is_rejected_before_package_creation(tmp_path):
    source = _copc(tmp_path, (17,))
    legacy = tmp_path / "legacy"
    _legacy(legacy, (16,))
    package = tmp_path / "package"

    with pytest.raises(RuntimeError, match="distribution does not match"):
        ExistingCopcPackager().package(
            source,
            legacy,
            package,
            tmp_path / "catalog.sqlite3",
        )

    assert not package.exists()


def test_republication_staging_upgrades_legacy_point_format_without_confidence(
    tmp_path,
):
    source = tmp_path / "legacy-format-3.laz"
    header = laspy.LasHeader(point_format=3, version="1.2")
    legacy = laspy.LasData(header)
    legacy.x = np.array([1.0, 2.0])
    legacy.y = np.array([3.0, 4.0])
    legacy.z = np.array([5.0, 6.0])
    legacy.red = legacy.green = legacy.blue = np.array([257, 514])
    legacy.point_source_id = np.array([4, 5], dtype=np.uint16)
    legacy.write(source)

    staged = tmp_path / "canonical.laz"
    _stage_canonical_laz(source, staged)

    with laspy.open(staged) as reader:
        points = reader.read()
        assert reader.header.point_format.id == 7
        assert str(reader.header.version) == "1.4"
        assert "Confidence" not in reader.header.point_format.dimension_names
    np.testing.assert_array_equal(points.point_source_id, [4, 5])
    np.testing.assert_array_equal(points["ContributorCount"], [1, 1])


class _Telemetry:
    def extract(self, video):
        return (
            GPSTrack(
                latitudes=np.array([37.0, 37.0001]),
                longitudes=np.array([-121.0, -121.0001]),
                altitudes=np.array([100.0, 101.0]),
                timestamps=np.array([0.0, 1.0]),
            ),
            None,
        )


class _Aligned:
    def align(self, pointcloud, poses, gps_track, imu_data, allow_scale):
        anchor = (37.0, -121.0, 100.0)
        ecef = GPSTrack(
            latitudes=np.array([anchor[0]]),
            longitudes=np.array([anchor[1]]),
            altitudes=np.array([anchor[2]]),
        ).to_ecef()[0]
        transform = np.eye(4)
        transform[:3, 3] = [10.0, 20.0, 30.0]
        return AlignmentResult(
            transform=transform,
            enu_to_ecef_transform=_enu_to_ecef_transform(anchor, ecef),
            scale=1.0,
            method="gps_robust_weighted_umeyama",
            status=AlignmentStatus.ALIGNED,
            inlier_count=4,
            correspondence_count=4,
            rmse_m=0.2,
            horizontal_rmse_m=0.1,
            vertical_rmse_m=0.1,
            anchor_wgs84=anchor,
            anchor_ecef=ecef,
        )


def test_video_alignment_republishes_geometry_and_registers_ecef_footprint(tmp_path):
    counts = (23,)
    source = _copc(tmp_path, counts)
    legacy = tmp_path / "legacy"
    _legacy(legacy, counts)
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"fixture")

    result = ExistingCopcPackager(
        telemetry_factory=_Telemetry,
        aligner=_Aligned(),
    ).package(
        source,
        legacy,
        tmp_path / "package",
        tmp_path / "catalog.sqlite3",
        source_video=video,
    )

    assert result.geometry_disposition == "republished"
    assert result.manifest.alignment.status == "aligned"
    assert result.manifest.coordinate_frame.transform_to_ecef is not None
    assert result.manifest.footprint_wgs84 is not None
    assert result.catalog_record["footprint"] is not None
    poses = pq.read_table(tmp_path / "package/cameras/poses.parquet").to_pandas()
    assert poses["tx"].iloc[0] == 10.0
