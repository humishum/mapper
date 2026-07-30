"""Small canonical package fixtures shared by backend tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from viewer.backend.domain.package import (
    Alignment,
    ArtifactFile,
    CaptureMetadata,
    CoordinateFrame,
    Manifest,
    Producer,
    WGS84Footprint,
)
from viewer.backend.services.package_writer import describe_artifact, write_manifest

IDENTITY = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]


def create_package(
    root: Path,
    *,
    run_id: str = "run_test",
    capture_id: str = "capture_test",
    artifact_id: str = "artifact_test",
    representation_id: str = "representation_test",
    aligned: bool = True,
    payload: bytes = b"not parsed COPC fixture",
) -> Manifest:
    geometry = root / "geometry" / "points.copc.laz"
    geometry.parent.mkdir(parents=True)
    geometry.write_bytes(payload)
    point_artifact: ArtifactFile = describe_artifact(
        root,
        "geometry/points.copc.laz",
        representation_id=representation_id,
        kind="points",
        format="copc/laz",
        media_type="application/vnd.laszip",
        frame="artifact_local",
        bounds_min=(-1.0, -2.0, -3.0),
        bounds_max=(4.0, 5.0, 6.0),
        point_count=42,
        required_dimensions=["X", "Y", "Z", "Red", "Green", "Blue"],
    )
    if aligned:
        coordinate_frame = CoordinateFrame(
            units="metre",
            axis_order=["east", "north", "up"],
            handedness="right",
            origin_wgs84=(-121.0, 37.0, 100.0),
            proj_pipeline="+proj=pipeline",
            transform_to_ecef=IDENTITY,
        )
        alignment = Alignment(
            status="aligned",
            method="weighted_umeyama",
            model_to_artifact_local=IDENTITY,
            scale=1.0,
            inlier_count=20,
            horizontal_rmse_m=0.25,
        )
        footprint = WGS84Footprint(
            coordinates=[
                (-121.01, 36.99),
                (-120.99, 36.99),
                (-120.99, 37.01),
                (-121.01, 37.01),
                (-121.01, 36.99),
            ]
        )
    else:
        coordinate_frame = CoordinateFrame(
            units="unknown",
            axis_order=["x", "y", "z"],
            handedness="unknown",
        )
        alignment = Alignment(
            status="unaligned",
            method="rejected",
            model_to_artifact_local=IDENTITY,
            scale=1.0,
            inlier_count=0,
            rejection_reason="insufficient quality-filtered GPS pairs",
        )
        footprint = None
    manifest = Manifest(
        run_id=run_id,
        capture_id=capture_id,
        artifact_id=artifact_id,
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
        capture=CaptureMetadata(
            started_at=datetime(2026, 7, 24, tzinfo=UTC),
            ended_at=datetime(2026, 7, 24, 0, 5, tzinfo=UTC),
            video_name="fixture.mp4",
            frame_count=100,
            fps=10.0,
        ),
        producer=Producer(
            model_name="fixture",
            model_config={"window_size": 50},
            adapter_name="fixture-adapter",
            adapter_version="1",
            publisher_name="fixture-publisher",
            publisher_version="1",
        ),
        coordinate_frame=coordinate_frame,
        alignment=alignment,
        footprint_wgs84=footprint,
        artifacts=[point_artifact],
    )
    write_manifest(root, manifest)
    return manifest
