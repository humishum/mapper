"""Canonical reconstruction-package v1 assembly.

This module is the application-side bridge between model-native results and the
strict package contract owned by the viewer backend.  It deliberately stages
one LAZ shard per source unit so current reconstruction windows remain bounded
and retain provenance; future SLAM submaps can use the same interface.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from viewer.backend.domain.geospatial import global_placement_from_local_bounds
from viewer.backend.domain.package import (
    Alignment,
    ArtifactFile,
    CaptureMetadata,
    ColumnContract,
    CoordinateFrame,
    LayerDefault,
    LossyOperation,
    Manifest,
    Metrics,
    Producer,
    SourceRecord,
    SourcesDocument,
    StageMetric,
    TABULAR_COLUMN_DTYPES,
)
from viewer.backend.services.catalog import Catalog, new_opaque_id
from viewer.backend.services.package_validator import PackageValidator, ValidatedPackage
from viewer.backend.services.package_writer import (
    describe_artifact,
    write_json_sidecar,
    write_manifest,
)

from ..core.types import (
    AlignmentResult,
    CameraPoses,
    GPSTrack,
    IMUData,
    PointCloud,
)
from .copc import COPC_CONVERTER_VERSION, CopcPublisher, PublishResult
from .consolidation import (
    VoxelConsolidationConfig,
    consolidate_laz_shards,
)
from .las_staging import LasStagingConfig, write_laz_shard


@dataclass(frozen=True)
class PackageSource:
    """One independently staged point source and its provenance."""

    pointcloud: PointCloud
    kind: str
    name: str | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    frame_indices: Sequence[int] | None = None
    timestamp_start_s: float | None = None
    timestamp_end_s: float | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PackageIdentity:
    """Opaque identities persisted in a package and the catalog."""

    capture_id: str
    run_id: str
    artifact_id: str

    @classmethod
    def create(cls) -> "PackageIdentity":
        return cls(
            capture_id=new_opaque_id("cap"),
            run_id=new_opaque_id("run"),
            artifact_id=new_opaque_id("art"),
        )


def capture_id_for_file(path: str | Path) -> str:
    """Create an opaque, repeatable capture ID for the same local file revision.

    A future ingest service can supply its own durable capture ID.  Until then,
    the resolved URI plus size and nanosecond mtime distinguish local revisions
    without exposing a filename in the identifier.
    """

    source = Path(path).resolve(strict=True)
    stat = source.stat()
    fingerprint = f"{source.as_uri()}|{stat.st_size}|{stat.st_mtime_ns}"
    return f"cap_{uuid.uuid5(uuid.NAMESPACE_URL, fingerprint).hex}"


@dataclass(frozen=True)
class PackagePublishResult:
    package_root: Path
    package: ValidatedPackage
    copc: PublishResult
    catalog_record: dict[str, Any] | None

    @property
    def manifest(self) -> Manifest:
        return self.package.manifest


def _json_safe(value: Any) -> Any:
    """Convert common scientific-Python values to JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _bounds(pointclouds: Sequence[PointCloud]) -> tuple[np.ndarray, np.ndarray]:
    nonempty = [pointcloud.points for pointcloud in pointclouds if len(pointcloud)]
    if not nonempty:
        raise ValueError("a COPC package requires at least one point")
    return (
        np.minimum.reduce([points.min(axis=0) for points in nonempty]).astype(
            np.float64
        ),
        np.maximum.reduce([points.max(axis=0) for points in nonempty]).astype(
            np.float64
        ),
    )


def _column_contracts(kind: str) -> list[ColumnContract]:
    return [
        ColumnContract(
            name=name,
            dtype=dtype,
            nullable=name in {"timestamp_s", "fix_type", "hdop"},
        )
        for name, dtype in TABULAR_COLUMN_DTYPES[kind].items()
    ]


def _write_canonical_parquet(
    dataframe: Any,
    path: Path,
    kind: str,
) -> None:
    """Write only the stable contract columns with exact physical dtypes."""

    contract = TABULAR_COLUMN_DTYPES[kind]
    table = dataframe.loc[:, list(contract)].copy()
    for name, dtype in contract.items():
        if dtype == "int64":
            # Pandas' nullable Int64 maps to an Arrow int64 column while still
            # representing absent GPS fix values.
            table[name] = table[name].astype("Int64")
        else:
            table[name] = table[name].astype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


class ReconstructionPackagePublisher:
    """Publish, validate, and optionally catalog one reconstruction package."""

    def __init__(
        self,
        copc_publisher: CopcPublisher | None = None,
        *,
        staging_writer: Callable[..., Any] = write_laz_shard,
        consolidation_writer: Callable[..., Any] = consolidate_laz_shards,
        validator: PackageValidator | None = None,
    ):
        self.copc_publisher = copc_publisher or CopcPublisher()
        self.staging_writer = staging_writer
        self.consolidation_writer = consolidation_writer
        self.validator = validator or PackageValidator()

    def publish(
        self,
        package_root: str | Path,
        *,
        identity: PackageIdentity,
        sources: Sequence[PackageSource],
        alignment: AlignmentResult,
        poses: CameraPoses | None,
        gps_track: GPSTrack | None,
        imu_data: IMUData | None,
        capture: CaptureMetadata,
        producer: Producer,
        reconstruction_metrics: Mapping[str, Any],
        model_metadata: Mapping[str, Any] | None = None,
        catalog_path: str | Path | None = None,
        voxel_size_m: float = 0.02,
    ) -> PackagePublishResult:
        root = Path(package_root)
        if voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if (root / "manifest.json").exists():
            raise FileExistsError(root / "manifest.json")
        root.mkdir(parents=True, exist_ok=True)
        if not sources:
            raise ValueError("at least one package source is required")
        if len(sources) > 65536:
            raise ValueError("LAS PointSourceId supports at most 65,536 sources")

        clouds = [source.pointcloud for source in sources]
        bounds_min, bounds_max = _bounds(clouds)
        confidence_presence = {cloud.confidence is not None for cloud in clouds}
        color_presence = {cloud.colors is not None for cloud in clouds}
        if len(confidence_presence) != 1:
            raise ValueError("all source units must share the confidence schema")
        if len(color_presence) != 1:
            raise ValueError("all source units must share the color schema")
        include_confidence = confidence_presence == {True}
        metric_sources = all(cloud.is_metric for cloud in clouds)

        source_records = [
            SourceRecord(
                source_index=index,
                kind=source.kind,
                capture_id=identity.capture_id,
                run_id=identity.run_id,
                name=source.name,
                frame_start=source.frame_start,
                frame_end=source.frame_end,
                frame_indices=(
                    [int(value) for value in source.frame_indices]
                    if source.frame_indices is not None
                    else None
                ),
                timestamp_start_s=source.timestamp_start_s,
                timestamp_end_s=source.timestamp_end_s,
                point_count=len(source.pointcloud),
                metadata=_json_safe(source.metadata or {}),
            )
            for index, source in enumerate(sources)
        ]
        source_kinds = {record.kind for record in source_records}
        granularity = next(iter(source_kinds)) if len(source_kinds) == 1 else "batch"
        sources_document = SourcesDocument(
            provenance_dimension="PointSourceId",
            granularity=granularity,
            sources=source_records,
        )

        artifacts: list[ArtifactFile] = []
        geometry_path = root / "geometry" / "points.copc.laz"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        consolidation_result = None
        consolidation_wall_time_s = 0.0
        with tempfile.TemporaryDirectory(prefix="mapper-laz-shards-") as staging_text:
            staging = Path(staging_text)
            shards = []
            staging_config = LasStagingConfig(
                include_confidence=include_confidence,
                include_contributor_count=True,
            )
            for index, source in enumerate(sources):
                if len(source.pointcloud) == 0:
                    continue
                shard = self.staging_writer(
                    staging / f"{index:05d}.laz",
                    source.pointcloud.points,
                    source.pointcloud.colors,
                    index,
                    confidence=source.pointcloud.confidence,
                    contributor_count=1,
                    config=staging_config,
                )
                shards.append(shard.path)
            if not shards:
                raise ValueError(
                    "a COPC package requires at least one non-empty source"
                )
            # Consolidation preserves PointSourceId and ContributorCount.
            # Confidence selects the winner when present; otherwise the
            # consolidator uses a deterministic lowest-source policy.
            publish_inputs = shards
            if metric_sources:
                consolidation_started = time.monotonic()
                consolidation_result = self.consolidation_writer(
                    shards,
                    staging / "consolidated",
                    config=VoxelConsolidationConfig(voxel_size=voxel_size_m),
                )
                consolidation_wall_time_s = time.monotonic() - consolidation_started
                publish_inputs = list(consolidation_result.shards)
            copc_result = self.copc_publisher.publish(publish_inputs, geometry_path)
        published_bounds_min = np.asarray(
            getattr(copc_result.structure, "bounds_min", bounds_min),
            dtype=np.float64,
        )
        published_bounds_max = np.asarray(
            getattr(copc_result.structure, "bounds_max", bounds_max),
            dtype=np.float64,
        )

        point_dimensions = ["X", "Y", "Z", "PointSourceId", "ContributorCount"]
        if color_presence == {True}:
            point_dimensions.extend(["Red", "Green", "Blue"])
        if include_confidence:
            point_dimensions.append("Confidence")
        artifacts.append(
            describe_artifact(
                root,
                "geometry/points.copc.laz",
                representation_id=new_opaque_id("rep"),
                kind="points",
                format="copc/laz",
                media_type="application/vnd.laszip",
                frame="artifact_local",
                bounds_min=tuple(float(value) for value in published_bounds_min),
                bounds_max=tuple(float(value) for value in published_bounds_max),
                point_count=int(copc_result.point_count),
                required_dimensions=point_dimensions,
                metadata={
                    "converter_version": copc_result.converter_version,
                    "publisher_wall_time_s": float(copc_result.wall_seconds),
                },
            )
        )

        sources_path = write_json_sidecar(root, "sources.json", sources_document)
        artifacts.append(
            describe_artifact(
                root,
                sources_path.relative_to(root).as_posix(),
                representation_id=new_opaque_id("rep"),
                kind="sources",
                format="json",
                media_type="application/json",
            )
        )

        if poses is not None:
            pose_path = root / "cameras" / "poses.parquet"
            _write_canonical_parquet(poses.to_dataframe(), pose_path, "poses")
            artifacts.append(
                describe_artifact(
                    root,
                    "cameras/poses.parquet",
                    representation_id=new_opaque_id("rep"),
                    kind="poses",
                    format="parquet",
                    media_type="application/vnd.apache.parquet",
                    frame="artifact_local",
                    columns=_column_contracts("poses"),
                )
            )
        if gps_track is not None:
            gps_path = root / "telemetry" / "gps.parquet"
            _write_canonical_parquet(
                gps_track.to_dataframe(), gps_path, "telemetry_gps"
            )
            artifacts.append(
                describe_artifact(
                    root,
                    "telemetry/gps.parquet",
                    representation_id=new_opaque_id("rep"),
                    kind="telemetry_gps",
                    format="parquet",
                    media_type="application/vnd.apache.parquet",
                    columns=_column_contracts("telemetry_gps"),
                )
            )
        if imu_data is not None:
            imu_path = root / "telemetry" / "imu.parquet"
            _write_canonical_parquet(imu_data.to_dataframe(), imu_path, "telemetry_imu")
            artifacts.append(
                describe_artifact(
                    root,
                    "telemetry/imu.parquet",
                    representation_id=new_opaque_id("rep"),
                    kind="telemetry_imu",
                    format="parquet",
                    media_type="application/vnd.apache.parquet",
                    columns=_column_contracts("telemetry_imu"),
                )
            )

        points_in = sum(len(cloud) for cloud in clouds)
        metric_stages = []
        if consolidation_result is not None:
            metric_stages.append(
                StageMetric(
                    name="voxel_consolidation",
                    wall_time_s=consolidation_wall_time_s,
                    metadata={
                        "voxel_size_m": consolidation_result.voxel_size,
                        "points_in": consolidation_result.points_in,
                        "points_out": consolidation_result.points_out,
                        "bucket_count": consolidation_result.bucket_count,
                        "max_bucket_points": consolidation_result.max_bucket_points,
                    },
                )
            )
        metric_stages.append(
            StageMetric(
                name="copc_publication",
                wall_time_s=float(copc_result.wall_seconds),
                metadata={
                    "converter_version": copc_result.converter_version,
                    "source_count": len(sources),
                    "model_metadata": _json_safe(model_metadata or {}),
                    "reconstruction_metrics": _json_safe(reconstruction_metrics),
                    "voxel_consolidation": (
                        "applied"
                        if consolidation_result is not None
                        else (
                            "skipped_nonmetric_coordinates"
                            if not metric_sources
                            else "not_requested"
                        )
                    ),
                },
            )
        )
        metrics_document = Metrics(
            stages=metric_stages,
            total_wall_time_s=(
                consolidation_wall_time_s + float(copc_result.wall_seconds)
            ),
            input_point_count=points_in,
            output_point_count=int(copc_result.point_count),
            output_byte_size=int(copc_result.file_bytes),
            validation={
                "package_contract": True,
                "copc_structure": True,
                "voxel_consolidation": (
                    "applied"
                    if consolidation_result is not None
                    else (
                        "skipped_nonmetric_coordinates"
                        if not metric_sources
                        else "not_requested"
                    )
                ),
            },
        )
        metrics_path = write_json_sidecar(root, "metrics.json", metrics_document)
        artifacts.append(
            describe_artifact(
                root,
                metrics_path.relative_to(root).as_posix(),
                representation_id=new_opaque_id("rep"),
                kind="metrics",
                format="json",
                media_type="application/json",
            )
        )

        if alignment.accepted:
            placement = global_placement_from_local_bounds(
                tuple(float(value) for value in published_bounds_min),
                tuple(float(value) for value in published_bounds_max),
                alignment.enu_to_ecef_transform.reshape(-1).tolist(),
            )
            coordinate_frame = CoordinateFrame(
                name="artifact_local",
                units="metre",
                axis_order=["east", "north", "up"],
                handedness="right",
                pose_convention="camera_to_world",
                origin_wgs84=placement.origin_wgs84,
                transform_to_ecef=alignment.enu_to_ecef_transform.reshape(-1).tolist(),
            )
            footprint = placement.footprint_wgs84
        else:
            coordinate_frame = CoordinateFrame(
                name="artifact_local",
                units=(
                    "metre"
                    if all(pointcloud.is_metric for pointcloud in clouds)
                    else "unknown"
                ),
                axis_order=["x", "y", "z"],
                handedness="unknown",
                pose_convention="camera_to_world",
            )
            footprint = None

        lossy_operations = []
        if consolidation_result is not None:
            lossy_operations.append(
                LossyOperation(
                    operation=(
                        "confidence_aware_voxel_consolidation"
                        if include_confidence
                        else "deterministic_voxel_consolidation"
                    ),
                    parameters={"voxel_size_m": consolidation_result.voxel_size},
                    points_in=consolidation_result.points_in,
                    points_out=consolidation_result.points_out,
                    provenance_policy=(
                        (
                            "PointSourceId is inherited from the highest-confidence "
                            "point in each voxel"
                        )
                        if include_confidence
                        else (
                            "PointSourceId is inherited from the lowest source "
                            "index in each voxel"
                        )
                    )
                    + "; ContributorCount records all inputs",
                )
            )

        manifest = Manifest(
            run_id=identity.run_id,
            capture_id=identity.capture_id,
            artifact_id=identity.artifact_id,
            created_at=datetime.now(UTC),
            capture=capture,
            producer=producer.model_copy(
                update={
                    "publisher_version": getattr(
                        copc_result, "converter_version", COPC_CONVERTER_VERSION
                    )
                }
            ),
            coordinate_frame=coordinate_frame,
            alignment=Alignment(
                status=alignment.status.value,
                method=alignment.method,
                model_to_artifact_local=alignment.transform.reshape(-1).tolist(),
                scale=float(alignment.scale),
                inlier_count=int(alignment.inlier_count),
                horizontal_rmse_m=alignment.horizontal_rmse_m,
                vertical_rmse_m=alignment.vertical_rmse_m,
                clock_offset_s=float(alignment.clock_offset_s),
                clock_peak_quality=float(alignment.clock_peak_quality),
                gravity_constrained="gravity" in alignment.method,
                rejection_reason=alignment.reason if not alignment.accepted else None,
            ),
            footprint_wgs84=footprint,
            artifacts=artifacts,
            lossy_operations=lossy_operations,
            layer_default=LayerDefault(
                visible=True,
                point_budget=2_000_000,
                point_size=1.0,
                color_dimension="rgb" if color_presence == {True} else None,
            ),
        )

        # The manifest is the package visibility/commit marker and must be last.
        write_manifest(root, manifest)
        validated = self.validator.validate(root)
        catalog_record = (
            Catalog(catalog_path, validator=self.validator).register_package(root)
            if catalog_path is not None
            else None
        )
        return PackagePublishResult(
            package_root=root.resolve(),
            package=validated,
            copc=copc_result,
            catalog_record=catalog_record,
        )


def package_source_from_window(
    pointcloud: PointCloud,
    window_metadata: Mapping[str, Any],
) -> PackageSource:
    """Convert current fixed-window metadata into generic source provenance."""

    frame_indices = window_metadata.get("frame_indices")
    return PackageSource(
        pointcloud=pointcloud,
        kind="window",
        name=(
            f"window_{int(window_metadata['window_id']):03d}"
            if window_metadata.get("window_id") is not None
            else None
        ),
        frame_start=window_metadata.get("frame_start"),
        frame_end=window_metadata.get("frame_end"),
        frame_indices=frame_indices,
        timestamp_start_s=window_metadata.get("timestamp_start_s"),
        timestamp_end_s=window_metadata.get("timestamp_end_s"),
        metadata=dict(window_metadata),
    )
