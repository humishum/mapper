"""Package and catalog an existing COPC without rerunning reconstruction."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.alignment.gps_aligner import GPSAligner
from src.core.types import AlignmentResult, CameraPoses, PointCloud
from src.preprocessing.telemetry import TelemetryExtractor
from viewer.backend.domain.geospatial import global_placement_from_local_bounds
from viewer.backend.domain.package import (
    Alignment,
    CaptureMetadata,
    CoordinateFrame,
    LayerDefault,
    Manifest,
    Metrics,
    Producer,
    SourceRecord,
    SourcesDocument,
    StageMetric,
)
from viewer.backend.services.catalog import Catalog, new_opaque_id
from viewer.backend.services.package_validator import PackageValidator
from viewer.backend.services.package_writer import (
    describe_artifact,
    write_json_sidecar,
    write_manifest,
)

from .copc import COPC_CONVERTER_VERSION, CopcPublisher, PublishResult
from .copc_validation import CopcStructure, inspect_copc
from .package import (
    PackageIdentity,
    _column_contracts,
    _write_canonical_parquet,
    capture_id_for_file,
)

CANONICAL_SCALE = (0.001, 0.001, 0.001)
CANONICAL_OFFSET = (0.0, 0.0, 0.0)


class ExistingCopcOnboardingError(RuntimeError):
    """The existing data cannot safely be represented by package v1."""


def _has_dimension(dimensions: tuple[str, ...], name: str) -> bool:
    return name.casefold() in {dimension.casefold() for dimension in dimensions}


@dataclass(frozen=True)
class ExistingCopcInspection:
    path: Path
    sha256: str
    structure: CopcStructure
    dimensions: tuple[str, ...]
    source_distribution: dict[int, int]
    compatible: bool
    incompatibilities: tuple[str, ...]


@dataclass(frozen=True)
class ExistingCopcPackageResult:
    package_root: Path
    manifest: Manifest
    catalog_record: dict[str, Any]
    input_sha256: str
    geometry_sha256: str
    geometry_disposition: str
    incompatibilities: tuple[str, ...]
    publication: PublishResult | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_directories(root: Path) -> list[Path]:
    candidates = sorted((root / "windows").glob("window_*"))
    if not candidates:
        candidates = sorted(root.glob("window_*"))
    return [path for path in candidates if path.is_dir()]


def _window_metadata(root: Path) -> list[dict[str, Any]]:
    records = []
    for expected, directory in enumerate(_window_directories(root)):
        path = directory / "metadata.json"
        if not path.is_file():
            raise ExistingCopcOnboardingError(f"window metadata is missing: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(raw.get("window_metadata", {}))
        window_id = int(metadata.get("window_id", raw.get("window_id", expected)))
        if window_id != expected:
            raise ExistingCopcOnboardingError(
                "window metadata must be contiguous: "
                f"expected {expected}, found {window_id} in {path}"
            )
        metadata.update(
            {
                "window_id": window_id,
                "point_count": int(raw["point_count"]),
                "is_metric": bool(raw.get("is_metric", False)),
                "model_metadata": raw.get("model_metadata", {}),
                "_directory": directory,
            }
        )
        records.append(metadata)
    return records


def _source_records(
    metadata: list[dict[str, Any]],
    distribution: Mapping[int, int],
    identity: PackageIdentity,
) -> list[SourceRecord]:
    if metadata:
        expected = {
            int(item["window_id"]): int(item["point_count"]) for item in metadata
        }
        if expected != dict(distribution):
            raise ExistingCopcOnboardingError(
                "COPC PointSourceId distribution does not match legacy window "
                f"metadata (COPC={dict(distribution)}, metadata={expected})"
            )
        return [
            SourceRecord(
                source_index=int(item["window_id"]),
                kind="window",
                capture_id=identity.capture_id,
                run_id=identity.run_id,
                name=f"window_{int(item['window_id']):03d}",
                frame_start=item.get("frame_start"),
                frame_end=item.get("frame_end"),
                frame_indices=item.get("frame_indices"),
                point_count=int(item["point_count"]),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key not in {"_directory", "model_metadata"}
                },
            )
            for item in metadata
        ]
    return [
        SourceRecord(
            source_index=source_id,
            kind="batch",
            capture_id=identity.capture_id,
            run_id=identity.run_id,
            name=f"source_{source_id:05d}",
            point_count=count,
            metadata={"migration_note": "recovered from COPC PointSourceId"},
        )
        for source_id, count in sorted(distribution.items())
    ]


def load_legacy_poses(metadata: list[dict[str, Any]]) -> CameraPoses | None:
    """Load and de-duplicate saved per-window pose archives by frame index."""

    rows: dict[int, tuple[np.ndarray, float, np.ndarray | None]] = {}
    for item in metadata:
        path = Path(item["_directory"]) / "poses.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            missing = {"poses", "frame_indices"} - set(archive.files)
            if missing:
                raise ExistingCopcOnboardingError(
                    f"{path} is missing pose arrays: {sorted(missing)}"
                )
            poses = np.asarray(archive["poses"], dtype=np.float64)
            indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            timestamps = (
                np.asarray(archive["timestamps"], dtype=np.float64)
                if "timestamps" in archive.files
                else np.full(len(indices), np.nan)
            )
            intrinsics = (
                np.asarray(archive["intrinsics"], dtype=np.float64)
                if "intrinsics" in archive.files
                else None
            )
            CameraPoses(poses=poses, frame_indices=indices, timestamps=timestamps)
            for sequence, frame_index in enumerate(indices):
                intrinsic = (
                    intrinsics
                    if intrinsics is not None and intrinsics.ndim == 2
                    else (
                        intrinsics[sequence]
                        if intrinsics is not None
                        else None
                    )
                )
                # Overlap frames occur in adjacent windows. The first saved
                # occurrence is deterministic and avoids duplicating frames.
                rows.setdefault(
                    int(frame_index),
                    (poses[sequence], float(timestamps[sequence]), intrinsic),
                )
    if not rows:
        return None
    ordered = sorted(rows)
    pose_array = np.stack([rows[index][0] for index in ordered])
    timestamp_array = np.asarray([rows[index][1] for index in ordered])
    intrinsics = [rows[index][2] for index in ordered]
    intrinsic_array = (
        np.stack(intrinsics) if all(value is not None for value in intrinsics) else None
    )
    return CameraPoses(
        poses=pose_array,
        timestamps=timestamp_array,
        intrinsics=intrinsic_array,
        frame_indices=np.asarray(ordered, dtype=np.int64),
    )


def inspect_existing_copc(path: str | Path) -> ExistingCopcInspection:
    """Validate COPC structure, schema, distribution, bounds, and checksum."""

    path = Path(path).resolve(strict=True)
    try:
        structure = inspect_copc(path)
        import laspy

        counts = np.zeros(65536, dtype=np.int64)
        with laspy.open(path) as reader:
            dimensions = tuple(reader.header.point_format.dimension_names)
            dimension_info = {
                dimension.name.casefold(): dimension
                for dimension in reader.header.point_format.dimensions
            }
            header_bounds = (
                tuple(float(value) for value in reader.header.mins),
                tuple(float(value) for value in reader.header.maxs),
            )
            for points in reader.chunk_iterator(2_000_000):
                xyz = np.column_stack((points.x, points.y, points.z))
                if not np.isfinite(xyz).all():
                    raise ExistingCopcOnboardingError(
                        "COPC contains non-finite coordinates"
                    )
                counts += np.bincount(
                    np.asarray(points.point_source_id), minlength=65536
                )
                if _has_dimension(dimensions, "Confidence") and not np.isfinite(
                    np.asarray(points[dimension_info["confidence"].name])
                ).all():
                    raise ExistingCopcOnboardingError(
                        "COPC Confidence contains non-finite values"
                    )
                if _has_dimension(dimensions, "ContributorCount") and np.any(
                    np.asarray(points[dimension_info["contributorcount"].name]) < 1
                ):
                    raise ExistingCopcOnboardingError(
                        "COPC ContributorCount contains values below one"
                    )
    except ExistingCopcOnboardingError:
        raise
    except Exception as exc:
        raise ExistingCopcOnboardingError(f"could not inspect COPC {path}: {exc}") from exc

    if structure.point_count <= 0:
        raise ExistingCopcOnboardingError("COPC contains no points")
    if structure.hierarchy_point_count != structure.point_count:
        raise ExistingCopcOnboardingError(
            "COPC header and hierarchy point counts disagree"
        )
    if not structure.root_present or not structure.all_nodes_reachable:
        raise ExistingCopcOnboardingError("COPC hierarchy is not root-reachable")
    if structure.invalid_node_keys:
        raise ExistingCopcOnboardingError("COPC hierarchy contains invalid node keys")
    bounds = (structure.bounds_min, structure.bounds_max)
    if not np.isfinite(np.asarray(bounds)).all() or any(
        low > high for low, high in zip(*bounds, strict=True)
    ):
        raise ExistingCopcOnboardingError("COPC bounds are invalid")
    if not (
        np.allclose(header_bounds[0], bounds[0], rtol=0, atol=1e-9)
        and np.allclose(header_bounds[1], bounds[1], rtol=0, atol=1e-9)
    ):
        raise ExistingCopcOnboardingError(
            "COPC LAS header bounds disagree with structural bounds"
        )
    distribution = {
        int(index): int(counts[index]) for index in np.flatnonzero(counts)
    }
    if sum(distribution.values()) != structure.point_count:
        raise ExistingCopcOnboardingError(
            "PointSourceId distribution does not sum to the COPC point count"
        )
    incompatibilities = []
    if structure.point_format not in {6, 7}:
        incompatibilities.append("point_format")
    if structure.scale != CANONICAL_SCALE:
        incompatibilities.append("scale")
    if structure.offset != CANONICAL_OFFSET:
        incompatibilities.append("offset")
    if structure.hierarchy_pages != 1:
        incompatibilities.append("paged_hierarchy")
    if not _has_dimension(dimensions, "ContributorCount"):
        incompatibilities.append("missing_contributor_count")
    elif dimension_info["contributorcount"].dtype != np.dtype(np.uint16):
        incompatibilities.append("contributor_count_dtype")
    if (
        _has_dimension(dimensions, "Confidence")
        and dimension_info["confidence"].dtype != np.dtype(np.float32)
    ):
        incompatibilities.append("confidence_dtype")
    return ExistingCopcInspection(
        path=path,
        sha256=_sha256(path),
        structure=structure,
        dimensions=dimensions,
        source_distribution=distribution,
        compatible=not incompatibilities,
        incompatibilities=tuple(incompatibilities),
    )


def _stage_canonical_laz(
    source: Path,
    target: Path,
    *,
    transform: np.ndarray | None = None,
) -> None:
    """Stream an existing LAS/COPC into canonical LAZ without inventing fields."""

    import laspy

    with laspy.open(source) as reader:
        source_dimensions = tuple(reader.header.point_format.dimension_names)
        source_by_name = {name.casefold(): name for name in source_dimensions}
        has_rgb = all(
            _has_dimension(source_dimensions, name)
            for name in ("Red", "Green", "Blue")
        )
        point_format = laspy.PointFormat(7 if has_rgb else 6)
        if _has_dimension(source_dimensions, "Confidence"):
            point_format.add_extra_dimension(
                laspy.ExtraBytesParams(name="Confidence", type=np.float32)
            )
        if not _has_dimension(tuple(point_format.dimension_names), "ContributorCount"):
            point_format.add_extra_dimension(
                laspy.ExtraBytesParams(name="ContributorCount", type=np.uint16)
            )
        header = laspy.LasHeader(point_format=point_format, version="1.4")
        header.scales = np.asarray(CANONICAL_SCALE)
        header.offsets = np.asarray(CANONICAL_OFFSET)
        target.parent.mkdir(parents=True, exist_ok=True)
        with laspy.open(target, mode="w", header=header, do_compress=True) as writer:
            for points in reader.chunk_iterator(1_000_000):
                staged = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
                staged.x = np.asarray(points.x)
                staged.y = np.asarray(points.y)
                staged.z = np.asarray(points.z)
                if transform is not None:
                    xyz = np.column_stack((staged.x, staged.y, staged.z))
                    xyz = (transform[:3, :3] @ xyz.T).T + transform[:3, 3]
                    staged.x, staged.y, staged.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                for name in tuple(point_format.dimension_names):
                    source_name = source_by_name.get(name.casefold())
                    if name not in {"X", "Y", "Z"} and source_name is not None:
                        staged[name] = points[source_name]
                if not _has_dimension(source_dimensions, "ContributorCount"):
                    staged["ContributorCount"] = np.ones(
                        len(points), dtype=np.uint16
                    )
                writer.write_points(staged)


def _adopt_file(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlinked"
    except OSError as exc:
        if exc.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOTSUP,
        }:
            raise
        shutil.copy2(source, target)
        return "copied"


class ExistingCopcPackager:
    """Adopt or republish existing reconstruction output into package v1."""

    def __init__(
        self,
        copc_publisher: CopcPublisher | None = None,
        *,
        telemetry_factory: Callable[[], Any] = TelemetryExtractor,
        aligner: Any | None = None,
        validator: PackageValidator | None = None,
    ) -> None:
        self.copc_publisher = copc_publisher or CopcPublisher()
        self.telemetry_factory = telemetry_factory
        self.aligner = aligner or GPSAligner()
        self.validator = validator or PackageValidator()

    def package(
        self,
        existing_copc: str | Path,
        legacy_reconstruction_dir: str | Path,
        package_root: str | Path,
        catalog_path: str | Path,
        *,
        source_video: str | Path | None = None,
        identity: PackageIdentity | None = None,
    ) -> ExistingCopcPackageResult:
        inspection = inspect_existing_copc(existing_copc)
        legacy_root = Path(legacy_reconstruction_dir).resolve(strict=True)
        root = Path(package_root)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise FileExistsError(f"package root is not an empty directory: {root}")
        metadata = _window_metadata(legacy_root)
        capture_metadata_path = legacy_root / "metadata.json"
        capture_raw = (
            json.loads(capture_metadata_path.read_text(encoding="utf-8"))
            if capture_metadata_path.is_file()
            else {}
        )
        if source_video:
            video = Path(source_video).resolve(strict=True)
        else:
            named_video = capture_raw.get("video_name")
            candidate = legacy_root / str(named_video) if named_video else None
            video = (
                candidate.resolve()
                if candidate is not None and candidate.is_file()
                else None
            )
        identity = identity or PackageIdentity(
            capture_id_for_file(video or inspection.path),
            new_opaque_id("run"),
            new_opaque_id("art"),
        )
        sources = _source_records(metadata, inspection.source_distribution, identity)
        poses = load_legacy_poses(metadata)

        gps_track = None
        imu_data = None
        if video is None:
            alignment = AlignmentResult.unaligned("source_video_missing")
        elif poses is None:
            alignment = AlignmentResult.unaligned("legacy_poses_missing")
        else:
            try:
                gps_track, imu_data = self.telemetry_factory().extract(video)
            except Exception as exc:
                alignment = AlignmentResult.unaligned(
                    "telemetry_extraction_failed",
                    diagnostics={"exception": type(exc).__name__},
                )
            else:
                if gps_track is None:
                    alignment = AlignmentResult.unaligned(
                        "gps_telemetry_unavailable"
                    )
                else:
                    dummy = PointCloud(
                        points=np.zeros((1, 3), dtype=np.float32),
                        is_metric=all(item["is_metric"] for item in metadata)
                        if metadata
                        else False,
                    )
                    alignment = self.aligner.align(
                        dummy,
                        poses,
                        gps_track,
                        imu_data,
                        allow_scale=not dummy.is_metric,
                    )

        root.mkdir(parents=True, exist_ok=True)
        geometry = root / "geometry" / "points.copc.laz"
        publication = None
        started = time.monotonic()
        if inspection.compatible and not alignment.accepted:
            disposition = _adopt_file(inspection.path, geometry)
            if _sha256(geometry) != inspection.sha256:
                geometry.unlink(missing_ok=True)
                raise ExistingCopcOnboardingError(
                    "adopted COPC checksum differs from its validated input"
                )
            structure = inspection.structure
            geometry_dimensions = inspection.dimensions
            geometry_distribution = inspection.source_distribution
        else:
            with tempfile.TemporaryDirectory(prefix="mapper-existing-copc-") as text:
                staged = Path(text) / "canonical.laz"
                _stage_canonical_laz(
                    inspection.path,
                    staged,
                    transform=alignment.transform if alignment.accepted else None,
                )
                publication = self.copc_publisher.publish(staged, geometry)
            disposition = "republished"
            structure = publication.structure
            geometry_distribution = publication.source_distribution
            import laspy

            with laspy.open(geometry) as reader:
                geometry_dimensions = tuple(reader.header.point_format.dimension_names)
        if geometry_distribution != inspection.source_distribution:
            raise ExistingCopcOnboardingError(
                "packaged COPC source distribution changed during onboarding"
            )
        if alignment.accepted and poses is not None:
            poses = alignment.transform_poses(poses)

        artifacts = []
        required_dimensions = ["X", "Y", "Z", "PointSourceId", "ContributorCount"]
        for name in ("Red", "Green", "Blue", "Confidence"):
            if _has_dimension(geometry_dimensions, name):
                required_dimensions.append(name)
        migration_defaults = (
            ["ContributorCount=1 (legacy migration default)"]
            if "missing_contributor_count" in inspection.incompatibilities
            else []
        )
        artifacts.append(
            describe_artifact(
                root,
                "geometry/points.copc.laz",
                representation_id=new_opaque_id("rep"),
                kind="points",
                format="copc/laz",
                media_type="application/vnd.laszip",
                frame="artifact_local",
                bounds_min=structure.bounds_min,
                bounds_max=structure.bounds_max,
                point_count=structure.point_count,
                required_dimensions=required_dimensions,
                metadata={
                    "onboarding_disposition": disposition,
                    "input_sha256": inspection.sha256,
                    "migration_defaults": migration_defaults,
                    "confidence_policy": (
                        "preserved"
                        if _has_dimension(geometry_dimensions, "Confidence")
                        else "absent"
                    ),
                    "converter_version": (
                        publication.converter_version if publication else None
                    ),
                },
            )
        )
        sources_document = SourcesDocument(
            provenance_dimension="PointSourceId",
            granularity="window" if metadata else "batch",
            sources=sources,
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
            _write_canonical_parquet(
                imu_data.to_dataframe(), imu_path, "telemetry_imu"
            )
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

        metric = Metrics(
            stages=[
                StageMetric(
                    name="existing_copc_onboarding",
                    wall_time_s=time.monotonic() - started,
                    metadata={
                        "disposition": disposition,
                        "input_sha256": inspection.sha256,
                        "incompatibilities": list(inspection.incompatibilities),
                        "migration_defaults": migration_defaults,
                        "confidence_invented": False,
                    },
                )
            ],
            input_point_count=inspection.structure.point_count,
            output_point_count=structure.point_count,
            output_byte_size=geometry.stat().st_size,
            validation={
                "copc_hierarchy": True,
                "dimensions": True,
                "checksum": True,
                "source_distribution": True,
                "bounds": True,
            },
        )
        metrics_path = write_json_sidecar(root, "metrics.json", metric)
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
                structure.bounds_min,
                structure.bounds_max,
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
            is_metric = bool(capture_raw.get("is_metric", False)) or (
                bool(metadata) and all(item["is_metric"] for item in metadata)
            )
            coordinate_frame = CoordinateFrame(
                name="artifact_local",
                units="metre" if is_metric else "unknown",
                axis_order=["x", "y", "z"],
                handedness="unknown",
                pose_convention="camera_to_world",
            )
            footprint = None
        model_metadata = metadata[0].get("model_metadata", {}) if metadata else {}
        model_name = str(model_metadata.get("model", capture_raw.get("model", "legacy")))
        manifest = Manifest(
            run_id=identity.run_id,
            capture_id=identity.capture_id,
            artifact_id=identity.artifact_id,
            created_at=datetime.now(UTC),
            capture=CaptureMetadata(
                video_name=(video.name if video else capture_raw.get("video_name")),
                source_uri=(video.as_uri() if video else None),
                frame_count=capture_raw.get("frames"),
                fps=capture_raw.get("fps"),
                gps_sample_count=len(gps_track) if gps_track is not None else 0,
                imu_sample_count=len(imu_data) if imu_data is not None else 0,
            ),
            producer=Producer(
                model_name=model_name,
                model_config={"legacy_metadata": model_metadata},
                capabilities={
                    "outputs_confidence": _has_dimension(
                        geometry_dimensions, "Confidence"
                    ),
                    "outputs_metric_scale": coordinate_frame.units == "metre",
                },
                git_status="unknown",
                adapter_name="existing_copc_onboarding",
                adapter_version="1.0.0",
                publisher_name="copc_converter",
                publisher_version=(
                    publication.converter_version
                    if publication is not None
                    else COPC_CONVERTER_VERSION
                ),
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
            layer_default=LayerDefault(
                visible=True,
                point_budget=2_000_000,
                point_size=1.0,
                color_dimension=(
                    "rgb"
                    if all(
                        _has_dimension(geometry_dimensions, name)
                        for name in ("Red", "Green", "Blue")
                    )
                    else None
                ),
            ),
        )
        write_manifest(root, manifest)
        validated = self.validator.validate(root)
        catalog_record = Catalog(
            catalog_path, validator=self.validator
        ).register_package(root)
        return ExistingCopcPackageResult(
            package_root=root.resolve(),
            manifest=validated.manifest,
            catalog_record=catalog_record,
            input_sha256=inspection.sha256,
            geometry_sha256=_sha256(geometry),
            geometry_disposition=disposition,
            incompatibilities=inspection.incompatibilities,
            publication=publication,
        )
