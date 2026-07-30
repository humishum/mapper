"""Strict, model-independent reconstruction package v1 contract."""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

POSE_COLUMNS = frozenset(
    {
        "timestamp_s",
        "frame_index",
        "tx",
        "ty",
        "tz",
        "qx",
        "qy",
        "qz",
        "qw",
        "fx",
        "fy",
        "cx",
        "cy",
    }
)
GPS_COLUMNS = frozenset(
    {
        "timestamp_s",
        "longitude_deg",
        "latitude_deg",
        "ellipsoidal_height_m",
        "fix_type",
        "hdop",
    }
)
IMU_COLUMNS = frozenset(
    {
        "timestamp_s",
        "accel_x_mps2",
        "accel_y_mps2",
        "accel_z_mps2",
        "gyro_x_radps",
        "gyro_y_radps",
        "gyro_z_radps",
    }
)
TABULAR_COLUMN_DTYPES: dict[str, dict[str, str]] = {
    "poses": {
        **{name: "float64" for name in POSE_COLUMNS},
        "frame_index": "int64",
    },
    "telemetry_gps": {
        **{name: "float64" for name in GPS_COLUMNS},
        "fix_type": "int64",
    },
    "telemetry_imu": {name: "float64" for name in IMU_COLUMNS},
}


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AlignmentStatus(StrEnum):
    UNALIGNED = "unaligned"
    APPROXIMATE = "approximate"
    ALIGNED = "aligned"
    REVIEWED = "reviewed"


class WGS84Footprint(ContractModel):
    """A single exterior WGS84 ring, in longitude/latitude order."""

    coordinates: list[tuple[float, float]] = Field(min_length=4)

    @field_validator("coordinates")
    @classmethod
    def validate_ring(
        cls, coordinates: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        for longitude, latitude in coordinates:
            if not (math.isfinite(longitude) and math.isfinite(latitude)):
                raise ValueError("footprint coordinates must be finite")
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError("footprint coordinates must be valid WGS84 lon/lat")
        if coordinates[0] != coordinates[-1]:
            raise ValueError("footprint exterior ring must be closed")
        if len(set(coordinates[:-1])) < 3:
            raise ValueError("footprint must contain at least three distinct vertices")
        return coordinates

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        longitudes, latitudes = zip(*self.coordinates, strict=True)
        return (
            min(longitudes),
            min(latitudes),
            max(longitudes),
            max(latitudes),
        )

    def as_geojson(self) -> dict[str, Any]:
        return {"type": "Polygon", "coordinates": [self.coordinates]}


class CaptureMetadata(ContractModel):
    started_at: datetime | None = None
    ended_at: datetime | None = None
    device: str | None = None
    lens: str | None = None
    video_name: str | None = None
    source_uri: str | None = None
    frame_count: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)
    gps_sample_count: int | None = Field(default=None, ge=0)
    imu_sample_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_times(self) -> "CaptureMetadata":
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("capture ended_at must not precede started_at")
        return self


class Producer(ContractModel):
    model_name: str = Field(min_length=1)
    model_version: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    capabilities: dict[str, bool] = Field(default_factory=dict)
    git_commit: str | None = None
    git_status: Literal["clean", "dirty", "unknown"] = "unknown"
    adapter_name: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    publisher_name: str = Field(min_length=1)
    publisher_version: str = Field(min_length=1)


class CoordinateFrame(ContractModel):
    name: str = Field(default="artifact_local", min_length=1)
    units: Literal["metre", "unknown"]
    axis_order: list[str] = Field(min_length=3, max_length=3)
    handedness: Literal["right", "left", "unknown"]
    pose_convention: str | None = None
    origin_wgs84: tuple[float, float, float] | None = None
    proj_pipeline: str | None = None
    transform_to_ecef: list[float] | None = None
    crs: str | None = None

    @field_validator("transform_to_ecef")
    @classmethod
    def validate_transform(cls, transform: list[float] | None) -> list[float] | None:
        if transform is not None:
            if len(transform) != 16:
                raise ValueError("transform_to_ecef must be a row-major 4x4 matrix")
            if not all(math.isfinite(value) for value in transform):
                raise ValueError("transform_to_ecef values must be finite")
            if any(
                not math.isclose(value, expected, abs_tol=1e-12)
                for value, expected in zip(
                    transform[12:16], (0.0, 0.0, 0.0, 1.0), strict=True
                )
            ):
                raise ValueError("transform_to_ecef must be an affine 4x4 matrix")
        return transform

    @field_validator("origin_wgs84")
    @classmethod
    def validate_origin(
        cls, origin: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if origin is None:
            return None
        longitude, latitude, height = origin
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("origin_wgs84 must be [longitude, latitude, height]")
        if not all(math.isfinite(value) for value in origin):
            raise ValueError("origin_wgs84 values must be finite")
        return origin


class Alignment(ContractModel):
    status: AlignmentStatus
    method: str
    model_to_artifact_local: list[float] = Field(min_length=16, max_length=16)
    scale: float = Field(gt=0)
    inlier_count: int = Field(ge=0)
    horizontal_rmse_m: float | None = Field(default=None, ge=0)
    vertical_rmse_m: float | None = Field(default=None, ge=0)
    clock_offset_s: float | None = None
    clock_peak_quality: float | None = Field(default=None, ge=0, le=1)
    gravity_constrained: bool = False
    rejection_reason: str | None = None

    @field_validator("model_to_artifact_local")
    @classmethod
    def validate_matrix(cls, transform: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in transform):
            raise ValueError("alignment matrix values must be finite")
        if any(
            not math.isclose(value, expected, abs_tol=1e-12)
            for value, expected in zip(
                transform[12:16], (0.0, 0.0, 0.0, 1.0), strict=True
            )
        ):
            raise ValueError("alignment matrix must be affine")
        return transform

    @model_validator(mode="after")
    def require_rejection_reason(self) -> "Alignment":
        if self.status == AlignmentStatus.UNALIGNED and not self.rejection_reason:
            raise ValueError("unaligned results must state a rejection_reason")
        return self


class ColumnContract(ContractModel):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    unit: str | None = None
    nullable: bool = False


class ArtifactFile(ContractModel):
    representation_id: str = Field(min_length=1, max_length=200)
    kind: Literal[
        "points",
        "mesh",
        "splats",
        "poses",
        "telemetry_gps",
        "telemetry_imu",
        "sources",
        "metrics",
        "raw",
    ]
    format: str = Field(min_length=1)
    path: str = Field(min_length=1)
    media_type: str
    byte_size: int = Field(ge=0)
    sha256: str
    frame: str | None = None
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None
    point_count: int | None = Field(default=None, ge=0)
    required_dimensions: list[str] = Field(default_factory=list)
    columns: list[ColumnContract] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, path: str) -> str:
        pure_path = PurePosixPath(path)
        if pure_path.is_absolute() or ".." in pure_path.parts or path in {"", "."}:
            raise ValueError("artifact path must be a contained, relative POSIX path")
        return path

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, checksum: str) -> str:
        checksum = checksum.lower()
        if not SHA256_RE.fullmatch(checksum):
            raise ValueError("sha256 must contain exactly 64 lowercase hex characters")
        return checksum

    @model_validator(mode="after")
    def validate_kind_contract(self) -> "ArtifactFile":
        columns = {column.name: column for column in self.columns}
        if len(columns) != len(self.columns):
            raise ValueError("column contract names must be unique")
        required_columns = {
            "poses": POSE_COLUMNS,
            "telemetry_gps": GPS_COLUMNS,
            "telemetry_imu": IMU_COLUMNS,
        }.get(self.kind)
        if required_columns and not required_columns.issubset(columns.keys()):
            missing = ", ".join(sorted(required_columns - columns.keys()))
            raise ValueError(f"{self.kind} column contract is missing: {missing}")
        if required_columns:
            if self.format.lower() != "parquet" or not self.path.endswith(".parquet"):
                raise ValueError(f"{self.kind} representations must be Parquet files")
            expected_dtypes = TABULAR_COLUMN_DTYPES[self.kind]
            for name, expected_dtype in expected_dtypes.items():
                if columns[name].dtype != expected_dtype:
                    raise ValueError(
                        f"{self.kind} column {name!r} must declare dtype "
                        f"{expected_dtype!r}"
                    )
        if self.kind == "points":
            if self.point_count is None or self.frame is None:
                raise ValueError("point representations require point_count and frame")
            if self.bounds_min is None or self.bounds_max is None:
                raise ValueError("point representations require local bounds")
            if not {"X", "Y", "Z"}.issubset(self.required_dimensions):
                raise ValueError("point representations require X, Y, and Z dimensions")
            bounds_min = self.bounds_min
            bounds_max = self.bounds_max
            if not all(math.isfinite(value) for value in (*bounds_min, *bounds_max)):
                raise ValueError("point bounds must be finite")
            if any(
                low > high for low, high in zip(bounds_min, bounds_max, strict=True)
            ):
                raise ValueError("point bounds minimums must not exceed maximums")
        return self


class LayerDefault(ContractModel):
    visible: bool = True
    point_budget: int | None = Field(default=None, gt=0)
    point_size: float | None = Field(default=None, gt=0)
    color_dimension: str | None = None
    opacity: float = Field(default=1.0, ge=0, le=1)


class LossyOperation(ContractModel):
    operation: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    points_in: int | None = Field(default=None, ge=0)
    points_out: int | None = Field(default=None, ge=0)
    provenance_policy: str | None = None


class Manifest(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1, max_length=200)
    capture_id: str = Field(min_length=1, max_length=200)
    artifact_id: str = Field(min_length=1, max_length=200)
    created_at: datetime
    capture: CaptureMetadata
    producer: Producer
    coordinate_frame: CoordinateFrame
    alignment: Alignment
    footprint_wgs84: WGS84Footprint | None = None
    artifacts: list[ArtifactFile] = Field(min_length=1)
    lossy_operations: list[LossyOperation] = Field(default_factory=list)
    layer_default: LayerDefault | None = None

    @model_validator(mode="after")
    def validate_package(self) -> "Manifest":
        representation_ids = [artifact.representation_id for artifact in self.artifacts]
        if len(set(representation_ids)) != len(representation_ids):
            raise ValueError("representation_id values must be unique within a package")
        paths = [artifact.path for artifact in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("artifact paths must be unique within a package")
        for artifact in self.artifacts:
            if (
                artifact.kind in {"points", "mesh", "splats", "poses"}
                and artifact.frame is not None
                and artifact.frame != self.coordinate_frame.name
            ):
                raise ValueError(
                    f"{artifact.kind} representation frame {artifact.frame!r} "
                    f"does not match package frame {self.coordinate_frame.name!r}"
                )

        is_global = self.alignment.status != AlignmentStatus.UNALIGNED
        frame = self.coordinate_frame
        if is_global:
            if frame.units != "metre":
                raise ValueError("globally placed artifacts must use metre units")
            if frame.origin_wgs84 is None or frame.transform_to_ecef is None:
                raise ValueError(
                    "globally placed artifacts require origin_wgs84 and transform_to_ecef"
                )
            if self.footprint_wgs84 is None:
                raise ValueError("globally placed artifacts require footprint_wgs84")
        else:
            forbidden = (
                frame.origin_wgs84,
                frame.proj_pipeline,
                frame.transform_to_ecef,
                frame.crs,
                self.footprint_wgs84,
            )
            if any(value is not None for value in forbidden):
                raise ValueError(
                    "unaligned artifacts cannot claim an origin, projection, CRS, "
                    "ECEF transform, or WGS84 footprint"
                )
        return self


class SourceRecord(ContractModel):
    source_index: int = Field(ge=0, le=65535)
    kind: Literal["window", "submap", "keyframe_group", "batch", "capture"]
    capture_id: str
    run_id: str
    name: str | None = None
    frame_start: int | None = Field(default=None, ge=0)
    frame_end: int | None = Field(default=None, ge=0)
    frame_indices: list[int] | None = None
    timestamp_start_s: float | None = None
    timestamp_end_s: float | None = None
    point_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SourceRecord":
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and self.frame_end < self.frame_start
        ):
            raise ValueError("source frame_end must not precede frame_start")
        if (
            self.timestamp_start_s is not None
            and self.timestamp_end_s is not None
            and self.timestamp_end_s < self.timestamp_start_s
        ):
            raise ValueError(
                "source timestamp_end_s must not precede timestamp_start_s"
            )
        return self


class SourcesDocument(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    provenance_dimension: Literal["PointSourceId", "SourceIndex"]
    granularity: Literal["window", "submap", "keyframe_group", "batch", "capture"]
    sources: list[SourceRecord]

    @model_validator(mode="after")
    def validate_sources(self) -> "SourcesDocument":
        indexes = [source.source_index for source in self.sources]
        if len(set(indexes)) != len(indexes):
            raise ValueError("source_index values must be unique")
        return self


class StageMetric(ContractModel):
    name: str
    wall_time_s: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Metrics(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    stages: list[StageMetric] = Field(default_factory=list)
    total_wall_time_s: float | None = Field(default=None, ge=0)
    peak_rss_bytes: int | None = Field(default=None, ge=0)
    input_point_count: int | None = Field(default=None, ge=0)
    output_point_count: int | None = Field(default=None, ge=0)
    input_byte_size: int | None = Field(default=None, ge=0)
    output_byte_size: int | None = Field(default=None, ge=0)
    validation: dict[str, bool | int | float | str | None] = Field(default_factory=dict)
