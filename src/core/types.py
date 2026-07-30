"""Core data types for the mapper pipeline.

The types in this module are also the boundary between reconstruction/model
code and durable reconstruction packages.  Coordinate-frame and pose
conventions are explicit here so downstream code never has to infer them from
model names.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, List
import numpy as np


class PoseConvention(str, Enum):
    """Meaning of the 4x4 matrices stored in :class:`CameraPoses`."""

    CAMERA_TO_WORLD = "camera_to_world"
    WORLD_TO_CAMERA = "world_to_camera"


class AlignmentStatus(str, Enum):
    """Review state for a model-to-world alignment."""

    UNALIGNED = "unaligned"
    APPROXIMATE = "approximate"
    ALIGNED = "aligned"
    REVIEWED = "reviewed"


@dataclass(frozen=True)
class CoordinateFrame:
    """Machine-readable description of a coordinate frame."""

    name: str
    units: str
    axes: str
    handedness: str = "right"
    vertical_datum: Optional[str] = None

    def __post_init__(self) -> None:
        if self.handedness not in {"left", "right"}:
            raise ValueError("handedness must be 'left' or 'right'")

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "name": self.name,
            "units": self.units,
            "axes": self.axes,
            "handedness": self.handedness,
            "vertical_datum": self.vertical_datum,
        }


MODEL_FRAME = CoordinateFrame(
    name="model",
    units="model_unit",
    axes="model_defined",
)
ARTIFACT_LOCAL_ENU_FRAME = CoordinateFrame(
    name="artifact_local",
    units="metre",
    axes="x=east,y=north,z=up",
    vertical_datum="WGS84 ellipsoidal",
)


def _write_parquet_dataframe(dataframe: Any, path: Path) -> None:
    """Write a DataFrame with an actionable error when no engine is installed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet persistence requires pyarrow (preferred) or fastparquet"
        ) from exc


@dataclass
class PointCloud:
    """
    Unified point cloud output from reconstruction models.

    All models must output at least points. Colors, confidence, and normals
    are optional depending on the model's capabilities.
    """

    points: np.ndarray  # (N, 3) XYZ coordinates
    colors: Optional[np.ndarray] = None  # (N, 3) RGB values 0-255

    # Optional fields - models may or may not provide these
    confidence: Optional[np.ndarray] = None  # (N,) per-point confidence scores
    normals: Optional[np.ndarray] = None  # (N, 3) surface normals

    # Geo-referencing (filled in by alignment step)
    origin_gps: Optional[Tuple[float, float, float]] = None  # (lat, lon, alt)
    scale: float = 1.0  # Scale factor applied to reach metric
    is_metric: bool = False  # Whether the scale is metric (meters)

    def __post_init__(self):
        """Validate point cloud data."""
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"Points must be (N, 3), got {self.points.shape}")

        if self.colors is not None:
            if self.colors.shape != self.points.shape:
                raise ValueError(
                    f"Colors shape {self.colors.shape} must match points shape {self.points.shape}"
                )

        if self.confidence is not None:
            if self.confidence.shape[0] != self.points.shape[0]:
                raise ValueError(
                    f"Confidence length {self.confidence.shape[0]} must match point count {self.points.shape[0]}"
                )

    def __len__(self) -> int:
        return self.points.shape[0]

    def filter_by_confidence(self, min_confidence: float) -> "PointCloud":
        """Return a new PointCloud with points above the confidence threshold."""
        if self.confidence is None:
            return self

        mask = self.confidence >= min_confidence
        return PointCloud(
            points=self.points[mask],
            colors=self.colors[mask] if self.colors is not None else None,
            confidence=self.confidence[mask],
            normals=self.normals[mask] if self.normals is not None else None,
            origin_gps=self.origin_gps,
            scale=self.scale,
            is_metric=self.is_metric,
        )

    def transform(self, matrix: np.ndarray) -> "PointCloud":
        """Apply a 4x4 transformation matrix to the point cloud."""
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4, got {matrix.shape}")

        # Apply rotation and translation
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        transformed_points = (rotation @ self.points.T).T + translation

        # Transform normals if present (rotation only, no translation)
        transformed_normals = None
        if self.normals is not None:
            transformed_normals = (rotation @ self.normals.T).T

        return PointCloud(
            points=transformed_points,
            colors=self.colors,
            confidence=self.confidence,
            normals=transformed_normals,
            origin_gps=self.origin_gps,
            scale=self.scale,
            is_metric=self.is_metric,
        )

    def save_ply(self, path: Path) -> None:
        """Save point cloud to PLY file."""
        from plyfile import PlyData, PlyElement

        # Build vertex data
        if self.colors is not None:
            vertex_data = np.zeros(
                len(self.points),
                dtype=[
                    ("x", "f4"),
                    ("y", "f4"),
                    ("z", "f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            vertex_data["x"] = self.points[:, 0]
            vertex_data["y"] = self.points[:, 1]
            vertex_data["z"] = self.points[:, 2]
            vertex_data["red"] = self.colors[:, 0].astype(np.uint8)
            vertex_data["green"] = self.colors[:, 1].astype(np.uint8)
            vertex_data["blue"] = self.colors[:, 2].astype(np.uint8)
        else:
            vertex_data = np.zeros(
                len(self.points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
            )
            vertex_data["x"] = self.points[:, 0]
            vertex_data["y"] = self.points[:, 1]
            vertex_data["z"] = self.points[:, 2]

        vertex_element = PlyElement.describe(vertex_data, "vertex")
        ply_data = PlyData([vertex_element])

        path.parent.mkdir(parents=True, exist_ok=True)
        ply_data.write(str(path))

    @classmethod
    def from_ply(cls, path: Path) -> "PointCloud":
        """Load point cloud from PLY file."""
        from plyfile import PlyData

        ply_data = PlyData.read(str(path))
        vertex = ply_data["vertex"]

        points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])

        # Try to load colors
        colors = None
        if "red" in vertex.data.dtype.names:
            colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]])

        return cls(points=points, colors=colors)


@dataclass
class CameraPoses:
    """
    Camera pose output from reconstruction models.

    Poses are 4x4 transformation matrices with an explicit convention.
    """

    poses: np.ndarray  # (M, 4, 4) transformation matrices
    timestamps: Optional[np.ndarray] = None  # (M,) timestamps in seconds
    intrinsics: Optional[np.ndarray] = None  # (3, 3) or (M, 3, 3) camera intrinsics
    frame_indices: Optional[np.ndarray] = None  # (M,) indices in original frame list
    pose_convention: PoseConvention = PoseConvention.CAMERA_TO_WORLD
    coordinate_frame: CoordinateFrame = field(default_factory=lambda: MODEL_FRAME)

    def __post_init__(self):
        """Validate pose data."""
        if self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4):
            raise ValueError(f"Poses must be (M, 4, 4), got {self.poses.shape}")
        if self.frame_indices is not None and len(self.frame_indices) != len(
            self.poses
        ):
            raise ValueError("frame_indices length must match pose count")
        if self.timestamps is not None and len(self.timestamps) != len(self.poses):
            raise ValueError("timestamps length must match pose count")
        if isinstance(self.pose_convention, str):
            self.pose_convention = PoseConvention(self.pose_convention)

    def __len__(self) -> int:
        return self.poses.shape[0]

    def get_positions(self) -> np.ndarray:
        """Extract camera positions from poses (translation component)."""
        if self.pose_convention == PoseConvention.CAMERA_TO_WORLD:
            return self.poses[:, :3, 3]
        rotations = self.poses[:, :3, :3]
        translations = self.poses[:, :3, 3]
        return -np.einsum("nij,nj->ni", np.swapaxes(rotations, 1, 2), translations)

    def get_trajectory_length(self) -> float:
        """Compute total trajectory length."""
        positions = self.get_positions()
        if len(positions) < 2:
            return 0.0

        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))

    def to_camera_to_world(self) -> "CameraPoses":
        """Return poses in camera-to-world convention."""

        if self.pose_convention == PoseConvention.CAMERA_TO_WORLD:
            return self
        return CameraPoses(
            poses=np.linalg.inv(self.poses),
            timestamps=self.timestamps,
            intrinsics=self.intrinsics,
            frame_indices=self.frame_indices,
            pose_convention=PoseConvention.CAMERA_TO_WORLD,
            coordinate_frame=self.coordinate_frame,
        )

    def to_dataframe(self):
        """Return the stable, plain-Parquet camera pose table."""

        import pandas as pd
        from scipy.spatial.transform import Rotation

        c2w = self.to_camera_to_world()
        rotations = Rotation.from_matrix(c2w.poses[:, :3, :3])
        quaternions = rotations.as_quat()  # x, y, z, w
        positions = c2w.poses[:, :3, 3]
        count = len(c2w)
        data: dict[str, Any] = {
            "timestamp_s": (
                c2w.timestamps if c2w.timestamps is not None else np.full(count, np.nan)
            ),
            "frame_index": (
                c2w.frame_indices
                if c2w.frame_indices is not None
                else np.arange(count, dtype=np.int64)
            ),
            "tx": positions[:, 0],
            "ty": positions[:, 1],
            "tz": positions[:, 2],
            "qx": quaternions[:, 0],
            "qy": quaternions[:, 1],
            "qz": quaternions[:, 2],
            "qw": quaternions[:, 3],
            "pose_convention": np.repeat(PoseConvention.CAMERA_TO_WORLD.value, count),
            "coordinate_frame": np.repeat(c2w.coordinate_frame.name, count),
        }
        if c2w.intrinsics is not None:
            intrinsics = c2w.intrinsics
            if intrinsics.ndim == 2:
                intrinsics = np.repeat(intrinsics[None, :, :], count, axis=0)
            data.update(
                {
                    "fx": intrinsics[:, 0, 0],
                    "fy": intrinsics[:, 1, 1],
                    "cx": intrinsics[:, 0, 2],
                    "cy": intrinsics[:, 1, 2],
                }
            )
        else:
            data.update(
                {
                    "fx": np.full(count, np.nan),
                    "fy": np.full(count, np.nan),
                    "cx": np.full(count, np.nan),
                    "cy": np.full(count, np.nan),
                }
            )
        return pd.DataFrame(data)

    def to_parquet(self, path: Path) -> None:
        """Persist poses using the canonical plain-Parquet representation."""

        _write_parquet_dataframe(self.to_dataframe(), path)


@dataclass
class ReconstructionResult:
    """
    Unified output from all reconstruction models.

    Every model must return at least a PointCloud. Poses and other
    metadata are optional depending on the model's capabilities.
    """

    pointcloud: PointCloud
    poses: Optional[CameraPoses] = None
    metadata: dict = field(default_factory=dict)  # Model-specific info
    chunks: Optional[List["ReconstructionResult"]] = None
    window_metadata: dict = field(default_factory=dict)


@dataclass
class GPSTrack:
    """
    GPS trajectory from video telemetry.

    Typically extracted from GoPro or phone video metadata.
    """

    latitudes: np.ndarray  # (N,) degrees
    longitudes: np.ndarray  # (N,) degrees
    altitudes: Optional[np.ndarray] = None  # (N,) meters
    timestamps: Optional[np.ndarray] = None  # (N,) seconds from video start
    accuracies: Optional[np.ndarray] = None  # (N,) horizontal accuracy in meters
    fixes: Optional[np.ndarray] = None  # (N,) 0=no lock, 2=2D, 3=3D
    position_dops: Optional[np.ndarray] = None  # (N,) GPS dilution of precision

    def __post_init__(self):
        """Validate GPS data."""
        n = len(self.latitudes)
        if len(self.longitudes) != n:
            raise ValueError("Latitudes and longitudes must have same length")

        if self.altitudes is not None and len(self.altitudes) != n:
            raise ValueError("Altitudes must match latitude/longitude length")

        if self.timestamps is not None and len(self.timestamps) != n:
            raise ValueError("Timestamps must match latitude/longitude length")
        if self.accuracies is not None and len(self.accuracies) != n:
            raise ValueError("Accuracies must match latitude/longitude length")
        if self.fixes is not None and len(self.fixes) != n:
            raise ValueError("GPS fixes must match latitude/longitude length")
        if self.position_dops is not None and len(self.position_dops) != n:
            raise ValueError("GPS DOP values must match latitude/longitude length")

    def __len__(self) -> int:
        return len(self.latitudes)

    def get_trajectory_length_meters(self) -> float:
        """
        Compute total GPS trajectory length in meters using haversine formula.
        """
        from math import radians, sin, cos, sqrt, atan2

        if len(self) < 2:
            return 0.0

        total = 0.0
        R = 6371000  # Earth radius in meters

        for i in range(1, len(self)):
            lat1 = radians(self.latitudes[i - 1])
            lat2 = radians(self.latitudes[i])
            lon1 = radians(self.longitudes[i - 1])
            lon2 = radians(self.longitudes[i])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            total += R * c

        return total

    def quality_mask(
        self,
        *,
        min_fix: int = 3,
        max_accuracy_m: float = 20.0,
        max_position_dop: float = 5.0,
    ) -> np.ndarray:
        """Return samples suitable for authoritative georeferencing."""

        mask = (
            np.isfinite(self.latitudes)
            & np.isfinite(self.longitudes)
            & (self.latitudes >= -90.0)
            & (self.latitudes <= 90.0)
            & (self.longitudes >= -180.0)
            & (self.longitudes <= 180.0)
            & ~((self.latitudes == 0.0) & (self.longitudes == 0.0))
        )
        if self.altitudes is not None:
            mask &= np.isfinite(self.altitudes)
        if self.timestamps is not None:
            mask &= np.isfinite(self.timestamps)
        if self.fixes is not None:
            mask &= np.isfinite(self.fixes) & (self.fixes >= min_fix)
        if self.accuracies is not None:
            mask &= (
                np.isfinite(self.accuracies)
                & (self.accuracies >= 0.0)
                & (self.accuracies <= max_accuracy_m)
            )
        if self.position_dops is not None:
            mask &= (
                np.isfinite(self.position_dops)
                & (self.position_dops >= 0.0)
                & (self.position_dops <= max_position_dop)
            )
        return mask

    def filter_quality(
        self,
        *,
        min_fix: int = 3,
        max_accuracy_m: float = 20.0,
        max_position_dop: float = 5.0,
    ) -> "GPSTrack":
        """Return a track containing only quality-approved samples."""

        mask = self.quality_mask(
            min_fix=min_fix,
            max_accuracy_m=max_accuracy_m,
            max_position_dop=max_position_dop,
        )

        def selected(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return values[mask] if values is not None else None

        return GPSTrack(
            latitudes=self.latitudes[mask],
            longitudes=self.longitudes[mask],
            altitudes=selected(self.altitudes),
            timestamps=selected(self.timestamps),
            accuracies=selected(self.accuracies),
            fixes=selected(self.fixes),
            position_dops=selected(self.position_dops),
        )

    def to_ecef(self) -> np.ndarray:
        """Convert WGS84 geodetic samples to float64 ECEF metres."""

        lat = np.deg2rad(np.asarray(self.latitudes, dtype=np.float64))
        lon = np.deg2rad(np.asarray(self.longitudes, dtype=np.float64))
        alt = (
            np.asarray(self.altitudes, dtype=np.float64)
            if self.altitudes is not None
            else np.zeros(len(self), dtype=np.float64)
        )
        semi_major = 6378137.0
        eccentricity_sq = 6.69437999014e-3
        sin_lat = np.sin(lat)
        prime_vertical = semi_major / np.sqrt(1.0 - eccentricity_sq * sin_lat**2)
        return np.column_stack(
            (
                (prime_vertical + alt) * np.cos(lat) * np.cos(lon),
                (prime_vertical + alt) * np.cos(lat) * np.sin(lon),
                (prime_vertical * (1.0 - eccentricity_sq) + alt) * sin_lat,
            )
        )

    def robust_anchor(
        self, *, max_iterations: int = 20
    ) -> Tuple[Tuple[float, float, float], np.ndarray]:
        """Compute a Huber-weighted ECEF centroid and its WGS84 coordinates."""

        if len(self) == 0:
            raise ValueError("Cannot compute an anchor for an empty GPS track")
        ecef = self.to_ecef()
        base_weights = np.ones(len(self), dtype=np.float64)
        if self.accuracies is not None:
            accuracy = np.maximum(np.asarray(self.accuracies, dtype=float), 0.25)
            base_weights = 1.0 / np.square(accuracy)
        center = np.average(ecef, axis=0, weights=base_weights)
        for _ in range(max_iterations):
            residuals = np.linalg.norm(ecef - center, axis=1)
            median = np.median(residuals)
            mad = np.median(np.abs(residuals - median))
            sigma = max(1.4826 * mad, 0.1)
            cutoff = 1.5 * sigma
            robust_weights = np.minimum(1.0, cutoff / np.maximum(residuals, 1e-12))
            weights = base_weights * robust_weights
            updated = np.average(ecef, axis=0, weights=weights)
            if np.linalg.norm(updated - center) < 1e-6:
                center = updated
                break
            center = updated
        return _ecef_to_wgs84(center), center.astype(np.float64)

    def to_local_enu(
        self,
        anchor_wgs84: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """
        Convert GPS coordinates to local ENU (East-North-Up) frame.

        Uses a robust centroid as the origin unless an explicit anchor is
        supplied.  Conversion is WGS84 -> ECEF -> ENU, not a flat-earth
        approximation.

        Returns:
            (N, 3) array of [east, north, up] coordinates in meters
        """
        if len(self) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if anchor_wgs84 is None:
            anchor_wgs84, anchor_ecef = self.robust_anchor()
        else:
            anchor_track = GPSTrack(
                latitudes=np.array([anchor_wgs84[0]], dtype=np.float64),
                longitudes=np.array([anchor_wgs84[1]], dtype=np.float64),
                altitudes=np.array([anchor_wgs84[2]], dtype=np.float64),
            )
            anchor_ecef = anchor_track.to_ecef()[0]
        lat0 = np.deg2rad(anchor_wgs84[0])
        lon0 = np.deg2rad(anchor_wgs84[1])
        ecef_to_enu = np.array(
            [
                [-np.sin(lon0), np.cos(lon0), 0.0],
                [
                    -np.sin(lat0) * np.cos(lon0),
                    -np.sin(lat0) * np.sin(lon0),
                    np.cos(lat0),
                ],
                [
                    np.cos(lat0) * np.cos(lon0),
                    np.cos(lat0) * np.sin(lon0),
                    np.sin(lat0),
                ],
            ],
            dtype=np.float64,
        )
        return (ecef_to_enu @ (self.to_ecef() - anchor_ecef).T).T

    def to_dataframe(self):
        """Return the stable GPS telemetry table."""

        import pandas as pd

        count = len(self)
        return pd.DataFrame(
            {
                "timestamp_s": (
                    self.timestamps
                    if self.timestamps is not None
                    else np.full(count, np.nan)
                ),
                "longitude_deg": self.longitudes,
                "latitude_deg": self.latitudes,
                "ellipsoidal_height_m": (
                    self.altitudes
                    if self.altitudes is not None
                    else np.full(count, np.nan)
                ),
                "fix_type": (
                    self.fixes if self.fixes is not None else np.full(count, np.nan)
                ),
                "hdop": (
                    self.position_dops
                    if self.position_dops is not None
                    else np.full(count, np.nan)
                ),
                "horizontal_accuracy_m": (
                    self.accuracies
                    if self.accuracies is not None
                    else np.full(count, np.nan)
                ),
            }
        )

    def to_parquet(self, path: Path) -> None:
        """Persist GPS telemetry as plain Parquet."""

        _write_parquet_dataframe(self.to_dataframe(), path)


def _ecef_to_wgs84(ecef: np.ndarray) -> Tuple[float, float, float]:
    """Convert one float64 ECEF coordinate to WGS84 geodetic."""

    x, y, z = np.asarray(ecef, dtype=np.float64)
    semi_major = 6378137.0
    eccentricity_sq = 6.69437999014e-3
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - eccentricity_sq))
    for _ in range(10):
        sin_lat = np.sin(lat)
        prime_vertical = semi_major / np.sqrt(1.0 - eccentricity_sq * sin_lat**2)
        alt = p / max(np.cos(lat), 1e-15) - prime_vertical
        updated = np.arctan2(
            z,
            p * (1.0 - eccentricity_sq * prime_vertical / (prime_vertical + alt)),
        )
        if abs(updated - lat) < 1e-14:
            lat = updated
            break
        lat = updated
    sin_lat = np.sin(lat)
    prime_vertical = semi_major / np.sqrt(1.0 - eccentricity_sq * sin_lat**2)
    alt = p / max(np.cos(lat), 1e-15) - prime_vertical
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(alt)


@dataclass
class IMUData:
    """
    IMU data from video telemetry (GoPro, phone, etc).

    Contains accelerometer, gyroscope, and optionally gravity vectors
    and camera orientations.
    """

    accelerometer: np.ndarray  # (N, 3) m/s^2
    gyroscope: np.ndarray  # (N, 3) rad/s
    timestamps: np.ndarray  # (N,) seconds from video start

    # Optional data (available on Hero8+)
    gravity_vectors: Optional[np.ndarray] = None  # (N, 3) normalized gravity direction
    orientations: Optional[np.ndarray] = None  # (N, 4) quaternions (w, x, y, z)

    def __post_init__(self):
        """Validate IMU data."""
        n = len(self.timestamps)

        if self.accelerometer.shape[0] != n:
            raise ValueError("Accelerometer samples must match timestamp count")

        if self.gyroscope.shape[0] != n:
            raise ValueError("Gyroscope samples must match timestamp count")
        if self.gravity_vectors is not None and len(self.gravity_vectors) != n:
            raise ValueError("Gravity samples must match timestamp count")
        if self.orientations is not None and len(self.orientations) != n:
            raise ValueError("Orientation samples must match timestamp count")

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_gravity_direction(self) -> Optional[np.ndarray]:
        """
        Get average gravity direction from gravity vectors or accelerometer.

        Returns:
            (3,) normalized gravity vector, or None if not available
        """
        if self.gravity_vectors is not None and len(self.gravity_vectors) > 0:
            # Average gravity vectors
            avg = np.mean(self.gravity_vectors, axis=0)
            return avg / np.linalg.norm(avg)

        if len(self.accelerometer) > 0:
            # During stationary periods, accelerometer measures gravity
            # This is a rough approximation
            avg = np.mean(self.accelerometer, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                return avg / norm

        return None

    def to_dataframe(self):
        """Return the stable IMU telemetry table."""

        import pandas as pd

        data: dict[str, Any] = {
            "timestamp_s": self.timestamps,
            "accel_x_mps2": self.accelerometer[:, 0],
            "accel_y_mps2": self.accelerometer[:, 1],
            "accel_z_mps2": self.accelerometer[:, 2],
            "gyro_x_radps": self.gyroscope[:, 0],
            "gyro_y_radps": self.gyroscope[:, 1],
            "gyro_z_radps": self.gyroscope[:, 2],
        }
        if self.gravity_vectors is not None:
            data.update(
                {
                    "gravity_x": self.gravity_vectors[:, 0],
                    "gravity_y": self.gravity_vectors[:, 1],
                    "gravity_z": self.gravity_vectors[:, 2],
                }
            )
        if self.orientations is not None:
            data.update(
                {
                    "orientation_w": self.orientations[:, 0],
                    "orientation_x": self.orientations[:, 1],
                    "orientation_y": self.orientations[:, 2],
                    "orientation_z": self.orientations[:, 3],
                }
            )
        return pd.DataFrame(data)

    def to_parquet(self, path: Path) -> None:
        """Persist IMU telemetry as plain Parquet."""

        _write_parquet_dataframe(self.to_dataframe(), path)


@dataclass
class AlignmentResult:
    """Complete outcome of one GPS/model alignment attempt.

    ``transform`` maps model coordinates to artifact-local ENU coordinates and
    includes similarity scale in its upper-left 3x3 block.  Durable publishers
    should only use the transformed cloud when :attr:`accepted` is true.
    """

    transform: np.ndarray
    enu_to_ecef_transform: np.ndarray
    scale: float
    method: str
    status: AlignmentStatus
    inlier_count: int
    correspondence_count: int
    rmse_m: Optional[float]
    horizontal_rmse_m: Optional[float]
    vertical_rmse_m: Optional[float]
    anchor_wgs84: Optional[Tuple[float, float, float]]
    anchor_ecef: Optional[np.ndarray]
    aligned_pointcloud: Optional[PointCloud] = None
    clock_offset_s: float = 0.0
    clock_peak_quality: float = 0.0
    reason: Optional[str] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.transform = np.asarray(self.transform, dtype=np.float64)
        self.enu_to_ecef_transform = np.asarray(
            self.enu_to_ecef_transform, dtype=np.float64
        )
        if self.transform.shape != (4, 4):
            raise ValueError("Alignment transform must be 4x4")
        if self.enu_to_ecef_transform.shape != (4, 4):
            raise ValueError("ENU-to-ECEF transform must be 4x4")
        if isinstance(self.status, str):
            self.status = AlignmentStatus(self.status)
        if self.anchor_ecef is not None:
            self.anchor_ecef = np.asarray(self.anchor_ecef, dtype=np.float64)
            if self.anchor_ecef.shape != (3,):
                raise ValueError("anchor_ecef must have shape (3,)")

    @property
    def accepted(self) -> bool:
        return self.status != AlignmentStatus.UNALIGNED

    @classmethod
    def unaligned(
        cls,
        reason: str,
        *,
        method: str = "gps_weighted_umeyama",
        correspondence_count: int = 0,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> "AlignmentResult":
        return cls(
            transform=np.eye(4, dtype=np.float64),
            enu_to_ecef_transform=np.eye(4, dtype=np.float64),
            scale=1.0,
            method=method,
            status=AlignmentStatus.UNALIGNED,
            inlier_count=0,
            correspondence_count=correspondence_count,
            rmse_m=None,
            horizontal_rmse_m=None,
            vertical_rmse_m=None,
            anchor_wgs84=None,
            anchor_ecef=None,
            reason=reason,
            diagnostics=diagnostics or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest-safe alignment record."""

        return {
            "transform": self.transform.tolist(),
            "enu_to_ecef_transform": self.enu_to_ecef_transform.tolist(),
            "scale": float(self.scale),
            "method": self.method,
            "status": self.status.value,
            "inlier_count": int(self.inlier_count),
            "correspondence_count": int(self.correspondence_count),
            "rmse_m": None if self.rmse_m is None else float(self.rmse_m),
            "horizontal_rmse_m": (
                None
                if self.horizontal_rmse_m is None
                else float(self.horizontal_rmse_m)
            ),
            "vertical_rmse_m": (
                None if self.vertical_rmse_m is None else float(self.vertical_rmse_m)
            ),
            # Package/manifests consistently serialize WGS84 geodetic
            # coordinates as GeoJSON order: longitude, latitude, height.
            "anchor_lon_lat_height": (
                list(self.anchor_lon_lat_height)
                if self.anchor_lon_lat_height is not None
                else None
            ),
            "anchor_ecef": (
                self.anchor_ecef.tolist() if self.anchor_ecef is not None else None
            ),
            "clock_offset_s": float(self.clock_offset_s),
            "clock_peak_quality": float(self.clock_peak_quality),
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }

    def transform_poses(self, poses: CameraPoses) -> CameraPoses:
        """Transform poses into the same artifact-local ENU frame as geometry."""

        if not self.accepted:
            raise ValueError("Cannot transform poses with a rejected alignment")
        c2w = poses.to_camera_to_world()
        if not np.isfinite(self.scale) or abs(self.scale) < 1e-15:
            raise ValueError("Alignment scale must be finite and non-zero")
        alignment_rotation = self.transform[:3, :3] / self.scale
        transformed = np.asarray(c2w.poses, dtype=np.float64).copy()
        transformed[:, :3, :3] = np.einsum(
            "ij,njk->nik", alignment_rotation, c2w.poses[:, :3, :3]
        )
        transformed[:, :3, 3] = (
            self.transform[:3, :3] @ c2w.poses[:, :3, 3].T
        ).T + self.transform[:3, 3]
        return CameraPoses(
            poses=transformed,
            timestamps=c2w.timestamps,
            intrinsics=c2w.intrinsics,
            frame_indices=c2w.frame_indices,
            pose_convention=PoseConvention.CAMERA_TO_WORLD,
            coordinate_frame=ARTIFACT_LOCAL_ENU_FRAME,
        )

    @property
    def anchor_lon_lat_height(self) -> Optional[Tuple[float, float, float]]:
        """Manifest-order WGS84 anchor: longitude, latitude, height.

        ``anchor_wgs84`` and legacy :attr:`PointCloud.origin_gps` remain in
        latitude/longitude/altitude order for compatibility with current model
        and viewer code.  Callers writing durable metadata must use this
        explicitly ordered property.
        """

        if self.anchor_wgs84 is None:
            return None
        latitude, longitude, height = self.anchor_wgs84
        return float(longitude), float(latitude), float(height)


@dataclass
class VideoInput:
    """
    Processed video input for reconstruction models.

    Contains the path to extracted frames and optional telemetry data.
    """

    video_path: Path  # Path to the original video file
    image_dir: Path  # Directory containing extracted frames
    fps: float  # Frame extraction rate
    frame_count: int  # Number of extracted frames

    # Optional telemetry (extracted from GoPro/phone)
    gps_track: Optional[GPSTrack] = None
    imu_data: Optional[IMUData] = None

    # Video metadata
    metadata: dict = field(default_factory=dict)

    def get_frame_paths(self) -> list[Path]:
        """Get sorted list of frame image paths."""
        return sorted(self.image_dir.glob("frame_*.jpg"))

    def get_frame_timestamps(self) -> np.ndarray:
        """Get timestamps for each frame based on FPS."""
        return np.arange(self.frame_count) / self.fps
