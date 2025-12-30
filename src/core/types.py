"""Core data types for the mapper pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


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
                    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ]
            )
            vertex_data['x'] = self.points[:, 0]
            vertex_data['y'] = self.points[:, 1]
            vertex_data['z'] = self.points[:, 2]
            vertex_data['red'] = self.colors[:, 0].astype(np.uint8)
            vertex_data['green'] = self.colors[:, 1].astype(np.uint8)
            vertex_data['blue'] = self.colors[:, 2].astype(np.uint8)
        else:
            vertex_data = np.zeros(
                len(self.points),
                dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            )
            vertex_data['x'] = self.points[:, 0]
            vertex_data['y'] = self.points[:, 1]
            vertex_data['z'] = self.points[:, 2]

        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_element])

        path.parent.mkdir(parents=True, exist_ok=True)
        ply_data.write(str(path))

    @classmethod
    def from_ply(cls, path: Path) -> "PointCloud":
        """Load point cloud from PLY file."""
        from plyfile import PlyData

        ply_data = PlyData.read(str(path))
        vertex = ply_data['vertex']

        points = np.column_stack([
            vertex['x'], vertex['y'], vertex['z']
        ])

        # Try to load colors
        colors = None
        if 'red' in vertex.data.dtype.names:
            colors = np.column_stack([
                vertex['red'], vertex['green'], vertex['blue']
            ])

        return cls(points=points, colors=colors)


@dataclass
class CameraPoses:
    """
    Camera pose output from reconstruction models.

    Poses are 4x4 transformation matrices (world-to-camera or camera-to-world,
    depending on the model - check model documentation).
    """
    poses: np.ndarray  # (M, 4, 4) transformation matrices
    timestamps: Optional[np.ndarray] = None  # (M,) timestamps in seconds
    intrinsics: Optional[np.ndarray] = None  # (3, 3) or (M, 3, 3) camera intrinsics

    def __post_init__(self):
        """Validate pose data."""
        if self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4):
            raise ValueError(f"Poses must be (M, 4, 4), got {self.poses.shape}")

    def __len__(self) -> int:
        return self.poses.shape[0]

    def get_positions(self) -> np.ndarray:
        """Extract camera positions from poses (translation component)."""
        return self.poses[:, :3, 3]

    def get_trajectory_length(self) -> float:
        """Compute total trajectory length."""
        positions = self.get_positions()
        if len(positions) < 2:
            return 0.0

        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))


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

    def __post_init__(self):
        """Validate GPS data."""
        n = len(self.latitudes)
        if len(self.longitudes) != n:
            raise ValueError("Latitudes and longitudes must have same length")

        if self.altitudes is not None and len(self.altitudes) != n:
            raise ValueError("Altitudes must match latitude/longitude length")

        if self.timestamps is not None and len(self.timestamps) != n:
            raise ValueError("Timestamps must match latitude/longitude length")

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

    def to_local_enu(self) -> np.ndarray:
        """
        Convert GPS coordinates to local ENU (East-North-Up) frame.

        Uses the first point as the origin.

        Returns:
            (N, 3) array of [east, north, up] coordinates in meters
        """
        from math import radians, cos

        if len(self) == 0:
            return np.zeros((0, 3))

        # Use first point as origin
        lat0 = radians(self.latitudes[0])
        lon0 = radians(self.longitudes[0])
        alt0 = self.altitudes[0] if self.altitudes is not None else 0.0

        R = 6371000  # Earth radius in meters

        enu = np.zeros((len(self), 3))

        for i in range(len(self)):
            lat = radians(self.latitudes[i])
            lon = radians(self.longitudes[i])
            alt = self.altitudes[i] if self.altitudes is not None else 0.0

            # Simple flat-earth approximation (good for distances < 10km)
            enu[i, 0] = R * (lon - lon0) * cos(lat0)  # East
            enu[i, 1] = R * (lat - lat0)  # North
            enu[i, 2] = alt - alt0  # Up

        return enu


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


@dataclass
class VideoInput:
    """
    Processed video input for reconstruction models.

    Contains the path to extracted frames and optional telemetry data.
    """
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
