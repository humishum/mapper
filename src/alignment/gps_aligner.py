"""GPS-based alignment and scale recovery.
# This should evnetually be replaced/updated to do a bundle adjustment 

"""

from typing import Optional, Tuple
import logging
import numpy as np

from ..core.types import PointCloud, CameraPoses, GPSTrack, IMUData

logger = logging.getLogger(__name__)

# Alignment guardrails for sparse GPS tracks.
MIN_GPS_TRAJECTORY_LENGTH_M = 2.0
MIN_GPS_STD_DEV_M = 0.5


class GPSAligner:
    """
    Align reconstruction to GPS coordinate system.

    This class handles two main tasks:
    1. Scale recovery - Match reconstruction trajectory length to GPS trajectory
    2. Position/rotation alignment - Align reconstruction to GPS coordinates
    """

    def __init__(self):
        pass

    def align(
        self,
        pointcloud: PointCloud,
        poses: CameraPoses,
        gps_track: GPSTrack,
        imu_data: Optional[IMUData] = None,
        allow_scale: bool = True,
    ) -> PointCloud:
        """
        Align point cloud to GPS coordinates.

        Steps:
        1. Compute scale from GPS vs pose trajectory lengths
        2. Compute rotation to align trajectories (Kabsch algorithm)
        3. Compute translation to GPS origin
        4. Apply transformation to point cloud

        Args:
            pointcloud: Point cloud to align
            poses: Camera poses from reconstruction
            gps_track: GPS trajectory from video telemetry
            imu_data: Optional IMU data for gravity alignment

        Returns:
            Aligned PointCloud with is_metric=True
        """
        if len(poses) < 2:
            logger.warning("Not enough poses for alignment, returning original")
            return pointcloud

        if len(gps_track) < 2:
            logger.warning("Not enough GPS points for alignment, returning original")
            return pointcloud

        # Step 1: Scale recovery (optional)
        if allow_scale:
            scale = self.compute_scale(poses, gps_track)
            if not np.isfinite(scale):
                logger.warning("Non-finite scale factor, skipping GPS alignment")
                return pointcloud
            logger.info(f"Computed scale factor: {scale:.4f}")
        else:
            scale = 1.0
            logger.info("Skipping GPS scale adjustment (metric alignment)")

        # Step 2: Get positions in local ENU coordinates
        pose_positions = poses.get_positions()
        gps_enu = gps_track.to_local_enu()
        if not np.isfinite(pose_positions).all() or not np.isfinite(gps_enu).all():
            logger.warning("Non-finite pose or GPS positions, skipping GPS alignment")
            return pointcloud
        if not np.isfinite(pointcloud.points).all():
            logger.warning("Non-finite pointcloud points detected, skipping GPS alignment")
            return pointcloud
        gps_traj_len = gps_track.get_trajectory_length_meters()
        gps_std = float(np.mean(np.std(gps_enu, axis=0)))
        if gps_traj_len < MIN_GPS_TRAJECTORY_LENGTH_M or gps_std < MIN_GPS_STD_DEV_M:
            logger.warning(
                "GPS track lacks motion (length=%.3f m, std=%.3f m), skipping GPS alignment",
                gps_traj_len,
                gps_std,
            )
            return pointcloud
        
        logger.debug(
            "GPS align stats: pose min %s max %s | gps min %s max %s",
            np.min(pose_positions, axis=0),
            np.max(pose_positions, axis=0),
            np.min(gps_enu, axis=0),
            np.max(gps_enu, axis=0),
        )

        # Step 3: Subsample to match counts
        pose_positions, gps_enu = self._align_sample_counts(
            pose_positions, gps_enu
        )

        # Step 4: Apply scale to pose positions
        scaled_poses = pose_positions * scale

        # Step 5: Compute rotation and translation (Kabsch algorithm)
        rotation, translation = self._kabsch_align(scaled_poses, gps_enu)
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            logger.warning("Non-finite alignment transform, skipping GPS alignment")
            return pointcloud

        # Step 6: Optionally refine rotation using gravity
        if imu_data is not None:
            rotation = self._refine_with_gravity(rotation, imu_data)

        # Step 7: Build transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        # Step 8: Apply to point cloud (including scale)
        scaled_points = pointcloud.points * scale
        aligned_points = (rotation @ scaled_points.T).T + translation
        aligned_finite = np.isfinite(aligned_points).all(axis=1)
        if not np.all(aligned_finite):
            logger.warning(
                "Aligned pointcloud has %d non-finite points, skipping GPS alignment",
                int(np.count_nonzero(~aligned_finite)),
            )
            return pointcloud

        # Get origin GPS
        origin_gps = (
            gps_track.latitudes[0],
            gps_track.longitudes[0],
            gps_track.altitudes[0] if gps_track.altitudes is not None else 0.0,
        )

        return PointCloud(
            points=aligned_points.astype(np.float32),
            colors=pointcloud.colors,
            confidence=pointcloud.confidence,
            normals=pointcloud.normals,
            origin_gps=origin_gps,
            scale=scale,
            is_metric=True,
        )

    def compute_scale(
        self,
        poses: CameraPoses,
        gps_track: GPSTrack,
    ) -> float:
        """
        Compute scale factor from GPS trajectory.

        Scale = GPS_trajectory_length / Pose_trajectory_length

        Args:
            poses: Camera poses from reconstruction
            gps_track: GPS trajectory

        Returns:
            Scale factor to convert reconstruction units to meters
        """
        gps_length = gps_track.get_trajectory_length_meters()
        pose_length = poses.get_trajectory_length()

        if pose_length < 1e-6:
            logger.warning("Pose trajectory length near zero, using scale=1.0")
            return 1.0

        if gps_length < 1e-6:
            logger.warning("GPS trajectory length near zero, using scale=1.0")
            return 1.0

        if not np.isfinite(pose_length) or not np.isfinite(gps_length):
            logger.warning("Non-finite trajectory length, using scale=1.0")
            return 1.0

        scale = gps_length / pose_length

        # Sanity check - scale should be reasonable
        if scale > 1000 or scale < 0.001:
            logger.warning(
                f"Unusual scale factor {scale:.4f}, GPS/pose trajectories may not match"
            )

        return scale

    def _align_sample_counts(
        self,
        pose_positions: np.ndarray,
        gps_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align sample counts between pose and GPS trajectories.

        Uses linear interpolation to match the trajectory with more samples
        to the one with fewer samples.
        """
        n_poses = len(pose_positions)
        n_gps = len(gps_positions)

        if n_poses == n_gps:
            return pose_positions, gps_positions

        # Interpolate the longer one to match the shorter
        if n_poses > n_gps:
            # Subsample poses to match GPS
            indices = np.linspace(0, n_poses - 1, n_gps).astype(int)
            return pose_positions[indices], gps_positions
        else:
            # Subsample GPS to match poses
            indices = np.linspace(0, n_gps - 1, n_poses).astype(int)
            return pose_positions, gps_positions[indices]

    def _kabsch_align(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rotation and translation to align source to target using Kabsch algorithm.

        The Kabsch algorithm finds the optimal rotation matrix that minimizes
        the RMSD between two paired sets of points.

        Args:
            source: (N, 3) source points
            target: (N, 3) target points

        Returns:
            Tuple of (rotation_matrix, translation_vector)
        """
        # Center both point sets
        source_centroid = source.mean(axis=0)
        target_centroid = target.mean(axis=0)

        source_centered = source - source_centroid
        target_centered = target - target_centroid

        # Compute covariance matrix
        H = source_centered.T @ target_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Kabsch singular values: %s", S)

        # Compute rotation
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = target_centroid - R @ source_centroid

        return R, t

    def _refine_with_gravity(
        self,
        rotation: np.ndarray,
        imu_data: IMUData,
    ) -> np.ndarray:
        """
        Refine rotation using gravity vector from IMU.

        This ensures the "up" direction in the reconstruction matches
        the real-world up direction.
        """
        gravity = imu_data.get_gravity_direction()
        if gravity is None:
            return rotation

        # The gravity vector in world coordinates should be [0, 0, -1] (down)
        # We want to find a small rotation correction that aligns
        # the transformed gravity to [0, 0, -1]

        # Transform gravity by current rotation
        transformed_gravity = rotation @ gravity

        # Target gravity direction
        target_gravity = np.array([0, 0, -1])

        # Compute rotation to align
        v = np.cross(transformed_gravity, target_gravity)
        c = np.dot(transformed_gravity, target_gravity)

        if abs(c + 1) < 1e-6:
            # Nearly opposite - 180 degree rotation
            # Find any perpendicular axis
            if abs(transformed_gravity[0]) < 0.9:
                axis = np.cross(transformed_gravity, [1, 0, 0])
            else:
                axis = np.cross(transformed_gravity, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            correction = self._axis_angle_to_rotation(axis, np.pi)
        elif np.linalg.norm(v) < 1e-6:
            # Already aligned
            correction = np.eye(3)
        else:
            # General case - Rodrigues' rotation formula
            s = np.linalg.norm(v)
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ])
            correction = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

        return correction @ rotation

    def _axis_angle_to_rotation(
        self,
        axis: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Convert axis-angle to rotation matrix using Rodrigues' formula."""
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    def compute_alignment_error(
        self,
        poses: CameraPoses,
        gps_track: GPSTrack,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> dict:
        """
        Compute alignment error metrics.

        Returns:
            Dictionary with error metrics (rmse, max_error, mean_error)
        """
        pose_positions = poses.get_positions()
        gps_enu = gps_track.to_local_enu()

        # Align sample counts
        pose_positions, gps_enu = self._align_sample_counts(
            pose_positions, gps_enu
        )

        # Apply transformation
        transformed = (rotation @ (pose_positions * scale).T).T + translation

        # Compute errors
        errors = np.linalg.norm(transformed - gps_enu, axis=1)

        return {
            "rmse_meters": float(np.sqrt(np.mean(errors ** 2))),
            "max_error_meters": float(np.max(errors)),
            "mean_error_meters": float(np.mean(errors)),
            "median_error_meters": float(np.median(errors)),
        }
