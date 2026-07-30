"""Metrics calculation for reconstruction evaluation."""

from typing import Optional, Sequence
import logging
import numpy as np

from ..core.types import PointCloud, CameraPoses, GPSTrack

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Compute evaluation metrics for reconstruction quality.

    Metrics computed:
    - Scale accuracy: How close to metric scale
    - GPS alignment RMSE: Error between camera trajectory and GPS
    - Point density: Points per cubic meter
    - Trajectory length: Total path length
    """

    def compute_all(
        self,
        pointcloud: PointCloud,
        poses: Optional[CameraPoses],
        gps_track: Optional[GPSTrack],
    ) -> dict:
        """
        Compute all available metrics.

        Args:
            pointcloud: Reconstructed point cloud
            poses: Camera poses (optional)
            gps_track: GPS trajectory (optional)

        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            "point_count": len(pointcloud),
            "is_metric": pointcloud.is_metric,
            "scale_factor": pointcloud.scale,
        }

        # Point cloud metrics
        metrics.update(self._compute_pointcloud_metrics(pointcloud))

        # Trajectory metrics (requires poses)
        if poses is not None:
            metrics.update(self._compute_trajectory_metrics(poses))

        # GPS comparison metrics (requires both poses and GPS)
        if poses is not None and gps_track is not None:
            metrics.update(self._compute_gps_metrics(poses, gps_track))

        return metrics

    def compute_chunks(
        self,
        pointclouds: Sequence[PointCloud],
        poses: Optional[CameraPoses],
        gps_track: Optional[GPSTrack],
    ) -> dict:
        """Compute the canonical metrics without concatenating source units."""
        if not pointclouds:
            raise ValueError("at least one point cloud is required")

        point_count = sum(len(pointcloud) for pointcloud in pointclouds)
        nonempty = [pointcloud for pointcloud in pointclouds if len(pointcloud)]
        metrics = {
            "point_count": point_count,
            "is_metric": all(pointcloud.is_metric for pointcloud in pointclouds),
            "scale_factor": pointclouds[0].scale,
        }

        if nonempty:
            bounds_min = np.minimum.reduce(
                [pointcloud.points.min(axis=0) for pointcloud in nonempty]
            )
            bounds_max = np.maximum.reduce(
                [pointcloud.points.max(axis=0) for pointcloud in nonempty]
            )
            dimensions = bounds_max - bounds_min
            volume = float(np.prod(dimensions))
            metrics.update(
                {
                    "point_density": point_count / volume if volume > 0 else 0.0,
                    "bounding_box_volume_m3": volume,
                    "bounding_box_x_m": float(dimensions[0]),
                    "bounding_box_y_m": float(dimensions[1]),
                    "bounding_box_z_m": float(dimensions[2]),
                }
            )
            confidence = [
                np.asarray(pointcloud.confidence, dtype=np.float64)
                for pointcloud in nonempty
                if pointcloud.confidence is not None
            ]
            if len(confidence) == len(nonempty):
                confidence_count = sum(len(values) for values in confidence)
                confidence_sum = sum(float(values.sum()) for values in confidence)
                confidence_sum_squares = sum(
                    float(np.square(values).sum()) for values in confidence
                )
                confidence_mean = confidence_sum / confidence_count
                confidence_variance = max(
                    confidence_sum_squares / confidence_count
                    - confidence_mean * confidence_mean,
                    0.0,
                )
                metrics.update(
                    {
                        "confidence_mean": confidence_mean,
                        "confidence_std": float(np.sqrt(confidence_variance)),
                        "confidence_min": min(
                            float(values.min()) for values in confidence
                        ),
                        "confidence_max": max(
                            float(values.max()) for values in confidence
                        ),
                    }
                )
        else:
            metrics.update(
                {
                    "point_density": 0.0,
                    "bounding_box_volume_m3": 0.0,
                }
            )

        if poses is not None:
            metrics.update(self._compute_trajectory_metrics(poses))
        if poses is not None and gps_track is not None:
            metrics.update(self._compute_gps_metrics(poses, gps_track))
        return metrics

    def _compute_pointcloud_metrics(self, pointcloud: PointCloud) -> dict:
        """Compute metrics from point cloud."""
        if len(pointcloud) == 0:
            return {
                "point_density": 0.0,
                "bounding_box_volume_m3": 0.0,
            }

        # Bounding box
        mins = pointcloud.points.min(axis=0)
        maxs = pointcloud.points.max(axis=0)
        dimensions = maxs - mins
        volume = float(np.prod(dimensions))

        # Point density
        density = len(pointcloud) / volume if volume > 0 else 0.0

        # Confidence statistics (if available)
        confidence_stats = {}
        if pointcloud.confidence is not None:
            confidence_stats = {
                "confidence_mean": float(np.mean(pointcloud.confidence)),
                "confidence_std": float(np.std(pointcloud.confidence)),
                "confidence_min": float(np.min(pointcloud.confidence)),
                "confidence_max": float(np.max(pointcloud.confidence)),
            }

        return {
            "point_density": density,
            "bounding_box_volume_m3": volume,
            "bounding_box_x_m": float(dimensions[0]),
            "bounding_box_y_m": float(dimensions[1]),
            "bounding_box_z_m": float(dimensions[2]),
            **confidence_stats,
        }

    def _compute_trajectory_metrics(self, poses: CameraPoses) -> dict:
        """Compute metrics from camera trajectory."""
        if len(poses) < 2:
            return {"trajectory_length_m": 0.0}

        trajectory_length = poses.get_trajectory_length()

        return {
            "trajectory_length_m": trajectory_length,
            "num_poses": len(poses),
        }

    def _compute_gps_metrics(
        self,
        poses: CameraPoses,
        gps_track: GPSTrack,
    ) -> dict:
        """Compute metrics comparing poses to GPS ground truth."""
        if len(poses) < 2 or len(gps_track) < 2:
            return {}

        # Scale accuracy (before alignment)
        pose_length = poses.get_trajectory_length()
        gps_length = gps_track.get_trajectory_length_meters()

        scale_ratio = pose_length / gps_length if gps_length > 0 else 0.0
        gps_enu = gps_track.to_local_enu()
        gps_std = float(np.mean(np.std(gps_enu, axis=0))) if len(gps_enu) else 0.0
        gps_var = float(np.mean(np.var(gps_enu, axis=0))) if len(gps_enu) else 0.0

        metrics = {
            "gps_point_count": len(gps_track),
            "gps_trajectory_length_m": gps_length,
            "gps_std_dev_m": gps_std,
            "gps_variance_m2": gps_var,
            "scale_ratio_before_alignment": scale_ratio,
        }

        # GPS alignment error (after applying scale)
        try:
            metrics.update(self._compute_alignment_error(poses, gps_track))
        except Exception as e:
            logger.warning(f"Could not compute alignment error: {e}")

        return metrics

    def _compute_alignment_error(
        self,
        poses: CameraPoses,
        gps_track: GPSTrack,
    ) -> dict:
        """
        Compute alignment error between poses and GPS.

        This computes the error after applying optimal scale and alignment.
        """
        # Get positions
        pose_positions = poses.get_positions()
        gps_enu = gps_track.to_local_enu()

        # Align sample counts
        n_poses = len(pose_positions)
        n_gps = len(gps_enu)

        if n_poses != n_gps:
            if n_poses > n_gps:
                indices = np.linspace(0, n_poses - 1, n_gps).astype(int)
                pose_positions = pose_positions[indices]
            else:
                indices = np.linspace(0, n_gps - 1, n_poses).astype(int)
                gps_enu = gps_enu[indices]

        # Compute optimal scale
        pose_length = np.sum(np.linalg.norm(np.diff(pose_positions, axis=0), axis=1))
        gps_length = np.sum(np.linalg.norm(np.diff(gps_enu, axis=0), axis=1))
        scale = gps_length / pose_length if pose_length > 0 else 1.0

        # Apply scale
        scaled_poses = pose_positions * scale

        # Compute optimal rotation and translation (Kabsch)
        source_centroid = scaled_poses.mean(axis=0)
        target_centroid = gps_enu.mean(axis=0)

        source_centered = scaled_poses - source_centroid
        target_centered = gps_enu - target_centroid

        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = target_centroid - R @ source_centroid

        # Apply transformation
        aligned = (R @ scaled_poses.T).T + t

        # Compute errors
        errors = np.linalg.norm(aligned - gps_enu, axis=1)

        return {
            "gps_rmse_m": float(np.sqrt(np.mean(errors**2))),
            "gps_max_error_m": float(np.max(errors)),
            "gps_mean_error_m": float(np.mean(errors)),
            "gps_median_error_m": float(np.median(errors)),
        }

    def scale_accuracy(
        self,
        poses: CameraPoses,
        gps_track: GPSTrack,
    ) -> float:
        """
        Compute scale accuracy as ratio of trajectory lengths.

        Returns:
            Ratio (1.0 = perfect, <1 = underestimate, >1 = overestimate)
        """
        pose_length = poses.get_trajectory_length()
        gps_length = gps_track.get_trajectory_length_meters()

        if gps_length == 0:
            return float("nan")

        return pose_length / gps_length
