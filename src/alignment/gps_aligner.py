"""Timestamped, quality-aware GPS alignment.

The durable transform maps model coordinates into an artifact-local ENU frame.
Its float64 ENU-to-ECEF placement is kept separately so large global
coordinates never need to be baked into point buffers.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..core.types import (
    AlignmentResult,
    AlignmentStatus,
    CameraPoses,
    GPSTrack,
    IMUData,
    PointCloud,
)

logger = logging.getLogger(__name__)

MIN_GPS_TRAJECTORY_LENGTH_M = 2.0
MIN_GPS_STD_DEV_M = 0.5


class GPSAligner:
    """Estimate a robust model-to-ENU similarity transform."""

    def __init__(
        self,
        *,
        min_correspondences: int = 6,
        min_gps_trajectory_length_m: float = MIN_GPS_TRAJECTORY_LENGTH_M,
        min_gps_std_dev_m: float = MIN_GPS_STD_DEV_M,
        max_rmse_m: float = 8.0,
        max_scale: float = 1000.0,
        min_scale: float = 0.001,
        min_inlier_fraction: float = 0.5,
        max_clock_offset_s: float = 10.0,
        clock_step_s: float = 0.1,
        min_clock_peak_quality: float = 0.25,
        max_gps_interpolation_gap_s: float = 2.0,
    ) -> None:
        self.min_correspondences = int(min_correspondences)
        self.min_gps_trajectory_length_m = float(min_gps_trajectory_length_m)
        self.min_gps_std_dev_m = float(min_gps_std_dev_m)
        self.max_rmse_m = float(max_rmse_m)
        self.max_scale = float(max_scale)
        self.min_scale = float(min_scale)
        self.min_inlier_fraction = float(min_inlier_fraction)
        self.max_clock_offset_s = float(max_clock_offset_s)
        self.clock_step_s = float(clock_step_s)
        self.min_clock_peak_quality = float(min_clock_peak_quality)
        self.max_gps_interpolation_gap_s = float(max_gps_interpolation_gap_s)

    def align(
        self,
        pointcloud: PointCloud,
        poses: CameraPoses,
        gps_track: GPSTrack,
        imu_data: Optional[IMUData] = None,
        allow_scale: bool = True,
        *,
        gravity_direction_model: Optional[np.ndarray] = None,
    ) -> AlignmentResult:
        """Align a reconstruction and return the complete attempt result.

        GPS is interpolated to pose timestamps after estimating a constant
        telemetry clock offset from trajectory speed.  ``imu_data`` is accepted
        for API compatibility, but gravity is only constrained when the caller
        supplies ``gravity_direction_model``.  Raw GoPro gravity is in a sensor
        frame whose camera/model extrinsic is not yet part of Mapper's contract;
        silently treating it as model-frame gravity would be unsafe.
        """

        if len(poses) < self.min_correspondences:
            return AlignmentResult.unaligned(
                "insufficient_poses",
                diagnostics={"pose_count": len(poses)},
            )
        if poses.timestamps is None:
            return AlignmentResult.unaligned("pose_timestamps_missing")
        if len(gps_track) < self.min_correspondences:
            return AlignmentResult.unaligned(
                "insufficient_gps_samples",
                diagnostics={"gps_count": len(gps_track)},
            )
        if gps_track.timestamps is None:
            return AlignmentResult.unaligned("gps_timestamps_missing")
        if not np.isfinite(pointcloud.points).all():
            return AlignmentResult.unaligned("pointcloud_contains_nonfinite_values")

        filtered_gps = gps_track.filter_quality()
        if len(filtered_gps) < self.min_correspondences:
            return AlignmentResult.unaligned(
                "insufficient_quality_gps_samples",
                diagnostics={
                    "gps_count": len(gps_track),
                    "quality_gps_count": len(filtered_gps),
                },
            )

        anchor_wgs84, anchor_ecef = filtered_gps.robust_anchor()
        gps_enu = filtered_gps.to_local_enu(anchor_wgs84)
        gps_length = _trajectory_length(gps_enu)
        gps_horizontal_std = float(np.sqrt(np.mean(np.var(gps_enu[:, :2], axis=0))))
        if (
            gps_length < self.min_gps_trajectory_length_m
            or gps_horizontal_std < self.min_gps_std_dev_m
        ):
            return AlignmentResult.unaligned(
                "gps_track_lacks_motion",
                diagnostics={
                    "gps_trajectory_length_m": gps_length,
                    "gps_horizontal_std_m": gps_horizontal_std,
                    "quality_gps_count": len(filtered_gps),
                },
            )

        pose_positions = np.asarray(poses.get_positions(), dtype=np.float64)
        pose_timestamps = np.asarray(poses.timestamps, dtype=np.float64)
        pose_valid = np.isfinite(pose_timestamps) & np.isfinite(pose_positions).all(
            axis=1
        )
        pose_positions = pose_positions[pose_valid]
        pose_timestamps = pose_timestamps[pose_valid]
        if len(pose_positions) < self.min_correspondences:
            return AlignmentResult.unaligned("insufficient_finite_poses")

        pose_timestamps, pose_positions = _sort_and_deduplicate(
            pose_timestamps, pose_positions
        )
        gps_timestamps, gps_enu, gps_order = _sort_and_deduplicate_with_indices(
            np.asarray(filtered_gps.timestamps, dtype=np.float64), gps_enu
        )
        if len(gps_timestamps) < self.min_correspondences:
            return AlignmentResult.unaligned("insufficient_unique_gps_timestamps")

        clock_offset_s, clock_quality = self.estimate_clock_offset(
            pose_positions,
            pose_timestamps,
            gps_enu,
            gps_timestamps,
        )
        paired = self._pair_by_timestamp(
            pose_positions,
            pose_timestamps,
            gps_enu,
            gps_timestamps,
            filtered_gps,
            gps_order,
            clock_offset_s,
        )
        source, target, weights = paired
        if len(source) < self.min_correspondences:
            return AlignmentResult.unaligned(
                "insufficient_timestamp_overlap",
                correspondence_count=len(source),
                diagnostics={
                    "clock_offset_s": clock_offset_s,
                    "clock_peak_quality": clock_quality,
                },
            )

        scale, rotation, translation, inliers, residuals = (
            self._robust_weighted_umeyama(
                source,
                target,
                weights,
                allow_scale=allow_scale,
                gravity_direction_model=gravity_direction_model,
            )
        )
        if not (
            np.isfinite(scale)
            and np.isfinite(rotation).all()
            and np.isfinite(translation).all()
        ):
            return AlignmentResult.unaligned(
                "nonfinite_alignment_solution",
                correspondence_count=len(source),
            )
        if allow_scale and not (self.min_scale <= scale <= self.max_scale):
            return AlignmentResult.unaligned(
                "scale_out_of_range",
                correspondence_count=len(source),
                diagnostics={"estimated_scale": float(scale)},
            )

        inlier_count = int(np.count_nonzero(inliers))
        inlier_fraction = inlier_count / len(source)
        rmse = (
            float(
                np.sqrt(np.average(residuals[inliers] ** 2, weights=weights[inliers]))
            )
            if inlier_count
            else float("inf")
        )
        transformed_inliers = (
            (scale * rotation @ source[inliers].T).T + translation
            if inlier_count
            else np.zeros((0, 3), dtype=np.float64)
        )
        residual_vectors = transformed_inliers - target[inliers]
        horizontal_rmse = (
            float(
                np.sqrt(
                    np.average(
                        np.sum(residual_vectors[:, :2] ** 2, axis=1),
                        weights=weights[inliers],
                    )
                )
            )
            if inlier_count
            else float("inf")
        )
        vertical_rmse = (
            float(
                np.sqrt(
                    np.average(
                        residual_vectors[:, 2] ** 2,
                        weights=weights[inliers],
                    )
                )
            )
            if inlier_count
            else float("inf")
        )
        diagnostics = {
            "gps_trajectory_length_m": gps_length,
            "gps_horizontal_std_m": gps_horizontal_std,
            "quality_gps_count": len(filtered_gps),
            "inlier_fraction": inlier_fraction,
            "median_residual_m": float(np.median(residuals)),
            "max_residual_m": float(np.max(residuals)),
            "arc_length_scale_diagnostic": self.compute_scale(poses, filtered_gps),
            "gravity_constrained": gravity_direction_model is not None,
            "imu_gravity_ignored_without_model_frame_extrinsic": (
                imu_data is not None and gravity_direction_model is None
            ),
        }
        if (
            inlier_count < self.min_correspondences
            or inlier_fraction < self.min_inlier_fraction
        ):
            return AlignmentResult.unaligned(
                "insufficient_alignment_inliers",
                correspondence_count=len(source),
                diagnostics={
                    **diagnostics,
                    "estimated_rmse_m": rmse,
                    "clock_offset_s": clock_offset_s,
                    "clock_peak_quality": clock_quality,
                },
            )
        if not np.isfinite(rmse) or rmse > self.max_rmse_m:
            return AlignmentResult.unaligned(
                "alignment_rmse_exceeds_threshold",
                correspondence_count=len(source),
                diagnostics={
                    **diagnostics,
                    "estimated_rmse_m": rmse,
                    "max_rmse_m": self.max_rmse_m,
                    "clock_offset_s": clock_offset_s,
                    "clock_peak_quality": clock_quality,
                },
            )

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = scale * rotation
        transform[:3, 3] = translation
        enu_to_ecef = _enu_to_ecef_transform(anchor_wgs84, anchor_ecef)

        aligned_points = (
            scale * rotation @ np.asarray(pointcloud.points, dtype=np.float64).T
        ).T + translation
        aligned_normals = None
        if pointcloud.normals is not None:
            aligned_normals = (
                rotation @ np.asarray(pointcloud.normals, dtype=np.float64).T
            ).T.astype(np.float32)
        aligned_cloud = PointCloud(
            points=aligned_points.astype(np.float32),
            colors=pointcloud.colors,
            confidence=pointcloud.confidence,
            normals=aligned_normals,
            origin_gps=anchor_wgs84,
            scale=float(pointcloud.scale * scale),
            is_metric=True,
        )
        status = (
            AlignmentStatus.ALIGNED
            if rmse <= min(2.0, self.max_rmse_m) and inlier_fraction >= 0.7
            else AlignmentStatus.APPROXIMATE
        )
        method = (
            "gps_gravity_constrained_weighted_umeyama"
            if gravity_direction_model is not None
            else "gps_robust_weighted_umeyama"
        )
        return AlignmentResult(
            transform=transform,
            enu_to_ecef_transform=enu_to_ecef,
            scale=float(scale),
            method=method,
            status=status,
            inlier_count=inlier_count,
            correspondence_count=len(source),
            rmse_m=rmse,
            horizontal_rmse_m=horizontal_rmse,
            vertical_rmse_m=vertical_rmse,
            anchor_wgs84=anchor_wgs84,
            anchor_ecef=anchor_ecef,
            aligned_pointcloud=aligned_cloud,
            clock_offset_s=clock_offset_s,
            clock_peak_quality=clock_quality,
            diagnostics=diagnostics,
        )

    def estimate_clock_offset(
        self,
        pose_positions: np.ndarray,
        pose_timestamps: np.ndarray,
        gps_positions: np.ndarray,
        gps_timestamps: np.ndarray,
    ) -> Tuple[float, float]:
        """Estimate ``gps_timestamp = pose_timestamp + offset`` using speed."""

        pose_speed_t, pose_speed = _trajectory_speeds(pose_positions, pose_timestamps)
        gps_speed_t, gps_speed = _trajectory_speeds(gps_positions, gps_timestamps)
        if len(pose_speed) < 3 or len(gps_speed) < 3:
            return 0.0, 0.0
        pose_scale = _robust_spread(pose_speed)
        gps_scale = _robust_spread(gps_speed)
        if pose_scale < 1e-9 or gps_scale < 1e-9:
            return 0.0, 0.0
        # Position spikes create two extreme speed samples.  Winsorizing at a
        # robust two-sigma envelope prevents those samples from dominating the
        # clock correlation before the spatial robust fit gets a chance to
        # reject them.
        normalized_pose = np.clip(
            (pose_speed - np.median(pose_speed)) / pose_scale, -2.0, 2.0
        )
        normalized_gps = np.clip(
            (gps_speed - np.median(gps_speed)) / gps_scale, -2.0, 2.0
        )

        candidates = np.arange(
            -self.max_clock_offset_s,
            self.max_clock_offset_s + self.clock_step_s * 0.5,
            self.clock_step_s,
        )
        best_offset = 0.0
        best_correlation = -np.inf
        for offset in candidates:
            query_t = pose_speed_t + offset
            valid = (query_t >= gps_speed_t[0]) & (query_t <= gps_speed_t[-1])
            if np.count_nonzero(valid) < max(3, self.min_correspondences - 1):
                continue
            sampled_gps = np.interp(query_t[valid], gps_speed_t, normalized_gps)
            sampled_pose = normalized_pose[valid]
            if np.std(sampled_gps) < 1e-9 or np.std(sampled_pose) < 1e-9:
                continue
            correlation = float(np.corrcoef(sampled_pose, sampled_gps)[0, 1])
            # Tiny tie breaker prefers the least invasive offset.
            score = correlation - 1e-9 * abs(float(offset))
            if score > best_correlation:
                best_correlation = score
                best_offset = float(offset)
        if not np.isfinite(best_correlation):
            return 0.0, 0.0
        peak_quality = float(np.clip(best_correlation, 0.0, 1.0))
        if peak_quality < self.min_clock_peak_quality:
            # A low-information speed trace (or unrelated sensor clocks)
            # should not induce an arbitrary edge-of-search offset.
            return 0.0, peak_quality
        return best_offset, peak_quality

    def _pair_by_timestamp(
        self,
        pose_positions: np.ndarray,
        pose_timestamps: np.ndarray,
        gps_positions: np.ndarray,
        gps_timestamps: np.ndarray,
        gps_track: GPSTrack,
        gps_order: np.ndarray,
        clock_offset_s: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_t = pose_timestamps + clock_offset_s
        valid = (query_t >= gps_timestamps[0]) & (query_t <= gps_timestamps[-1])
        right = np.searchsorted(gps_timestamps, query_t, side="left")
        right = np.clip(right, 1, len(gps_timestamps) - 1)
        left = right - 1
        interpolation_gap = gps_timestamps[right] - gps_timestamps[left]
        valid &= interpolation_gap <= self.max_gps_interpolation_gap_s
        source = pose_positions[valid]
        query_t = query_t[valid]
        target = np.column_stack(
            [
                np.interp(query_t, gps_timestamps, gps_positions[:, axis])
                for axis in range(3)
            ]
        )
        weights = np.ones(len(source), dtype=np.float64)
        if gps_track.accuracies is not None:
            accuracy = np.asarray(gps_track.accuracies, dtype=float)[gps_order]
            sampled = np.interp(query_t, gps_timestamps, accuracy)
            weights *= 1.0 / np.square(np.maximum(sampled, 0.25))
        if gps_track.position_dops is not None:
            dop = np.asarray(gps_track.position_dops, dtype=float)[gps_order]
            sampled = np.interp(query_t, gps_timestamps, dop)
            weights *= 1.0 / np.square(np.maximum(sampled, 1.0))
        weights /= max(float(np.mean(weights)), 1e-12)
        return source, target, weights

    def _robust_weighted_umeyama(
        self,
        source: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        *,
        allow_scale: bool,
        gravity_direction_model: Optional[np.ndarray],
        max_iterations: int = 20,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base_weights = np.asarray(weights, dtype=np.float64)
        robust_weights = base_weights.copy()
        previous = np.full(len(source), np.inf)
        for _ in range(max_iterations):
            scale, rotation, translation = _weighted_similarity(
                source,
                target,
                robust_weights,
                allow_scale=allow_scale,
                gravity_direction_model=gravity_direction_model,
            )
            residuals = np.linalg.norm(
                (scale * rotation @ source.T).T + translation - target,
                axis=1,
            )
            median = np.median(residuals)
            mad = np.median(np.abs(residuals - median))
            sigma = max(1.4826 * mad, 0.25)
            cutoff = max(1.5 * sigma, 0.5)
            huber = np.minimum(1.0, cutoff / np.maximum(residuals, 1e-12))
            robust_weights = base_weights * huber
            if np.max(np.abs(previous - residuals)) < 1e-6:
                break
            previous = residuals

        threshold = max(
            3.0
            * max(1.4826 * np.median(np.abs(residuals - np.median(residuals))), 0.25),
            1.0,
        )
        inliers = residuals <= threshold
        if np.count_nonzero(inliers) >= self.min_correspondences:
            scale, rotation, translation = _weighted_similarity(
                source[inliers],
                target[inliers],
                base_weights[inliers],
                allow_scale=allow_scale,
                gravity_direction_model=gravity_direction_model,
            )
            residuals = np.linalg.norm(
                (scale * rotation @ source.T).T + translation - target,
                axis=1,
            )
        return scale, rotation, translation, inliers, residuals

    def compute_scale(self, poses: CameraPoses, gps_track: GPSTrack) -> float:
        """Return the noisy arc-length ratio as a diagnostic only."""

        pose_length = poses.get_trajectory_length()
        gps_length = gps_track.get_trajectory_length_meters()
        if (
            pose_length < 1e-9
            or gps_length < 1e-9
            or not np.isfinite(pose_length)
            or not np.isfinite(gps_length)
        ):
            return 1.0
        return float(gps_length / pose_length)

    def compute_alignment_error(
        self,
        poses: CameraPoses,
        gps_track: GPSTrack,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
        *,
        clock_offset_s: float = 0.0,
    ) -> dict:
        """Compute timestamp-paired residual summaries for a known transform."""

        if poses.timestamps is None or gps_track.timestamps is None:
            raise ValueError("Pose and GPS timestamps are required")
        anchor, _ = gps_track.robust_anchor()
        gps_enu = gps_track.to_local_enu(anchor)
        order = np.argsort(gps_track.timestamps, kind="stable")
        gps_t = np.asarray(gps_track.timestamps)[order]
        gps_enu = gps_enu[order]
        query_t = np.asarray(poses.timestamps) + clock_offset_s
        valid = (query_t >= gps_t[0]) & (query_t <= gps_t[-1])
        source = poses.get_positions()[valid]
        query_t = query_t[valid]
        target = np.column_stack(
            [np.interp(query_t, gps_t, gps_enu[:, axis]) for axis in range(3)]
        )
        transformed = (scale * rotation @ source.T).T + translation
        errors = np.linalg.norm(transformed - target, axis=1)
        return {
            "rmse_meters": float(np.sqrt(np.mean(errors**2))),
            "max_error_meters": float(np.max(errors)),
            "mean_error_meters": float(np.mean(errors)),
            "median_error_meters": float(np.median(errors)),
        }


def _weighted_similarity(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    allow_scale: bool,
    gravity_direction_model: Optional[np.ndarray],
) -> Tuple[float, np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    source_mean = np.sum(weights[:, None] * source, axis=0)
    target_mean = np.sum(weights[:, None] * target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    if gravity_direction_model is None:
        covariance = (weights[:, None] * target_centered).T @ source_centered
        u, singular_values, vt = np.linalg.svd(covariance)
        sign = np.ones(3)
        if np.linalg.det(u @ vt) < 0:
            sign[-1] = -1.0
        rotation = u @ np.diag(sign) @ vt
        numerator = float(np.dot(singular_values, sign))
    else:
        gravity = np.asarray(gravity_direction_model, dtype=np.float64)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("gravity_direction_model must be one finite 3-vector")
        norm = np.linalg.norm(gravity)
        if norm < 1e-9:
            raise ValueError("gravity_direction_model cannot be zero")
        level_rotation = _rotation_between(gravity / norm, np.array([0.0, 0.0, -1.0]))
        leveled = (level_rotation @ source_centered.T).T
        cross = np.sum(
            weights
            * (
                leveled[:, 0] * target_centered[:, 1]
                - leveled[:, 1] * target_centered[:, 0]
            )
        )
        dot = np.sum(
            weights
            * (
                leveled[:, 0] * target_centered[:, 0]
                + leveled[:, 1] * target_centered[:, 1]
            )
        )
        yaw = np.arctan2(cross, dot)
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotation = yaw_rotation @ level_rotation
        rotated = (rotation @ source_centered.T).T
        numerator = float(np.sum(weights * np.sum(rotated * target_centered, axis=1)))

    source_variance = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    scale = numerator / source_variance if allow_scale and source_variance > 0 else 1.0
    translation = target_mean - scale * rotation @ source_mean
    return float(scale), rotation, translation


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-12:
        if dot > 0:
            return np.eye(3)
        basis = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.9:
            basis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)
    axis = cross / cross_norm
    return _axis_angle(axis, np.arctan2(cross_norm, dot))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _enu_to_ecef_transform(
    anchor_wgs84: Tuple[float, float, float],
    anchor_ecef: np.ndarray,
) -> np.ndarray:
    lat = np.deg2rad(anchor_wgs84[0])
    lon = np.deg2rad(anchor_wgs84[1])
    ecef_to_enu = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = ecef_to_enu.T
    transform[:3, 3] = anchor_ecef
    return transform


def _sort_and_deduplicate(
    timestamps: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    values = values[order]
    unique, indices = np.unique(timestamps, return_index=True)
    return unique, values[indices]


def _sort_and_deduplicate_with_indices(
    timestamps: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    values = values[order]
    unique, indices = np.unique(timestamps, return_index=True)
    return unique, values[indices], order[indices]


def _trajectory_speeds(
    positions: np.ndarray, timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    delta_t = np.diff(timestamps)
    valid = np.isfinite(delta_t) & (delta_t > 1e-9)
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1)[valid] / delta_t[valid]
    midpoint = ((timestamps[:-1] + timestamps[1:]) * 0.5)[valid]
    finite = np.isfinite(speed) & np.isfinite(midpoint)
    return midpoint[finite], speed[finite]


def _robust_spread(values: np.ndarray) -> float:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(max(1.4826 * mad, np.std(values) * 0.1))


def _trajectory_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
