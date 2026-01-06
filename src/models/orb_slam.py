"""ORB-SLAM Python implementation for visual odometry."""

from pathlib import Path
from typing import Optional, List, Tuple
import logging
import numpy as np
import cv2

from .base import BaseModel
from ..core.types import (
    ReconstructionResult,
    PointCloud,
    CameraPoses,
    VideoInput,
)

logger = logging.getLogger(__name__)


class ORBSLAMModel(BaseModel):
    """
    ORB-SLAM reimplemented in Python.

    A simplified visual odometry implementation using ORB features.
    This is not the full ORB-SLAM3 system, but a basic implementation
    suitable for generating sparse point clouds and camera trajectories.

    Core components:
    1. ORB feature extraction (OpenCV)
    2. Feature matching between consecutive frames
    3. Essential matrix / PnP pose estimation
    4. Point triangulation
    5. Optional: Local bundle adjustment
    6. Optional: Loop closure detection

    Does NOT output metric scale without IMU or GPS.

    """

    name = "orb_slam"
    outputs_metric_scale = False  # Needs IMU/GPS for scale
    outputs_poses = True
    outputs_confidence = False
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # ORB parameters
        self.num_features = self.config.get("num_features", 2000)
        self.scale_factor = self.config.get("scale_factor", 1.2)
        self.num_levels = self.config.get("num_levels", 8)

        # Matching parameters
        self.match_ratio = self.config.get("match_ratio", 0.75)
        self.min_matches = self.config.get("min_matches", 50)

        # Use IMU for scale recovery if available
        self.use_imu = self.config.get("use_imu", False)

        # Camera intrinsics (will be estimated if not provided)
        self.fx = self.config.get("fx")
        self.fy = self.config.get("fy")
        self.cx = self.config.get("cx")
        self.cy = self.config.get("cy")

        self.orb = None
        self.matcher = None

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default ORB-SLAM configuration."""
        return {
            "num_features": 2000,
            "scale_factor": 1.2,
            "num_levels": 8,
            "match_ratio": 0.75,
            "min_matches": 50,
            "use_imu": False,
            # Camera intrinsics (None = estimate from image size)
            "fx": None,
            "fy": None,
            "cx": None,
            "cy": None,
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        """Initialize ORB detector and matcher."""
        self.orb = cv2.ORB_create(
            nfeatures=self.num_features,
            scaleFactor=self.scale_factor,
            nlevels=self.num_levels,
        )

        # Use BFMatcher with Hamming distance for ORB
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._is_loaded = True
        logger.info("ORB-SLAM initialized")

    def reconstruct(
        self,
        video_input: VideoInput,
        output_dir: Path,
    ) -> ReconstructionResult:
        """
        Run ORB-SLAM reconstruction.

        Pipeline:
        1. Extract ORB features from each frame
        2. Match features between consecutive frames
        3. Estimate relative pose using Essential matrix
        4. Triangulate 3D points
        5. Accumulate poses and points
        """
        self.ensure_loaded()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load images
        image_paths = video_input.get_frame_paths()
        logger.info(f"ORB-SLAM: Processing {len(image_paths)} frames")

        if len(image_paths) < 2:
            raise ValueError("Need at least 2 frames for reconstruction")

        # Initialize camera matrix
        first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            raise ValueError(f"Could not read first frame: {image_paths[0]}")
        K = self._get_camera_matrix(first_image.shape)

        # Process frames
        poses = [np.eye(4)]  # First frame at origin
        all_points = []
        all_colors = []

        prev_image = first_image
        prev_kp, prev_desc = self.orb.detectAndCompute(prev_image, None)

        for i, img_path in enumerate(image_paths[1:], 1):
            if i % 50 == 0:
                logger.info(f"  Processing frame {i}/{len(image_paths)}")

            # Load current frame
            curr_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if curr_image is None:
                logger.warning(f"  Frame {i}: Could not read image, skipping")
                poses.append(poses[-1].copy())
                continue
            curr_kp, curr_desc = self.orb.detectAndCompute(curr_image, None)

            if curr_desc is None or len(curr_kp) < self.min_matches:
                logger.warning(f"  Frame {i}: Not enough features, skipping")
                poses.append(poses[-1].copy())
                prev_image = curr_image
                prev_kp, prev_desc = curr_kp, curr_desc
                continue

            # Match features
            matches = self._match_features(prev_desc, curr_desc)

            if len(matches) < self.min_matches:
                logger.warning(f"  Frame {i}: Not enough matches ({len(matches)})")
                poses.append(poses[-1].copy())
                prev_image = curr_image
                prev_kp, prev_desc = curr_kp, curr_desc
                continue

            # Get matched points
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in matches])
            pts2 = np.float32([curr_kp[m.trainIdx].pt for m in matches])

            # Estimate Essential matrix
            E, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )

            if E is None:
                logger.warning(f"  Frame {i}: Could not estimate Essential matrix")
                poses.append(poses[-1].copy())
                prev_image = curr_image
                prev_kp, prev_desc = curr_kp, curr_desc
                continue

            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
            if not np.isfinite(R).all() or not np.isfinite(t).all():
                logger.warning(f"  Frame {i}: Non-finite pose recovered, skipping")
                poses.append(poses[-1].copy())
                prev_image = curr_image
                prev_kp, prev_desc = curr_kp, curr_desc
                continue

            inlier_count = int(np.count_nonzero(mask))
            logger.debug(
                "  Frame %d: keypoints=%d/%d matches=%d inliers=%d",
                i,
                len(prev_kp),
                len(curr_kp),
                len(matches),
                inlier_count,
            )

            # Build transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            # Accumulate pose
            current_pose = poses[-1] @ T
            poses.append(current_pose)

            # Triangulate points
            inlier_pts1 = pts1[mask.ravel() == 255]
            inlier_pts2 = pts2[mask.ravel() == 255]

            if len(inlier_pts1) >= 8:
                points_3d, valid_indices = self._triangulate_points(
                    inlier_pts1, inlier_pts2, poses[-2], current_pose, K
                )

                if len(points_3d) > 0:
                    all_points.extend(points_3d)

                    # Get colors from image
                    color_image = cv2.imread(str(img_path))
                    colors = self._get_point_colors(
                        inlier_pts2[valid_indices], color_image
                    )
                    all_colors.extend(colors)

            prev_image = curr_image
            prev_kp, prev_desc = curr_kp, curr_desc

        # Convert to arrays
        poses_array = np.array(poses)

        if len(all_points) > 0:
            points_array = np.array(all_points, dtype=np.float32)
            colors_array = np.array(all_colors, dtype=np.uint8) if all_colors else None
            finite_mask = np.isfinite(points_array).all(axis=1)
            non_finite = int(np.count_nonzero(~finite_mask))
            if non_finite:
                logger.warning(
                    "ORB-SLAM: Dropping %d non-finite points before saving",
                    non_finite,
                )
                points_array = points_array[finite_mask]
                if colors_array is not None:
                    colors_array = colors_array[finite_mask]
        else:
            points_array = np.zeros((0, 3), dtype=np.float32)
            colors_array = None

        if not np.isfinite(poses_array).all():
            logger.warning("ORB-SLAM: Non-finite pose values detected")

        logger.info(
            f"ORB-SLAM: Generated {len(points_array)} points, {len(poses_array)} poses"
        )

        # Optional: Use IMU for scale recovery
        scale = 1.0
        is_metric = False
        if self.use_imu and video_input.imu_data is not None:
            scale = self._estimate_scale_from_imu(poses_array, video_input.imu_data)
            points_array *= scale
            is_metric = True
            logger.info(f"ORB-SLAM: Applied IMU scale factor: {scale:.4f}")

        # Get timestamps
        timestamps = video_input.get_frame_timestamps()
        if len(timestamps) > len(poses_array):
            timestamps = timestamps[: len(poses_array)]

        return ReconstructionResult(
            pointcloud=PointCloud(
                points=points_array,
                colors=colors_array,
                is_metric=is_metric,
                scale=scale,
            ),
            poses=CameraPoses(
                poses=poses_array,
                timestamps=timestamps,
                intrinsics=K,
            ),
            metadata={
                "model": "orb_slam",
                "num_features": self.num_features,
                "total_matches": len(all_points),
                "frames_processed": len(poses_array),
            },
        )

    def _get_camera_matrix(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Get or estimate camera intrinsics matrix."""
        h, w = image_shape[:2]

        # Use provided intrinsics or estimate
        fx = self.fx or w  # Approximate focal length
        fy = self.fy or w
        cx = self.cx or w / 2
        cy = self.cy or h / 2

        return np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def _match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List:
        """Match features using ratio test."""
        if desc1 is None or desc2 is None:
            return []

        # KNN match
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)

        return good_matches

    def _triangulate_points(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        pose1: np.ndarray,
        pose2: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Triangulate 3D points from matched 2D points."""
        # Build projection matrices
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert to 3D
        w = points_4d[3, :]
        valid_w = np.abs(w) > 1e-8
        if not np.all(valid_w):
            logger.debug(
                "Triangulation: Dropped %d points with near-zero w",
                int(np.count_nonzero(~valid_w)),
            )
        points_3d = (points_4d[:3, valid_w] / w[valid_w]).T
        valid_indices = np.flatnonzero(valid_w)
        finite_mask = np.isfinite(points_3d).all(axis=1)
        if not np.all(finite_mask):
            logger.debug(
                "Triangulation: Dropped %d non-finite points",
                int(np.count_nonzero(~finite_mask)),
            )
        points_3d = points_3d[finite_mask]
        valid_indices = valid_indices[finite_mask]

        # Filter points behind camera or too far
        valid = []
        kept_indices = []
        for i, pt in enumerate(points_3d):
            # Check if in front of both cameras
            pt_cam1 = pose1[:3, :3] @ pt + pose1[:3, 3]
            pt_cam2 = pose2[:3, :3] @ pt + pose2[:3, 3]

            if pt_cam1[2] > 0 and pt_cam2[2] > 0:
                # Check reasonable distance
                dist = np.linalg.norm(pt)
                if 0.1 < dist < 1000:
                    valid.append(pt)
                    kept_indices.append(valid_indices[i])

        return valid, np.array(kept_indices, dtype=int)

    def _get_point_colors(
        self,
        pts2d: np.ndarray,
        color_image: np.ndarray,
    ) -> List[np.ndarray]:
        """Get colors for points from image."""
        colors = []
        h, w = color_image.shape[:2]

        for pt in pts2d:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                # BGR to RGB
                bgr = color_image[y, x]
                colors.append([bgr[2], bgr[1], bgr[0]])
            else:
                colors.append([128, 128, 128])

        return colors

    def _estimate_scale_from_imu(
        self,
        poses: np.ndarray,
        imu_data,
    ) -> float:
        """
        Estimate scale using IMU integration.

        This is a simplified approach - integrates accelerometer
        to estimate displacement and compares to visual odometry.
        """
        # TODO: Implement proper IMU integration
        # This would require:
        # 1. Integrate accelerometer (double integration with gravity removal)
        # 2. Compare integrated displacement to visual odometry
        # 3. Compute optimal scale factor

        logger.warning("IMU scale estimation not fully implemented")
        return 1.0
