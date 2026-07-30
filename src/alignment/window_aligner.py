"""Chunk alignment and merging utilities."""

from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from ..core.types import CameraPoses, PointCloud, ReconstructionResult

logger = logging.getLogger(__name__)


class WindowAligner:
    """Align and merge windowed reconstructions."""

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.method = config.get("window_alignment_method", "pose_overlap")
        self.min_overlap_frames = int(config.get("min_overlap_frames", 10))
        self.allow_scale = config.get("allow_scale", "auto")
        self.pose_overlap_max_rmse = float(config.get("pose_overlap_max_rmse", 2.0))
        self.icp_voxel_size = float(config.get("icp_voxel_size", 0.1))
        self.icp_max_corr_dist = float(config.get("icp_max_corr_dist", 0.2))
        self.icp_init = config.get("icp_init", "none")
        self.save_alignment_debug = bool(config.get("save_alignment_debug", False))

    def align_and_merge(
        self,
        chunks: List[ReconstructionResult],
        is_metric: bool,
    ) -> Tuple[PointCloud, Optional[CameraPoses], dict]:
        """Align windowed chunks and merge outputs.

        This compatibility method materializes a merged cloud. Canonical
        publication should call :meth:`align_chunks` and stage each returned
        source unit independently.
        """
        transformed_chunks, merged_poses, alignment_metadata = self.align_chunks(
            chunks, is_metric
        )
        merged_cloud = self._merge_pointclouds(
            [chunk.pointcloud for chunk in transformed_chunks]
        )
        return merged_cloud, merged_poses, alignment_metadata

    def align_chunks(
        self,
        chunks: List[ReconstructionResult],
        is_metric: bool,
    ) -> Tuple[List[ReconstructionResult], Optional[CameraPoses], dict]:
        """Align chunks without concatenating their point arrays.

        Keeping source units separate allows the publisher to transform,
        consolidate, and stage them with bounded memory. It also preserves the
        natural provenance boundary for ``PointSourceId``. The input list is
        replaced one element at a time so model-native point buffers can be
        released instead of retaining a second full set of windows.
        """
        if not chunks:
            raise ValueError("No chunks provided for alignment")

        allow_scale = self._resolve_allow_scale(is_metric)
        transforms = [(1.0, np.eye(3), np.zeros(3, dtype=np.float32))]
        alignment_log: List[dict] = [
            {
                "window_id": 0,
                "method": "identity",
                "scale": 1.0,
                "rotation": np.eye(3, dtype=np.float64).tolist(),
                "translation": np.zeros(3, dtype=np.float64).tolist(),
            }
        ]

        for idx in range(1, len(chunks)):
            prev = chunks[idx - 1]
            curr = chunks[idx]
            prev_scale, prev_R, prev_t = transforms[idx - 1]

            transform, log_entry = self._align_pair(
                prev,
                curr,
                prev_scale,
                prev_R,
                prev_t,
                allow_scale,
            )
            transforms.append(transform)
            alignment_log.append(log_entry)

        transformed_chunks = []
        transformed_poses = []
        poses_available = all(chunk.poses is not None for chunk in chunks)
        for index, (chunk, (scale, rotation, translation)) in enumerate(
            zip(chunks, transforms, strict=True)
        ):
            transformed_pointcloud = self._transform_pointcloud(
                chunk.pointcloud, scale, rotation, translation
            )
            transformed_pose = None
            if poses_available and chunk.poses is not None:
                transformed_pose = self._transform_poses(
                    chunk.poses, scale, rotation, translation
                )
                transformed_poses.append(transformed_pose)
            transformed_result = ReconstructionResult(
                pointcloud=transformed_pointcloud,
                poses=transformed_pose,
                metadata=dict(chunk.metadata),
                window_metadata=dict(chunk.window_metadata),
            )
            chunks[index] = transformed_result
            transformed_chunks.append(transformed_result)

        merged_poses = self._merge_poses(transformed_poses) if poses_available else None

        alignment_metadata = {
            "method": self.method,
            "allow_scale": allow_scale,
            "chunks": alignment_log,
        }

        return transformed_chunks, merged_poses, alignment_metadata

    def _resolve_allow_scale(self, is_metric: bool) -> bool:
        if isinstance(self.allow_scale, str) and self.allow_scale == "auto":
            return not is_metric
        return bool(self.allow_scale)

    def _align_pair(
        self,
        prev: ReconstructionResult,
        curr: ReconstructionResult,
        prev_scale: float,
        prev_R: np.ndarray,
        prev_t: np.ndarray,
        allow_scale: bool,
    ) -> Tuple[Tuple[float, np.ndarray, np.ndarray], dict]:
        """Align current chunk to previous chunk's global frame."""
        method_used = self.method
        log_entry = {"window_id": curr.window_metadata.get("window_id", None)}

        can_pose_overlap = (
            self.method == "pose_overlap"
            and prev.poses is not None
            and curr.poses is not None
        )

        if can_pose_overlap:
            overlap = self._get_overlap_positions(prev.poses, curr.poses)
            if overlap is not None:
                prev_positions, curr_positions, overlap_count = overlap
                prev_global = self._apply_sim3(
                    prev_positions, prev_scale, prev_R, prev_t
                )
                scale, rotation, translation, rmse = self._align_sim3(
                    curr_positions,
                    prev_global,
                    allow_scale,
                )
                if np.isfinite(rmse) and rmse <= self.pose_overlap_max_rmse:
                    log_entry.update(
                        {
                            "method": "pose_overlap",
                            "overlap_count": overlap_count,
                            "rmse": rmse,
                            "scale": scale,
                        }
                    )
                    self._maybe_add_transform_debug(
                        log_entry, scale, rotation, translation
                    )
                    return (scale, rotation, translation), log_entry

                logger.warning(
                    "Pose-overlap alignment rejected (rmse=%.3f, count=%d)",
                    rmse,
                    overlap_count,
                )

            method_used = "icp" if self.method == "pose_overlap" else self.method
        elif self.method == "pose_overlap":
            method_used = "icp"

        if method_used == "icp":
            prev_global = self._transform_pointcloud(
                prev.pointcloud, prev_scale, prev_R, prev_t
            )
            icp_result = self._align_icp(curr.pointcloud, prev_global)
            if icp_result is not None:
                transform, rmse = icp_result
                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                log_entry.update(
                    {
                        "method": "icp",
                        "rmse": rmse,
                        "scale": 1.0,
                    }
                )
                self._maybe_add_transform_debug(log_entry, 1.0, rotation, translation)
                return (1.0, rotation, translation), log_entry

        log_entry.update(
            {
                "method": "none",
                "scale": 1.0,
            }
        )
        self._maybe_add_transform_debug(
            log_entry,
            1.0,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        return (1.0, np.eye(3), np.zeros(3, dtype=np.float32)), log_entry

    def _get_overlap_positions(
        self,
        prev: CameraPoses,
        curr: CameraPoses,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        prev_positions = prev.get_positions()
        curr_positions = curr.get_positions()

        if prev.frame_indices is not None and curr.frame_indices is not None:
            prev_map = {
                int(idx): prev_positions[i] for i, idx in enumerate(prev.frame_indices)
            }
            curr_map = {
                int(idx): curr_positions[i] for i, idx in enumerate(curr.frame_indices)
            }
            overlap_indices = sorted(set(prev_map) & set(curr_map))
            if len(overlap_indices) >= self.min_overlap_frames:
                prev_overlap = np.stack([prev_map[i] for i in overlap_indices])
                curr_overlap = np.stack([curr_map[i] for i in overlap_indices])
                return prev_overlap, curr_overlap, len(overlap_indices)

        overlap_count = min(len(prev_positions), len(curr_positions))
        if overlap_count < self.min_overlap_frames:
            return None
        overlap_count = self.min_overlap_frames
        prev_overlap = prev_positions[-overlap_count:]
        curr_overlap = curr_positions[:overlap_count]
        return prev_overlap, curr_overlap, overlap_count

    def _align_sim3(
        self,
        source: np.ndarray,
        target: np.ndarray,
        allow_scale: bool,
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """Compute similarity transform from source to target."""
        if source.shape[0] < 2 or target.shape[0] < 2:
            return 1.0, np.eye(3), np.zeros(3, dtype=np.float32), float("inf")

        src_mean = source.mean(axis=0)
        tgt_mean = target.mean(axis=0)
        src_centered = source - src_mean
        tgt_centered = target - tgt_mean

        cov = tgt_centered.T @ src_centered / source.shape[0]
        U, S, Vt = np.linalg.svd(cov)

        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        if allow_scale:
            var_src = np.mean(np.sum(src_centered**2, axis=1))
            scale = np.sum(S) / var_src if var_src > 0 else 1.0
        else:
            scale = 1.0

        t = tgt_mean - scale * R @ src_mean
        aligned = self._apply_sim3(source, scale, R, t)
        rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))

        return float(scale), R.astype(np.float32), t.astype(np.float32), rmse

    def _align_icp(
        self,
        source: PointCloud,
        target: PointCloud,
    ) -> Optional[Tuple[np.ndarray, float]]:
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("Open3D not available, skipping ICP alignment")
            return None

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source.points)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target.points)

        source_down = source_pcd.voxel_down_sample(self.icp_voxel_size)
        target_down = target_pcd.voxel_down_sample(self.icp_voxel_size)

        if len(source_down.points) == 0 or len(target_down.points) == 0:
            return None

        source_down.estimate_normals()
        target_down.estimate_normals()

        init_transform = np.eye(4)
        if self.icp_init == "ransac":
            init_transform = self._estimate_ransac_transform(source_down, target_down)

        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            self.icp_max_corr_dist,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        return result.transformation, float(result.inlier_rmse)

    def _estimate_ransac_transform(self, source_down, target_down) -> np.ndarray:
        import open3d as o3d

        radius_feature = self.icp_voxel_size * 5
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )

        distance_threshold = self.icp_voxel_size * 1.5
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
        )

        return result.transformation

    def _apply_sim3(
        self,
        points: np.ndarray,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        return (rotation @ (points * scale).T).T + translation

    def _transform_pointcloud(
        self,
        pointcloud: PointCloud,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> PointCloud:
        points = self._apply_sim3(pointcloud.points, scale, rotation, translation)

        normals = None
        if pointcloud.normals is not None:
            normals = (rotation @ pointcloud.normals.T).T

        return PointCloud(
            points=points.astype(np.float32),
            colors=pointcloud.colors,
            confidence=pointcloud.confidence,
            normals=normals,
            origin_gps=pointcloud.origin_gps,
            scale=pointcloud.scale,
            is_metric=pointcloud.is_metric,
        )

    def _transform_poses(
        self,
        poses: CameraPoses,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> CameraPoses:
        new_poses = poses.poses.copy()
        for i in range(new_poses.shape[0]):
            pose = new_poses[i]
            pose[:3, :3] = rotation @ pose[:3, :3]
            pose[:3, 3] = rotation @ (pose[:3, 3] * scale) + translation
        return CameraPoses(
            poses=new_poses,
            timestamps=poses.timestamps,
            intrinsics=poses.intrinsics,
            frame_indices=poses.frame_indices,
            pose_convention=poses.pose_convention,
            coordinate_frame=poses.coordinate_frame,
        )

    def _merge_pointclouds(self, clouds: List[PointCloud]) -> PointCloud:
        points = np.concatenate([pc.points for pc in clouds], axis=0)

        colors = None
        if all(pc.colors is not None for pc in clouds):
            colors = np.concatenate([pc.colors for pc in clouds], axis=0)

        confidence = None
        if all(pc.confidence is not None for pc in clouds):
            confidence = np.concatenate([pc.confidence for pc in clouds], axis=0)

        normals = None
        if all(pc.normals is not None for pc in clouds):
            normals = np.concatenate([pc.normals for pc in clouds], axis=0)

        is_metric = all(pc.is_metric for pc in clouds)
        scale = clouds[0].scale if clouds else 1.0

        return PointCloud(
            points=points.astype(np.float32),
            colors=colors,
            confidence=confidence,
            normals=normals,
            scale=scale,
            is_metric=is_metric,
        )

    def _merge_poses(self, poses_list: List[CameraPoses]) -> Optional[CameraPoses]:
        if not poses_list:
            return None
        if any(p is None for p in poses_list):
            return None

        entries = []
        intrinsics_per_frame = False
        for poses in poses_list:
            if poses.intrinsics is not None and poses.intrinsics.ndim == 3:
                intrinsics_per_frame = True
            for i, pose in enumerate(poses.poses):
                frame_idx = None
                if poses.frame_indices is not None:
                    frame_idx = int(poses.frame_indices[i])
                timestamp = (
                    float(poses.timestamps[i]) if poses.timestamps is not None else None
                )
                intrinsic = None
                if poses.intrinsics is not None:
                    intrinsic = (
                        poses.intrinsics[i]
                        if poses.intrinsics.ndim == 3
                        else poses.intrinsics
                    )
                entries.append((frame_idx, pose, timestamp, intrinsic))

        if all(e[0] is not None for e in entries):
            entries_by_idx: Dict[int, Tuple] = {}
            for entry in entries:
                if entry[0] not in entries_by_idx:
                    entries_by_idx[entry[0]] = entry
            entries = [entries_by_idx[k] for k in sorted(entries_by_idx)]

        if not entries:
            return None

        poses_arr = np.stack([e[1] for e in entries], axis=0).astype(np.float32)

        timestamps = None
        if any(e[2] is not None for e in entries):
            timestamps = np.array([e[2] for e in entries], dtype=np.float64)

        frame_indices = None
        if all(e[0] is not None for e in entries):
            frame_indices = np.array([e[0] for e in entries], dtype=np.int64)

        intrinsics = None
        if all(e[3] is not None for e in entries):
            if intrinsics_per_frame:
                intrinsics = np.stack([e[3] for e in entries], axis=0).astype(
                    np.float32
                )
            else:
                intrinsics = entries[0][3].astype(np.float32)

        return CameraPoses(
            poses=poses_arr,
            timestamps=timestamps,
            intrinsics=intrinsics,
            frame_indices=frame_indices,
        )

    def _maybe_add_transform_debug(
        self,
        log_entry: dict,
        scale: float,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> None:
        """Persist the source-unit transform required for durable provenance.

        These values are part of the package lineage contract, not optional
        debug output. ``save_alignment_debug`` remains available for future
        verbose diagnostics but never controls the transform itself.
        """
        log_entry.update(
            {
                "scale": float(scale),
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            }
        )
