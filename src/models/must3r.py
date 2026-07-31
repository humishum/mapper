"""must3r model wrapper for 3D reconstruction."""

from pathlib import Path
from typing import Optional, List, Tuple
import logging
import numpy as np

from .base import BaseModel
from ..core.types import (
    ReconstructionResult,
    PointCloud,
    CameraPoses,
    PoseConvention,
    VideoInput,
)

logger = logging.getLogger(__name__)


class MUSt3RModel(BaseModel):
    """
    MUSt3R: Multi-view Network for Stereo 3D Reconstruction.


    Outputs:
    - Dense colored point cloud
    - Per-point confidence scores
    - Camera poses (relative, not metric scale)

    Reference: https://github.com/naver/must3r
    """

    name = "must3r"
    outputs_metric_scale = False  # MUSt3R outputs relative scale only
    outputs_poses = True
    outputs_confidence = True
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # Model parameters from config
        self.weights_path = self.config.get("weights_path")
        self.retrieval_path = self.config.get("retrieval_path")
        self.image_size = self.config.get("image_size", 512)
        self.device = self.config.get("device", "cuda")

        # Processing parameters
        self.use_chunking = self.config.get("use_chunking", False)
        self.window_size = self.config.get("window_size", 500)
        self.window_overlap = self.config.get("window_overlap", 20)
        # Threshold families were previously exported as separate PLY files.  The
        # canonical publisher keeps confidence as a point attribute, so inference
        # now applies one minimum threshold and never round-trips through PLY.
        legacy_thresholds = self.config.get("confidence_thresholds", [5.0])
        self.min_confidence = float(
            self.config.get(
                "min_confidence",
                legacy_thresholds[0] if legacy_thresholds else 1.05,
            )
        )
        self.num_mem_imgs = self.config.get("num_mem_imgs", 50)
        self.subsample = self.config.get("subsample", 2)

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default MUSt3R configuration."""
        return {
            "weights_path": "weights/must3r/MUSt3R_512.pth",
            "retrieval_path": (
                "weights/must3r/MUSt3R_512_retrieval_trainingfree.pth"
            ),
            "image_size": 512,
            "device": "cuda",
            "use_chunking": False,
            "window_size": 500,
            "window_overlap": 20,
            "min_confidence": 5.0,
            "num_mem_imgs": 50,
            "subsample": 2,
            "min_conf_thr": 1.05,
            "execution_mode": "linseq",
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        """Load MUSt3R model weights."""
        from must3r.model import load_model
        from must3r.model.blocks.attention import toggle_memory_efficient_attention

        # Use provided path or fall back to config
        weights = weights_path or self.weights_path
        if weights is None:
            raise ValueError(
                "weights_path must be provided either in config or load() argument"
            )
        weights = self.resolve_workspace_path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"MUSt3R checkpoint not found: {weights}. Run "
                "`scripts/setup_models/download_models.sh must3r`."
            )
        if self.retrieval_path is None:
            raise ValueError("retrieval_path must be provided")
        self.retrieval_path = self.resolve_workspace_path(self.retrieval_path)
        if not self.retrieval_path.is_file():
            raise FileNotFoundError(
                f"MUSt3R retrieval checkpoint not found: {self.retrieval_path}"
            )

        # Enable memory-efficient attention
        toggle_memory_efficient_attention(enabled=True)

        logger.info(f"Loading MUSt3R model from {weights}")
        self.model = load_model(
            weights,
            encoder=None,
            decoder=None,
            device=self.device,
            img_size=self.image_size,
            memory_mode=None,
        )
        self._is_loaded = True
        logger.info("MUSt3R model loaded successfully")

    def reconstruct(
        self,
        video_input: VideoInput,
        output_dir: Path,
    ) -> ReconstructionResult:
        """
        Run MUSt3R reconstruction on video frames.

        For long videos, this handles windowing automatically to avoid OOM.

        Args:
            video_input: Video input with extracted frames
            output_dir: Directory to save PLY outputs

        Returns:
            ReconstructionResult with point cloud, poses, and metadata
        """
        self.ensure_loaded()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get sorted image paths
        image_paths = video_input.get_frame_paths()
        images = [str(p) for p in image_paths]

        if len(images) == 0:
            raise ValueError(f"No images found in {video_input.image_dir}")

        windows = self.build_windows(
            len(images),
            self.use_chunking,
            self.window_size,
            self.window_overlap,
        )
        if len(windows) > 1:
            logger.info(
                "Chunking enabled: %d windows (size=%d, overlap=%d)",
                len(windows),
                self.window_size,
                self.window_overlap,
            )

        chunk_results: List[ReconstructionResult] = []
        use_window_dirs = len(windows) > 1
        for window_id, (start, end) in enumerate(windows):
            window_images = images[start:end]
            frame_indices = list(range(start, end))
            window_output_dir = (
                output_dir / f"window_{window_id:03d}"
                if use_window_dirs
                else output_dir
            )
            chunk_results.append(
                self._reconstruct_window(
                    window_images,
                    video_input,
                    window_output_dir,
                    frame_indices,
                    window_id,
                )
            )

        if len(chunk_results) == 1:
            return chunk_results[0]

        return ReconstructionResult(
            pointcloud=chunk_results[0].pointcloud,
            poses=chunk_results[0].poses,
            metadata={
                "model": "must3r",
                "image_count": len(images),
                "min_confidence": self.min_confidence,
                "window_size": self.window_size,
                "window_overlap": self.window_overlap,
                "image_size": self.image_size,
                "use_chunking": True,
                "num_chunks": len(chunk_results),
            },
            chunks=chunk_results,
        )

    def _build_windows(self, total_images: int) -> List[Tuple[int, int]]:
        """Backward-compatible window builder."""
        return self.build_windows(
            total_images,
            self.use_chunking,
            self.window_size,
            self.window_overlap,
        )

    def _reconstruct_window(
        self,
        images: List[str],
        video_input: VideoInput,
        output_dir: Path,
        frame_indices: List[int],
        window_id: int,
    ) -> ReconstructionResult:
        """Run reconstruction on a single window of images."""
        from must3r.demo.gradio import get_reconstructed_scene

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running MUSt3R reconstruction on {len(images)} images")
        logger.info(f"Output directory: {output_dir}")

        # Adjust num_mem_imgs based on actual image count
        num_mem_imgs = min(self.num_mem_imgs, len(images))

        # Run reconstruction
        scene, _ = get_reconstructed_scene(
            outdir=str(output_dir),
            viser_server=None,
            should_save_glb=False,
            model=self.model,
            retrieval=self.retrieval_path,
            device=self.device,
            verbose=True,
            image_size=self.image_size,
            amp=False,
            filelist=images,
            min_conf_thr=self.config.get("min_conf_thr", 1.05),
            as_pointcloud=True,
            transparent_cams=False,
            local_pointmaps=False,
            cam_size=0.05,
            num_mem_images=num_mem_imgs,
            max_bs=1,
            render_once=False,
            camera_conf_thr=0.0,
            num_refinements_iterations=0,
            execution_mode=self.config.get("execution_mode", "linseq"),
            vidseq_local_context_size=0,
            keyframe_interval=3,
            slam_local_context_size=0,
            subsample=self.subsample,
            min_conf_keyframe=1.5,
            keyframe_overlap_thr=0.05,
            overlap_percentile=85,
        )

        # Extract directly from the scene so confidence survives publication.
        # The old threshold-family PLY exports consumed substantial disk and then
        # discarded confidence when they were loaded again.
        pointcloud = self._extract_pointcloud(scene)

        # Extract camera poses from scene
        poses = self._extract_poses(scene, video_input, frame_indices=frame_indices)
        source_frame_indices = video_input.get_source_frame_indices()[frame_indices]

        window_metadata = {
            "window_id": window_id,
            "frame_start": int(source_frame_indices[0]) if frame_indices else 0,
            "frame_end": int(source_frame_indices[-1]) if frame_indices else 0,
            "frame_indices": source_frame_indices.tolist(),
            "window_size": self.window_size,
            "window_overlap": self.window_overlap,
        }

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={
                "model": "must3r",
                "image_count": len(images),
                "min_confidence": self.min_confidence,
                "window_size": self.window_size,
                "image_size": self.image_size,
            },
            window_metadata=window_metadata,
        )

    def _extract_pointcloud(self, scene) -> PointCloud:
        """Extract aligned points, RGB and confidence directly from a scene.

        ``SceneState.x_out`` is the authoritative MUSt3R result.  Each entry
        contains point and confidence maps in the same shape as the corresponding
        image.  Keeping them together here avoids the lossy PLY interchange used
        by the original experiment wrapper.
        """

        if not hasattr(scene, "x_out") or not hasattr(scene, "imgs"):
            raise ValueError("MUSt3R scene is missing x_out/imgs")
        if len(scene.x_out) != len(scene.imgs):
            raise ValueError("MUSt3R point maps and images have different lengths")

        point_parts: List[np.ndarray] = []
        color_parts: List[np.ndarray] = []
        confidence_parts: List[np.ndarray] = []

        for index, (prediction, image) in enumerate(zip(scene.x_out, scene.imgs)):
            if "pts3d" not in prediction or "conf" not in prediction:
                raise ValueError(f"MUSt3R prediction {index} lacks pts3d or conf")

            points = self._as_numpy(prediction["pts3d"]).reshape(-1, 3)
            confidence = self._as_numpy(prediction["conf"]).reshape(-1)
            colors = self._image_colors(image)
            if len(points) != len(confidence) or len(points) != len(colors):
                raise ValueError(
                    f"MUSt3R prediction {index} has inconsistent point, "
                    "confidence, and color shapes"
                )

            keep = (
                np.isfinite(points).all(axis=1)
                & np.isfinite(confidence)
                & (confidence >= self.min_confidence)
            )
            point_parts.append(points[keep].astype(np.float32, copy=False))
            color_parts.append(colors[keep])
            confidence_parts.append(confidence[keep].astype(np.float32, copy=False))

        if not point_parts:
            return PointCloud(points=np.empty((0, 3), dtype=np.float32))

        return PointCloud(
            points=np.concatenate(point_parts),
            colors=np.concatenate(color_parts),
            confidence=np.concatenate(confidence_parts),
            is_metric=False,
        )

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @classmethod
    def _image_colors(cls, image) -> np.ndarray:
        colors = cls._as_numpy(image)
        if colors.ndim == 3 and colors.shape[0] == 3 and colors.shape[-1] != 3:
            colors = np.moveaxis(colors, 0, -1)
        if colors.ndim != 3 or colors.shape[-1] != 3:
            raise ValueError(f"MUSt3R image has unsupported shape {colors.shape}")
        colors = colors.reshape(-1, 3)
        if np.issubdtype(colors.dtype, np.floating):
            finite_max = float(np.nanmax(colors)) if colors.size else 0.0
            if finite_max <= 1.0:
                colors = colors * 255.0
        return np.clip(colors, 0, 255).astype(np.uint8)

    def _extract_poses(
        self,
        scene,
        video_input: VideoInput,
        frame_indices: Optional[List[int]] = None,
    ) -> Optional[CameraPoses]:
        """Extract camera poses from MUSt3R scene object."""
        try:
            if hasattr(scene, "cams2world") and scene.cams2world is not None:
                poses = scene.cams2world
                if isinstance(poses, list):
                    poses = np.stack(
                        [
                            p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
                            for p in poses
                        ],
                        axis=0,
                    )
                elif hasattr(poses, "cpu"):
                    poses = poses.cpu().numpy()
                else:
                    poses = np.asarray(poses)

                # Ensure correct shape
                if poses.ndim == 2:  # (N, 16) flattened
                    poses = poses.reshape(-1, 4, 4)

                if frame_indices is None:
                    frame_indices = list(range(video_input.frame_count))
                selection = np.asarray(frame_indices, dtype=np.int64)
                if len(selection) != poses.shape[0]:
                    raise ValueError(
                        "MUSt3R output count does not match the selected input "
                        f"frames: {poses.shape[0]} poses for {len(selection)} frames"
                    )
                timestamps = video_input.get_frame_timestamps()[selection]
                source_frame_indices = video_input.get_source_frame_indices()[
                    selection
                ]

                # Try to get intrinsics
                intrinsics = None
                if hasattr(scene, "focals") and scene.focals is not None:
                    focals = np.asarray(
                        [
                            self._as_numpy(focal).reshape(-1)[0]
                            for focal in scene.focals
                        ],
                        dtype=np.float32,
                    )
                    if len(focals) != len(poses):
                        raise ValueError(
                            "MUSt3R focal count does not match its pose count"
                        )
                    shapes = np.repeat(
                        np.array([[self.image_size, self.image_size]]),
                        len(poses),
                        axis=0,
                    )
                    if (
                        hasattr(scene, "true_shape")
                        and scene.true_shape is not None
                    ):
                        candidate_shapes = self._as_numpy(scene.true_shape).reshape(
                            len(poses), -1
                        )
                        if candidate_shapes.shape[1] >= 2:
                            shapes = candidate_shapes[:, :2]
                    intrinsics = np.repeat(
                        np.eye(3, dtype=np.float32)[None], len(poses), axis=0
                    )
                    intrinsics[:, 0, 0] = focals
                    intrinsics[:, 1, 1] = focals
                    intrinsics[:, 0, 2] = shapes[:, 1] / 2.0
                    intrinsics[:, 1, 2] = shapes[:, 0] / 2.0

                return CameraPoses(
                    poses=poses.astype(np.float32),
                    timestamps=timestamps,
                    intrinsics=intrinsics,
                    frame_indices=source_frame_indices,
                    pose_convention=PoseConvention.CAMERA_TO_WORLD,
                )
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Could not extract poses from scene: {e}")

        return None


# Backward-compatible import for existing experiments.  The upstream repository
# and model are MUSt3R; MASt3R is a different Naver model family.
MASt3RModel = MUSt3RModel
