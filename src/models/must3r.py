"""must3r model wrapper for 3D reconstruction."""

from pathlib import Path
from typing import Optional, List, Tuple
import os
import logging
import numpy as np

from .base import BaseModel
from ..core.types import (
    ReconstructionResult,
    PointCloud,
    CameraPoses,
    VideoInput,
)

logger = logging.getLogger(__name__)


class MASt3RModel(BaseModel):
    """
    Must3r: Multi-view Stereo 3D Reconstruction.


    Outputs:
    - Dense colored point cloud
    - Per-point confidence scores
    - Camera poses (relative, not metric scale)

    Reference: https://github.com/naver/must3r
    """

    name = "must3r"
    outputs_metric_scale = False  # MASt3R outputs relative scale only
    outputs_poses = True
    outputs_confidence = True
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # Model parameters from config
        self.weights_path = self.config.get("weights_path")
        self.retrieval_path = self.config.get("retrieval_path")
        self.image_size = self.config.get("image_size", 512)

        # Processing parameters
        self.use_chunking = self.config.get("use_chunking", False)
        self.window_size = self.config.get("window_size", 500)
        self.window_overlap = self.config.get("window_overlap", 20)
        self.confidence_thresholds = self.config.get(
            "confidence_thresholds", [5.0, 2.0, 1.5, 1.05]
        )
        self.num_mem_imgs = self.config.get("num_mem_imgs", 50)
        self.subsample = self.config.get("subsample", 2)

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default MASt3R configuration."""
        return {
            "weights_path": None,  # Required - must be set
            "retrieval_path": None,  # Required - must be set
            "image_size": 512,
            "use_chunking": False,
            "window_size": 500,
            "window_overlap": 20,
            "confidence_thresholds": [5.0],  # [5.0, 2.0, 1.5, 1.05],
            "num_mem_imgs": 50,
            "subsample": 2,
            "min_conf_thr": 1.05,
            "execution_mode": "linseq",
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        """Load MASt3R model weights."""
        from must3r.model import load_model
        from must3r.model.blocks.attention import toggle_memory_efficient_attention

        # Use provided path or fall back to config
        weights = weights_path or self.weights_path
        if weights is None:
            raise ValueError(
                "weights_path must be provided either in config or load() argument"
            )

        # Enable memory-efficient attention
        toggle_memory_efficient_attention(enabled=True)

        logger.info(f"Loading MASt3R model from {weights}")
        self.model = load_model(
            weights,
            encoder=None,
            decoder=None,
            device="cuda",
            img_size=self.image_size,
            memory_mode=None,
        )
        self._is_loaded = True
        logger.info("MASt3R model loaded successfully")

    def reconstruct(
        self,
        video_input: VideoInput,
        output_dir: Path,
    ) -> ReconstructionResult:
        """
        Run MASt3R reconstruction on video frames.

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
                "thresholds": self.confidence_thresholds,
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
        from must3r.demo.gradio import get_reconstructed_scene, get_3D_model_from_scene

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running MASt3R reconstruction on {len(images)} images")
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
            device="cuda",
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

        # Save PLY files at different confidence thresholds
        for thr in self.confidence_thresholds:
            try:
                logger.info(f"Generating PLY file with confidence threshold {thr}")
                get_3D_model_from_scene(
                    outdir=str(output_dir),
                    verbose=True,
                    scene=scene,
                    min_conf_thr=thr,
                    as_pointcloud=True,
                    transparent_cams=False,
                    cam_size=0.05,
                    filename=f"scene_thr{thr}.ply",
                )
            except Exception as e:
                logger.error(f"Error generating PLY at threshold {thr}: {e}")

        # Extract point cloud from scene at highest threshold
        pointcloud = self._extract_pointcloud(scene, output_dir)

        # Extract camera poses from scene
        poses = self._extract_poses(scene, video_input, frame_indices=frame_indices)

        window_metadata = {
            "window_id": window_id,
            "frame_start": frame_indices[0] if frame_indices else 0,
            "frame_end": frame_indices[-1] if frame_indices else 0,
            "frame_indices": frame_indices,
            "window_size": self.window_size,
            "window_overlap": self.window_overlap,
        }

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={
                "model": "must3r",
                "image_count": len(images),
                "thresholds": self.confidence_thresholds,
                "window_size": self.window_size,
                "image_size": self.image_size,
            },
            window_metadata=window_metadata,
        )

    def _extract_pointcloud(self, scene, output_dir: Path) -> PointCloud:
        """Extract PointCloud from MASt3R scene object."""
        # Load the highest-threshold PLY we just saved
        best_threshold = self.confidence_thresholds[0]
        ply_path = output_dir / f"scene_thr{best_threshold}.ply"

        if ply_path.exists():
            return PointCloud.from_ply(ply_path)

        # Fallback: try to extract directly from scene
        # MASt3R scene structure varies, so we handle multiple cases
        try:
            if hasattr(scene, "pts3d") and scene.pts3d is not None:
                pts3d = scene.pts3d
                if hasattr(pts3d, "cpu"):
                    pts3d = pts3d.cpu().numpy()

                # Reshape if needed
                if pts3d.ndim == 4:  # (B, H, W, 3)
                    pts3d = pts3d.reshape(-1, 3)
                elif pts3d.ndim == 3:  # (N, H*W, 3) or similar
                    pts3d = pts3d.reshape(-1, 3)

                # Try to get colors
                colors = None
                if hasattr(scene, "imgs") and scene.imgs is not None:
                    imgs = scene.imgs
                    if hasattr(imgs, "cpu"):
                        imgs = imgs.cpu().numpy()
                    # Reshape to match points
                    if imgs.ndim >= 3:
                        colors = (imgs.reshape(-1, 3) * 255).astype(np.uint8)

                # Try to get confidence
                confidence = None
                if hasattr(scene, "conf") and scene.conf is not None:
                    conf = scene.conf
                    if hasattr(conf, "cpu"):
                        conf = conf.cpu().numpy()
                    confidence = conf.flatten()

                return PointCloud(
                    points=pts3d.astype(np.float32),
                    colors=colors,
                    confidence=confidence,
                    is_metric=False,
                )
        except Exception as e:
            logger.warning(f"Could not extract point cloud from scene: {e}")
            raise
        # TODO: handle this better
        # logger.warning("Returning empty point cloud - extraction failed")
        # return PointCloud(points=np.zeros((0, 3), dtype=np.float32))

    def _extract_poses(
        self,
        scene,
        video_input: VideoInput,
        frame_indices: Optional[List[int]] = None,
    ) -> Optional[CameraPoses]:
        """Extract camera poses from MASt3R scene object."""
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

                # Get timestamps from video input
                if frame_indices is None:
                    frame_indices = list(range(video_input.frame_count))
                frame_indices = np.array(frame_indices, dtype=np.int64)

                timestamps = None
                if len(frame_indices) > 0 and video_input.fps > 0:
                    timestamps = frame_indices / float(video_input.fps)

                if len(frame_indices) > poses.shape[0]:
                    step = len(frame_indices) // poses.shape[0]
                    frame_indices = frame_indices[::step][: poses.shape[0]]
                    if timestamps is not None:
                        timestamps = timestamps[::step][: poses.shape[0]]

                # Try to get intrinsics
                intrinsics = None
                if hasattr(scene, "focals") and scene.focals is not None:
                    focals = scene.focals
                    if isinstance(focals, list):
                        focals = np.array(
                            [
                                f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f)
                                for f in focals
                            ]
                        )
                    elif hasattr(focals, "cpu"):
                        focals = focals.cpu().numpy()
                    else:
                        focals = np.asarray(focals)
                    # Build intrinsics matrix
                    # Assume principal point at image center
                    cx, cy = self.image_size / 2, self.image_size / 2
                    if focals.ndim == 1:
                        fx = fy = float(focals[0])
                    else:
                        fx = fy = float(focals.mean())
                    intrinsics = np.array(
                        [
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1],
                        ],
                        dtype=np.float32,
                    )

                return CameraPoses(
                    poses=poses.astype(np.float32),
                    timestamps=timestamps,
                    intrinsics=intrinsics,
                    frame_indices=frame_indices,
                )
        except Exception as e:
            logger.warning(f"Could not extract poses from scene: {e}")

        return None
