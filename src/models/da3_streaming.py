"""Depth Anything 3 Streaming model wrapper."""

from pathlib import Path
from typing import Optional
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


class DA3StreamingModel(BaseModel):
    """
    Depth Anything 3 - Streaming variant.

    DA3-Streaming is designed for video input with temporal consistency.
    It outputs metric-scale depth maps, camera poses, and fused point clouds.

    Outputs:
    - camera_poses.txt - Extrinsic matrix parameters
    - intrinsic.txt - Camera intrinsics (fx, fy, cx, cy)
    - combined_pcd.ply - Fused point cloud from all frames
    - Per-frame depth maps (optional)

    Reference: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/

    Installation:
        # Clone the DA3 repository
        git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
        cd Depth-Anything-3
        pip install -r requirements.txt
    """

    name = "da3_streaming"
    outputs_metric_scale = True  # To verify with testing
    outputs_poses = True
    outputs_confidence = True
    supports_video_input = True  # Designed for video

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.save_per_frame_depth = self.config.get("save_per_frame_depth", False)
        self.da3_path = self.config.get("da3_path")  # Path to DA3 installation

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default DA3-Streaming configuration."""
        return {
            "da3_path": None,  # Path to DA3 installation
            "save_per_frame_depth": False,
            "device": "cuda",
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        """
        Load DA3-Streaming model.

        DA3 requires the repository to be installed separately.
        """
        # TODO: Implement actual DA3 loading
        logger.warning(
            "DA3-Streaming model loading not yet implemented. "
            "Clone DA3 repo and update this wrapper."
        )
        self._is_loaded = True

    def reconstruct(
        self,
        video_input: VideoInput,
        output_dir: Path,
    ) -> ReconstructionResult:
        """
        Run DA3-Streaming reconstruction.

        TODO: Implement actual DA3 inference.

        Expected workflow:
        1. Run DA3-Streaming on image directory
        2. Parse camera_poses.txt and intrinsic.txt
        3. Load combined_pcd.ply
        4. Return as ReconstructionResult
        """
        self.ensure_loaded()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = video_input.get_frame_paths()
        logger.info(f"DA3-Streaming: Processing {len(image_paths)} frames")

        # TODO: Implement actual reconstruction
        # Expected DA3-Streaming outputs:
        # - output_dir/camera_poses.txt
        # - output_dir/intrinsic.txt
        # - output_dir/combined_pcd.ply

        # self._run_da3_streaming(video_input.image_dir, output_dir)
        # pointcloud = self._load_combined_ply(output_dir / "combined_pcd.ply")
        # poses = self._load_poses(output_dir / "camera_poses.txt")
        # intrinsics = self._load_intrinsics(output_dir / "intrinsic.txt")

        # Placeholder - return empty result
        logger.warning(
            "DA3-Streaming reconstruction not implemented. "
            "Returning empty point cloud."
        )

        return ReconstructionResult(
            pointcloud=PointCloud(
                points=np.zeros((0, 3), dtype=np.float32),
                is_metric=True,  # DA3 should output metric scale
            ),
            poses=None,
            metadata={
                "model": "da3_streaming",
                "status": "not_implemented",
                "frames": len(image_paths),
            },
        )

    def _run_da3_streaming(self, image_dir: Path, output_dir: Path) -> None:
        """Run DA3-Streaming inference."""
        # TODO: Call DA3-Streaming
        # Either via subprocess or by importing the module
        pass

    def _load_combined_ply(self, ply_path: Path) -> PointCloud:
        """Load the combined point cloud from DA3 output."""
        if ply_path.exists():
            return PointCloud.from_ply(ply_path)
        return PointCloud(points=np.zeros((0, 3), dtype=np.float32))

    def _load_poses(self, poses_path: Path) -> Optional[CameraPoses]:
        """Load camera poses from DA3 output."""
        # TODO: Parse camera_poses.txt format
        pass

    def _load_intrinsics(self, intrinsics_path: Path) -> Optional[np.ndarray]:
        """Load camera intrinsics from DA3 output."""
        # TODO: Parse intrinsic.txt format
        # Expected format: fx, fy, cx, cy
        pass
