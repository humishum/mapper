"""VGGT model wrapper for 3D reconstruction."""

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


class VGGTModel(BaseModel):
    """
    VGGT: Visual Geometry Grounded Transformer.

    VGGT produces metric-scale 3D reconstructions from image sequences.
    It outputs camera intrinsics/extrinsics, depth maps, point maps,
    and 3D point clouds.

    Outputs:
    - Metric-scale point cloud (likely)
    - Camera poses with intrinsics
    - Per-point confidence
    - Depth maps

    Reference: https://github.com/facebookresearch/vggt

    Installation:
        pip install torch torchvision numpy Pillow huggingface_hub
        # Model auto-downloads from HuggingFace on first use
    """

    name = "vggt"
    outputs_metric_scale = True  # To verify with testing
    outputs_poses = True
    outputs_confidence = True
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 100)
        self.model_name = self.config.get("model_name", "facebook/vggt-1b")

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default VGGT configuration."""
        return {
            "model_name": "facebook/vggt-1b",
            "max_frames": 100,
            "device": "cuda",
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        """
        Load VGGT model.

        VGGT auto-downloads weights from HuggingFace on first use.
        """
        try:
            # TODO: Implement actual VGGT loading
            # from vggt.model import VGGT
            # self.model = VGGT.from_pretrained(self.model_name)
            # self.model.cuda().eval()

            logger.warning(
                "VGGT model loading not yet implemented. "
                "Install VGGT and update this wrapper."
            )
            self._is_loaded = True

        except ImportError as e:
            raise ImportError(
                "VGGT not installed. Install with:\n"
                "pip install torch torchvision numpy Pillow huggingface_hub\n"
                f"Error: {e}"
            )

    def reconstruct(
        self,
        video_input: VideoInput,
        output_dir: Path,
    ) -> ReconstructionResult:
        """
        Run VGGT reconstruction.

        TODO: Implement actual VGGT inference.

        Expected workflow:
        1. Load images from video_input.image_dir
        2. Preprocess images for VGGT
        3. Run VGGT inference
        4. Extract point cloud from depth maps + camera params
        5. Save outputs
        """
        self.ensure_loaded()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get image paths
        image_paths = video_input.get_frame_paths()
        n_frames = min(len(image_paths), self.max_frames)

        logger.info(f"VGGT: Processing {n_frames} frames")

        # TODO: Implement actual reconstruction
        # images = self._load_images(image_paths[:n_frames])
        # with torch.no_grad():
        #     outputs = self.model.predict(images)
        # pointcloud = self._build_pointcloud(outputs)
        # poses = self._extract_poses(outputs)

        # Placeholder - return empty result
        logger.warning(
            "VGGT reconstruction not implemented. "
            "Returning empty point cloud."
        )

        return ReconstructionResult(
            pointcloud=PointCloud(
                points=np.zeros((0, 3), dtype=np.float32),
                is_metric=True,  # VGGT should output metric scale
            ),
            poses=None,
            metadata={
                "model": "vggt",
                "status": "not_implemented",
                "frames_requested": n_frames,
            },
        )

    def _load_images(self, image_paths: list) -> list:
        """Load and preprocess images for VGGT."""
        # TODO: Implement image loading with VGGT preprocessing
        pass

    def _build_pointcloud(self, outputs) -> PointCloud:
        """Build PointCloud from VGGT outputs."""
        # TODO: Extract point cloud from depth maps and camera params
        pass

    def _extract_poses(self, outputs) -> Optional[CameraPoses]:
        """Extract camera poses from VGGT outputs."""
        # TODO: Convert VGGT camera params to CameraPoses
        pass
