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
        self.model_name = self.config.get("model_name", "facebook/VGGT-1B")
        self.weights_path = self.config.get("weights_path")
        self.device = self.config.get("device", "cuda")
        self.image_size = self.config.get("image_size", 518)
        self.preprocess_mode = self.config.get("preprocess_mode", "square")
        self.use_point_map = self.config.get("use_point_map", False)
        self.min_confidence = self.config.get("min_confidence", 5.0)
        self.max_points = self.config.get("max_points", 500000)
        self.sample_seed = self.config.get("sample_seed")

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default VGGT configuration."""
        return {
            "model_name": "facebook/VGGT-1B",
            "max_frames": 100,
            "device": "cuda",
            "image_size": 518,
            "preprocess_mode": "square",  # square | pad | crop
            "use_point_map": False,  # False uses depth unprojection
            "min_confidence": 5.0,
            "max_points": 500000,
            "sample_seed": None,
            "weights_path": None,
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        """
        Load VGGT model.

        VGGT auto-downloads weights from HuggingFace on first use.
        """
        try:
            import torch
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise ImportError(
                "VGGT not installed. Install with:\n"
                "pip install torch torchvision numpy Pillow huggingface_hub\n"
                f"Error: {e}"
            )

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device

        weights = weights_path or self.weights_path
        if weights is not None:
            weights = Path(weights)

        if weights is not None and weights.exists():
            model = VGGT(img_size=self.image_size)
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)
            self.model = model
        else:
            self.model = VGGT.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

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
        images_list = [str(p) for p in image_paths]

        if len(images_list) == 0:
            raise ValueError(f"No images found in {video_input.image_dir}")

        if self.max_frames is not None and len(images_list) > self.max_frames:
            logger.warning(
                f"Limiting to {self.max_frames} frames (had {len(images_list)}) for VRAM constraints"
            )
            images_list = images_list[: self.max_frames]

        logger.info(f"VGGT: Processing {len(images_list)} frames")

        images = self._load_images(images_list)

        outputs = self._run_inference(images)
        pointcloud = self._build_pointcloud(images, outputs)
        poses = self._extract_poses(images, outputs, video_input)

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={
                "model": "vggt",
                "frames_processed": len(images_list),
                "image_size": self.image_size,
                "preprocess_mode": self.preprocess_mode,
                "min_confidence": self.min_confidence,
                "use_point_map": self.use_point_map,
            },
        )

    def _load_images(self, image_paths: list) -> list:
        """Load and preprocess images for VGGT."""
        from vggt.utils.load_fn import (
            load_and_preprocess_images,
            load_and_preprocess_images_square,
        )

        if self.preprocess_mode == "square":
            images, _ = load_and_preprocess_images_square(
                image_paths, target_size=self.image_size
            )
        elif self.preprocess_mode in {"pad", "crop"}:
            if self.image_size != 518:
                logger.warning(
                    "preprocess_mode %s uses fixed size 518; image_size=%s ignored",
                    self.preprocess_mode,
                    self.image_size,
                )
            images = load_and_preprocess_images(
                image_paths, mode=self.preprocess_mode
            )
        else:
            raise ValueError(
                f"Invalid preprocess_mode '{self.preprocess_mode}'. "
                "Use 'square', 'pad', or 'crop'."
            )

        return images.to(self.device)

    def _run_inference(self, images):
        """Run VGGT forward pass."""
        import torch

        if self.device.startswith("cuda"):
            amp_dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
            amp_enabled = True
        else:
            amp_dtype = torch.float32
            amp_enabled = False

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                outputs = self.model(images)

        return outputs

    def _build_pointcloud(self, images, outputs) -> PointCloud:
        """Build PointCloud from VGGT outputs."""
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        pose_enc = outputs.get("pose_enc")
        if pose_enc is None:
            raise ValueError("VGGT outputs missing pose_enc")

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )

        if self.use_point_map:
            points_map = outputs.get("world_points")
            conf_map = outputs.get("world_points_conf")
            if points_map is None or conf_map is None:
                raise ValueError("VGGT outputs missing point map predictions")
            points_map = points_map.squeeze(0).cpu().numpy()
            conf_map = conf_map.squeeze(0).cpu().numpy()
        else:
            depth_map = outputs.get("depth")
            conf_map = outputs.get("depth_conf")
            if depth_map is None or conf_map is None:
                raise ValueError("VGGT outputs missing depth predictions")
            points_map = unproject_depth_map_to_point_map(
                depth_map.squeeze(0),
                extrinsic.squeeze(0),
                intrinsic.squeeze(0),
            )
            conf_map = conf_map.squeeze(0).cpu().numpy()

        points = points_map.reshape(-1, 3).astype(np.float32)
        confidence = conf_map.reshape(-1).astype(np.float32)

        colors = self._extract_colors(images)

        if self.min_confidence is not None:
            mask = confidence >= self.min_confidence
            points = points[mask]
            confidence = confidence[mask]
            if colors is not None:
                colors = colors[mask]

        if self.max_points is not None and len(points) > self.max_points:
            rng = np.random.default_rng(self.sample_seed)
            keep_idx = rng.choice(len(points), size=self.max_points, replace=False)
            points = points[keep_idx]
            confidence = confidence[keep_idx]
            if colors is not None:
                colors = colors[keep_idx]

        if len(points) == 0:
            logger.warning("VGGT produced no points after filtering")

        return PointCloud(
            points=points,
            colors=colors,
            confidence=confidence,
            is_metric=True,
        )

    def _extract_colors(self, images) -> Optional[np.ndarray]:
        """Extract colors aligned to the point map."""
        if images is None:
            return None

        imgs = images.detach().cpu().numpy()
        if imgs.ndim == 5:
            imgs = imgs.squeeze(0)
        imgs = np.clip(imgs, 0.0, 1.0)
        imgs = (imgs * 255).astype(np.uint8)
        imgs = np.transpose(imgs, (0, 2, 3, 1))
        return imgs.reshape(-1, 3)

    def _extract_poses(
        self, images, outputs, video_input: VideoInput
    ) -> Optional[CameraPoses]:
        """Extract camera poses from VGGT outputs."""
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        pose_enc = outputs.get("pose_enc")
        if pose_enc is None:
            return None

        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images.shape[-2:]
        )

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        if extrinsic.ndim == 2:
            extrinsic = extrinsic.reshape(-1, 3, 4)

        poses = np.zeros((extrinsic.shape[0], 4, 4), dtype=np.float32)
        poses[:, :3, :4] = extrinsic.astype(np.float32)
        poses[:, 3, 3] = 1.0

        intrinsics = intrinsic.squeeze(0).cpu().numpy()

        timestamps = None
        if video_input.frame_count > 0:
            timestamps = video_input.get_frame_timestamps()
            if len(timestamps) > poses.shape[0]:
                step = len(timestamps) // poses.shape[0]
                timestamps = timestamps[::step][: poses.shape[0]]

        return CameraPoses(
            poses=poses,
            timestamps=timestamps,
            intrinsics=intrinsics.astype(np.float32),
        )
