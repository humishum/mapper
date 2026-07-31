"""VGGT-Omega adapter for bounded dense reconstruction windows."""

from pathlib import Path
from typing import List, Optional
import logging
import os
import subprocess
import sys

import numpy as np

from .base import BaseModel
from ..core.types import (
    CameraPoses,
    PointCloud,
    PoseConvention,
    ReconstructionResult,
    VideoInput,
)

logger = logging.getLogger(__name__)


class VGGTOmegaModel(BaseModel):
    """Run VGGT-Omega over one or more bounded windows.

    VGGT-Omega improves robustness in unconstrained/dynamic scenes, but it is
    not a long-sequence SLAM backend.  Chunk outputs therefore remain separate
    for Mapper's alignment stage.

    Reference: https://github.com/facebookresearch/vggt-omega
    """

    name = "vggt_omega"
    outputs_metric_scale = False
    outputs_poses = True
    outputs_confidence = True
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.vggt_omega_path = self.config.get("vggt_omega_path")
        self.checkpoint_path = self.config.get("checkpoint_path")
        self.device = self.config.get("device", "cuda")
        self.image_resolution = int(self.config.get("image_resolution", 512))
        self.preprocess_mode = self.config.get("preprocess_mode", "max_size")
        self.use_chunking = self.config.get("use_chunking", True)
        self.window_size = int(self.config.get("window_size", 100))
        self.window_overlap = int(self.config.get("window_overlap", 20))
        self.confidence_percentile = float(
            self.config.get("confidence_percentile", 40.0)
        )
        self.max_points = self.config.get("max_points", 500_000)
        self.sample_seed = self.config.get("sample_seed")
        self.upstream_revision = self.config.get(
            "upstream_revision",
            "39a0cb8af88554f15ddcb5354cd52bde588fa014",
        )
        self.verify_upstream_revision = self.config.get(
            "verify_upstream_revision", True
        )
        self.vggt_omega_dir: Optional[Path] = None

    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "vggt_omega_path": None,
            "checkpoint_path": "weights/vggt_omega/vggt_omega_1b_512.pt",
            "device": "cuda",
            "image_resolution": 512,
            "preprocess_mode": "max_size",
            "use_chunking": True,
            "window_size": 100,
            "window_overlap": 20,
            "confidence_percentile": 40.0,
            "max_points": 500_000,
            "sample_seed": None,
            "upstream_revision": "39a0cb8af88554f15ddcb5354cd52bde588fa014",
            "verify_upstream_revision": True,
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        repo_dir = self._resolve_repo_dir(required=False)
        if repo_dir is not None:
            self.vggt_omega_dir = repo_dir
            self._verify_revision(repo_dir)
            if str(repo_dir) not in sys.path:
                sys.path.insert(0, str(repo_dir))
        elif self.verify_upstream_revision:
            raise FileNotFoundError(
                "A VGGT-Omega checkout is required to verify upstream_revision; "
                "set vggt_omega_path or disable verify_upstream_revision."
            )

        try:
            import torch
            from vggt_omega.models import VGGTOmega
        except ImportError as exc:
            raise ImportError(
                "VGGT-Omega is not installed. Install the pinned checkout with "
                "`pip install -e .` in its own inference environment."
            ) from exc

        checkpoint = weights_path or self.checkpoint_path
        if checkpoint is None:
            raise ValueError(
                "checkpoint_path is required; VGGT-Omega checkpoints are gated"
            )
        checkpoint = self.resolve_workspace_path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("VGGT-Omega requires the configured CUDA device")
        self.model = VGGTOmega().to(self.device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self._is_loaded = True

    def reconstruct(
        self, video_input: VideoInput, output_dir: Path
    ) -> ReconstructionResult:
        self.ensure_loaded()
        image_paths = video_input.get_frame_paths()
        if not image_paths:
            raise ValueError(f"No images found in {video_input.image_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        windows = self.build_windows(
            len(image_paths),
            self.use_chunking,
            self.window_size,
            self.window_overlap,
        )
        chunks = []
        for window_id, (start, end) in enumerate(windows):
            window_dir = (
                output_dir / f"window_{window_id:03d}"
                if len(windows) > 1
                else output_dir
            )
            chunks.append(
                self._reconstruct_window(
                    [str(path) for path in image_paths[start:end]],
                    video_input,
                    window_dir,
                    list(range(start, end)),
                    window_id,
                )
            )
        if len(chunks) == 1:
            return chunks[0]
        return ReconstructionResult(
            pointcloud=chunks[0].pointcloud,
            poses=chunks[0].poses,
            chunks=chunks,
            metadata={
                "model": self.name,
                "frames_processed": len(image_paths),
                "num_chunks": len(chunks),
                "window_size": self.window_size,
                "window_overlap": self.window_overlap,
                "upstream_revision": self.upstream_revision,
                "scale_status": "relative",
            },
        )

    def _reconstruct_window(
        self,
        image_paths: List[str],
        video_input: VideoInput,
        output_dir: Path,
        selection_indices: List[int],
        window_id: int,
    ) -> ReconstructionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        images = self._load_images(image_paths)
        predictions = self._run_inference(images)
        extrinsics, intrinsics = self._decode_cameras(predictions)
        pointcloud = self._build_pointcloud(
            images, predictions, extrinsics, intrinsics
        )
        poses = self._build_poses(
            extrinsics, intrinsics, video_input, selection_indices
        )
        source_indices = video_input.get_source_frame_indices()[selection_indices]
        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={
                "model": self.name,
                "frames_processed": len(image_paths),
                "image_resolution": self.image_resolution,
                "preprocess_mode": self.preprocess_mode,
                "confidence_percentile": self.confidence_percentile,
                "upstream_revision": self.upstream_revision,
                "scale_status": "relative",
                "pose_convention": PoseConvention.CAMERA_TO_WORLD.value,
            },
            window_metadata={
                "window_id": window_id,
                "frame_start": int(source_indices[0]),
                "frame_end": int(source_indices[-1]),
                "frame_indices": source_indices.tolist(),
                "window_size": self.window_size,
                "window_overlap": self.window_overlap,
            },
        )

    def _load_images(self, image_paths: List[str]):
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        if self.preprocess_mode not in {"balanced", "max_size"}:
            raise ValueError(
                "VGGT-Omega preprocess_mode must be 'balanced' or 'max_size'"
            )
        return load_and_preprocess_images(
            image_paths,
            mode=self.preprocess_mode,
            image_resolution=self.image_resolution,
        ).to(self.device)

    def _run_inference(self, images):
        import torch

        with torch.inference_mode():
            return self.model(images)

    @staticmethod
    def _decode_cameras(predictions):
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if "pose_enc" not in predictions or "images" not in predictions:
            raise ValueError("VGGT-Omega outputs missing pose_enc/images")
        return encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )

    def _build_pointcloud(
        self, images, predictions, extrinsics, intrinsics
    ) -> PointCloud:
        if "depth" not in predictions or "depth_conf" not in predictions:
            raise ValueError("VGGT-Omega outputs missing depth/depth_conf")
        depth = self._to_numpy(predictions["depth"]).squeeze(0)
        confidence = self._to_numpy(predictions["depth_conf"]).squeeze(0)
        extrinsics_np = self._to_numpy(extrinsics).squeeze(0)
        intrinsics_np = self._to_numpy(intrinsics).squeeze(0)
        points = self._unproject_depth(depth, extrinsics_np, intrinsics_np)
        colors = self._extract_colors(images)

        points = points.reshape(-1, 3).astype(np.float32)
        confidence = confidence.reshape(-1).astype(np.float32)
        finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
        if np.any(finite):
            threshold = np.percentile(
                confidence[finite], self.confidence_percentile
            )
            finite &= confidence >= threshold
        finite &= confidence > 1e-5
        points = points[finite]
        confidence = confidence[finite]
        colors = colors[finite]

        if self.max_points is not None and len(points) > int(self.max_points):
            rng = np.random.default_rng(self.sample_seed)
            selected = rng.choice(
                len(points), size=int(self.max_points), replace=False
            )
            points = points[selected]
            confidence = confidence[selected]
            colors = colors[selected]
        return PointCloud(
            points=points,
            colors=colors,
            confidence=confidence,
            is_metric=False,
        )

    @classmethod
    def _build_poses(
        cls,
        extrinsics,
        intrinsics,
        video_input: VideoInput,
        selection_indices: List[int],
    ) -> CameraPoses:
        w2c = cls._to_numpy(extrinsics).squeeze(0)
        intrinsics_np = cls._to_numpy(intrinsics).squeeze(0)
        matrices = np.repeat(
            np.eye(4, dtype=np.float32)[None], len(w2c), axis=0
        )
        matrices[:, :3, :4] = w2c
        c2w = np.linalg.inv(matrices)
        selection = np.asarray(selection_indices, dtype=np.int64)
        return CameraPoses(
            poses=c2w,
            timestamps=video_input.get_frame_timestamps()[selection],
            intrinsics=intrinsics_np.astype(np.float32),
            frame_indices=video_input.get_source_frame_indices()[selection],
            pose_convention=PoseConvention.CAMERA_TO_WORLD,
        )

    @classmethod
    def _extract_colors(cls, images) -> np.ndarray:
        values = cls._to_numpy(images)
        if values.ndim == 5:
            values = values.squeeze(0)
        if values.ndim != 4:
            raise ValueError(f"Unsupported VGGT-Omega image shape: {values.shape}")
        if values.shape[1] == 3:
            values = np.transpose(values, (0, 2, 3, 1))
        return (np.clip(values, 0.0, 1.0) * 255).astype(np.uint8).reshape(-1, 3)

    @staticmethod
    def _unproject_depth(
        depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray
    ) -> np.ndarray:
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 3:
            raise ValueError(f"Unsupported VGGT-Omega depth shape: {depth.shape}")
        frame_count, height, width = depth.shape
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        x = np.broadcast_to(x[None], (frame_count, height, width))
        y = np.broadcast_to(y[None], (frame_count, height, width))
        fx = intrinsics[:, 0, 0, None, None]
        fy = intrinsics[:, 1, 1, None, None]
        cx = intrinsics[:, 0, 2, None, None]
        cy = intrinsics[:, 1, 2, None, None]
        camera_points = np.stack(
            (
                (x - cx) / fx * depth,
                (y - cy) / fy * depth,
                depth,
            ),
            axis=-1,
        )
        rotation = extrinsics[:, :3, :3]
        translation = extrinsics[:, :3, 3]
        return np.einsum(
            "sij,shwj->shwi",
            np.transpose(rotation, (0, 2, 1)),
            camera_points - translation[:, None, None, :],
        )

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _resolve_repo_dir(self, *, required: bool) -> Optional[Path]:
        configured = self.vggt_omega_path or os.environ.get("VGGT_OMEGA_PATH")
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(Path(__file__).resolve().parents[3] / "vggt-omega")
        for candidate in candidates:
            candidate = candidate.resolve()
            if (candidate / "vggt_omega").is_dir():
                return candidate
        if required:
            raise FileNotFoundError(
                "VGGT-Omega checkout not found. Set vggt_omega_path or "
                "VGGT_OMEGA_PATH."
            )
        return None

    def _verify_revision(self, repo_dir: Path) -> None:
        if not self.verify_upstream_revision:
            return
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Cannot verify VGGT-Omega revision in {repo_dir}"
            ) from exc
        actual = completed.stdout.strip()
        if actual != self.upstream_revision:
            raise RuntimeError(
                "VGGT-Omega checkout revision mismatch: "
                f"expected {self.upstream_revision}, found {actual}"
            )
