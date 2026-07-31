"""Depth Anything 3 Streaming model wrapper."""

from pathlib import Path
from typing import Optional
import logging
import os
import subprocess
import sys
import numpy as np
import yaml

from .base import BaseModel
from ..core.types import (
    ReconstructionResult,
    PointCloud,
    CameraPoses,
    PoseConvention,
    VideoInput,
)

logger = logging.getLogger(__name__)


class DA3StreamingModel(BaseModel):
    """
    Depth Anything 3 - Streaming variant.

    DA3-Streaming is designed for long ordered image sequences.  Its upstream
    default performs Sim(3) chunk alignment, so the exported reconstruction has
    a consistent but not guaranteed metric scale.

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
    outputs_metric_scale = False
    outputs_poses = True
    # Confidence is available in optional per-frame NPZ files, but the
    # authoritative combined PLY consumed by this adapter does not retain it.
    outputs_confidence = False
    supports_video_input = False

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.save_per_frame_depth = self.config.get("save_per_frame_depth", False)
        self.da3_path = self.config.get("da3_path")  # Path to DA3 installation
        self.da3_config_path = self.config.get("da3_config_path")
        self.weights_dir = self.config.get(
            "weights_dir", "weights/da3_streaming"
        )
        self.upstream_revision = self.config.get(
            "upstream_revision",
            "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
        )
        self.verify_upstream_revision = self.config.get(
            "verify_upstream_revision", True
        )
        self.da3_dir: Optional[Path] = None

    @classmethod
    def get_default_config(cls) -> dict:
        """Return default DA3-Streaming configuration."""
        return {
            "da3_path": None,  # Path to DA3 installation
            "da3_config_path": None,  # Path to DA3 config YAML
            "weights_dir": "weights/da3_streaming",
            "save_per_frame_depth": False,
            "device": "cuda",
            "upstream_revision": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
            "verify_upstream_revision": True,
        }

    def load(self, weights_path: Optional[Path] = None) -> None:
        """
        Load DA3-Streaming model.

        DA3 requires the repository to be installed separately.
        """
        self.da3_dir = self._resolve_da3_dir()
        config_path = self._resolve_da3_config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                f"DA3 config not found: {config_path}. "
                "Set da3_config_path in the model config."
            )
        self._verify_revision(self.da3_dir)
        self._is_loaded = True

    def reconstruct(
        self,
        video_input: VideoInput,
        output_dir: Path,
    ) -> ReconstructionResult:
        """
        Run DA3-Streaming reconstruction.

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
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {video_input.image_dir}")
        logger.info(f"DA3-Streaming: Processing {len(image_paths)} frames")

        # Expected DA3-Streaming outputs:
        # - output_dir/camera_poses.txt
        # - output_dir/intrinsic.txt
        # - output_dir/pcd/combined_pcd.ply

        self._run_da3_streaming(video_input.image_dir, output_dir)

        pointcloud_path = output_dir / "pcd" / "combined_pcd.ply"
        if not pointcloud_path.exists():
            pointcloud_path = output_dir / "combined_pcd.ply"
        pointcloud = self._load_combined_ply(pointcloud_path)
        intrinsics = self._load_intrinsics(output_dir / "intrinsic.txt")
        timestamps = None
        if video_input.frame_count > 0:
            timestamps = video_input.get_frame_timestamps()
        poses = self._load_poses(
            output_dir / "camera_poses.txt",
            timestamps=timestamps,
            intrinsics=intrinsics,
            frame_indices=video_input.get_source_frame_indices(),
        )
        if poses is None:
            raise FileNotFoundError(
                f"DA3-Streaming did not produce camera poses in {output_dir}"
            )
        if len(poses) != len(image_paths):
            raise ValueError(
                "DA3-Streaming output count does not match the input: "
                f"{len(poses)} poses for {len(image_paths)} images"
            )

        return ReconstructionResult(
            pointcloud=pointcloud,
            poses=poses,
            metadata={
                "model": "da3_streaming",
                "frames": len(image_paths),
                "output_dir": str(output_dir),
                "config_path": str(self._resolve_da3_config_path()),
                "upstream_revision": self.upstream_revision,
                "scale_status": "relative",
                "pose_convention": PoseConvention.CAMERA_TO_WORLD.value,
            },
        )

    def _run_da3_streaming(self, image_dir: Path, output_dir: Path) -> None:
        """Run DA3-Streaming inference."""
        da3_dir = self.da3_dir or self._resolve_da3_dir()
        config_path = self._materialize_runtime_config(output_dir)

        cmd = [
            sys.executable,
            "da3_streaming.py",
            "--image_dir",
            str(image_dir),
            "--config",
            str(config_path),
            "--output_dir",
            str(output_dir),
        ]
        logger.info("Running DA3-Streaming: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=da3_dir, check=True)

    def _materialize_runtime_config(self, output_dir: Path) -> Path:
        """Write an upstream config with absolute paths to Mapper-owned weights."""
        source = self._resolve_da3_config_path()
        payload = yaml.safe_load(source.read_text()) or {}
        weights = payload.get("Weights")
        if not isinstance(weights, dict):
            raise ValueError(f"DA3 config has no Weights mapping: {source}")

        for name, value in weights.items():
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = self.resolve_workspace_path(path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"DA3 {name} weight not found: {path}. Run "
                    "`scripts/setup_models/download_models.sh da3-streaming`."
                )
            weights[name] = str(path)

        runtime_config = Path(output_dir) / "da3_streaming.resolved.yaml"
        runtime_config.write_text(yaml.safe_dump(payload, sort_keys=False))
        return runtime_config

    def _load_combined_ply(self, ply_path: Path) -> PointCloud:
        """Load the combined point cloud from DA3 output."""
        if not ply_path.exists():
            raise FileNotFoundError(
                f"DA3-Streaming combined point cloud not found: {ply_path}"
            )
        pointcloud = PointCloud.from_ply(ply_path)
        pointcloud.is_metric = False
        return pointcloud

    def _load_poses(
        self,
        poses_path: Path,
        timestamps: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        frame_indices: Optional[np.ndarray] = None,
    ) -> Optional[CameraPoses]:
        """Load camera poses from DA3 output."""
        if not poses_path.exists():
            logger.warning("DA3 camera poses not found: %s", poses_path)
            return None

        poses = []
        with poses_path.open("r") as handle:
            for line_idx, line in enumerate(handle):
                values = [float(x) for x in line.strip().split()]
                if len(values) != 16:
                    raise ValueError(
                        f"Invalid pose line {line_idx} in {poses_path}: "
                        f"expected 16 values, got {len(values)}"
                    )
                poses.append(np.array(values, dtype=np.float32).reshape(4, 4))

        if not poses:
            return None

        poses_array = np.stack(poses, axis=0)
        timestamps = self._align_timestamps(timestamps, poses_array.shape[0])
        intrinsics = self._align_intrinsics(intrinsics, poses_array.shape[0])
        if frame_indices is None:
            frame_indices = np.arange(poses_array.shape[0], dtype=np.int64)
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if len(frame_indices) != len(poses_array):
            raise ValueError("DA3 frame indices and poses have different lengths")

        return CameraPoses(
            poses=poses_array,
            timestamps=timestamps,
            intrinsics=intrinsics,
            frame_indices=frame_indices,
            pose_convention=PoseConvention.CAMERA_TO_WORLD,
        )

    def _load_intrinsics(self, intrinsics_path: Path) -> Optional[np.ndarray]:
        """Load camera intrinsics from DA3 output."""
        if not intrinsics_path.exists():
            logger.warning("DA3 intrinsics not found: %s", intrinsics_path)
            return None

        intrinsics = []
        with intrinsics_path.open("r") as handle:
            for line_idx, line in enumerate(handle):
                values = [float(x) for x in line.strip().split()]
                if len(values) < 4:
                    raise ValueError(
                        f"Invalid intrinsics line {line_idx} in {intrinsics_path}: "
                        f"expected 4 values, got {len(values)}"
                    )
                fx, fy, cx, cy = values[:4]
                intrinsics.append(
                    np.array(
                        [
                            [fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    )
                )

        if not intrinsics:
            return None

        return np.stack(intrinsics, axis=0)

    def _resolve_da3_dir(self) -> Path:
        if self.da3_path is not None:
            candidate = Path(self.da3_path).expanduser().resolve()
            da3_dir = self._locate_da3_dir(candidate)
            if da3_dir is not None:
                return da3_dir
            raise FileNotFoundError(
                f"da3_path does not contain da3_streaming.py: {candidate}"
            )

        env_value = os.environ.get("DA3_STREAMING_PATH")
        if env_value:
            env_path = Path(env_value).expanduser()
            da3_dir = self._locate_da3_dir(env_path.resolve())
            if da3_dir is not None:
                return da3_dir

        repos_root = Path(__file__).resolve().parents[3]
        for sibling_name in ("Depth-Anything-3", "da3-streaming"):
            da3_dir = self._locate_da3_dir(repos_root / sibling_name)
            if da3_dir is not None:
                return da3_dir

        raise FileNotFoundError(
            "DA3-Streaming repo not found. Set da3_path or DA3_STREAMING_PATH."
        )

    def _locate_da3_dir(self, base_path: Path) -> Optional[Path]:
        if (base_path / "da3_streaming.py").exists():
            return base_path
        candidate = base_path / "da3_streaming"
        if (candidate / "da3_streaming.py").exists():
            return candidate
        return None

    def _verify_revision(self, da3_dir: Path) -> None:
        if not self.verify_upstream_revision:
            return
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=da3_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Cannot verify Depth-Anything-3 revision in {da3_dir}"
            ) from exc
        actual = completed.stdout.strip()
        if actual != self.upstream_revision:
            raise RuntimeError(
                "Depth-Anything-3 checkout revision mismatch: "
                f"expected {self.upstream_revision}, found {actual}"
            )

    def _resolve_da3_config_path(self) -> Path:
        if self.da3_config_path is not None:
            config_path = Path(self.da3_config_path).expanduser()
            if not config_path.is_absolute():
                candidates = []
                cwd = Path.cwd()
                candidates.append(cwd / config_path)
                repo_root = Path(__file__).resolve().parents[2]
                if repo_root != cwd:
                    candidates.append(repo_root / config_path)
                if self.da3_dir is not None:
                    candidates.append(self.da3_dir / config_path)
                for candidate in candidates:
                    if candidate.exists():
                        return candidate.resolve()
                config_path = (self.da3_dir or cwd) / config_path
            return config_path.resolve()
        return (
            (self.da3_dir or self._resolve_da3_dir()) / "configs" / "base_config.yaml"
        )

    def _align_timestamps(
        self,
        timestamps: Optional[np.ndarray],
        pose_count: int,
    ) -> Optional[np.ndarray]:
        if timestamps is None:
            return None
        if len(timestamps) >= pose_count:
            return timestamps[:pose_count]
        logger.warning(
            "DA3 timestamps length %s shorter than pose count %s; dropping timestamps",
            len(timestamps),
            pose_count,
        )
        return None

    def _align_intrinsics(
        self,
        intrinsics: Optional[np.ndarray],
        pose_count: int,
    ) -> Optional[np.ndarray]:
        if intrinsics is None:
            return None
        if intrinsics.shape[0] == pose_count:
            return intrinsics
        if intrinsics.shape[0] == 1 and pose_count > 1:
            return np.repeat(intrinsics, pose_count, axis=0)
        if intrinsics.shape[0] > pose_count:
            return intrinsics[:pose_count]
        logger.warning(
            "DA3 intrinsics length %s shorter than pose count %s; dropping intrinsics",
            intrinsics.shape[0],
            pose_count,
        )
        return None
